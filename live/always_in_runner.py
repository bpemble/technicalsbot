# ============================================================
# live/always_in_runner.py — Multi-asset always-in regime trader
#
# Philosophy:
#   The bot maintains a portfolio of up to MAX_POSITIONS assets.
#   Each tick it scores every asset, selects the top-conviction
#   names, and keeps the portfolio aligned to those positions.
#   Conviction fades → trim/close. Regime flips → flip position.
#   Hard stop circuit breaker at 8% adverse move regardless of score.
# ============================================================

import json
import os
import time
import signal
import sys
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from utils import fmt_ts, fmt_now
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

import config
from data.fetcher import (
    fetch_funding_and_oi, fetch_funding_rates, fetch_latest_prices,
    fetch_fear_and_greed, DataCache,
)
from indicators.compute import add_indicators
from strategy.regime import RegimeEngine
from live.paper_wallet import PaperWallet

_log_handler = RotatingFileHandler(
    config.LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
)
_log_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
logging.basicConfig(level=logging.INFO, handlers=[_log_handler])
logger = logging.getLogger(__name__)

console = Console()
_running = True

HARD_STOP_PCT       = 0.08   # 8% hard circuit breaker
REBALANCE_THRESHOLD = 0.25   # only resize if >25% off target


def apply_funding_adjustment(score: float, funding_rate: float) -> float:
    """
    Penalise score when funding is crowded in the same direction as the regime.

    Positive funding (longs pay shorts) + bullish score  → crowded long  → reduce score.
    Negative funding (shorts pay longs) + bearish score  → crowded short → reduce score.
    Funding in the *opposite* direction provides no bonus (we don't chase contrarian).
    """
    normalised = max(-1.0, min(1.0, funding_rate / config.FUNDING_NORM_RATE))
    if score > 0 and normalised > 0:
        return score - normalised * config.FUNDING_PENALTY_MAX
    if score < 0 and normalised < 0:
        return score + abs(normalised) * config.FUNDING_PENALTY_MAX
    # Opposite extreme funding — contrarian bonus (crowded in their direction, confirms ours)
    if abs(normalised) >= config.FUNDING_CONTRARIAN_THRESHOLD:
        if score > 0 and normalised < 0:
            return score + abs(normalised) * config.FUNDING_CONTRARIAN_BONUS
        if score < 0 and normalised > 0:
            return score - normalised * config.FUNDING_CONTRARIAN_BONUS
    return score


def apply_oi_adjustment(score: float, oi_change_pct: float) -> float:
    """
    Amplify or dampen score based on open-interest change since last tick.

    OI growing  → market participants adding conviction → amplify score.
    OI shrinking → positions being unwound → dampen score.

    oi_change_pct = (current_oi - prev_oi) / prev_oi.
    Clamped to ±OI_CHANGE_CLAMP before linear interpolation.
    """
    clamped = max(-config.OI_CHANGE_CLAMP, min(config.OI_CHANGE_CLAMP, oi_change_pct))
    factor  = clamped / config.OI_CHANGE_CLAMP   # maps to [-1, +1]
    if factor >= 0:
        return score * (1.0 + factor * config.OI_AMPLIFY_MAX)
    else:
        return score * (1.0 + factor * config.OI_DAMPEN_MAX)


def _direction_from_score(score: float) -> str:
    if score > config.MIN_CONVICTION_SCORE:
        return "long"
    if score < -config.MIN_CONVICTION_SCORE:
        return "short"
    return "flat"


def _handle_sigint(sig, frame):
    global _running
    console.print("\n[yellow]Shutting down...[/yellow]")
    _running = False


signal.signal(signal.SIGINT, _handle_sigint)


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

def _score_asset(
    asset: dict,
    cache: "DataCache",
    engine: "RegimeEngine",
    funding_rates: dict | None,
    oi_values: dict | None = None,
    prev_oi:   dict | None = None,
) -> tuple[str, dict | None]:
    """
    Fetch data (cached), compute indicators (cached), score regime for one asset.
    Module-level so it is picklable. ThreadPoolExecutor is used (not ProcessPool)
    because indicator caching eliminates the CPU bottleneck — I/O dominates.
    """
    name = asset["name"]
    coin = asset["hl"]
    try:
        df_1d  = cache.get_with_indicators(coin, "1d",  config.LOOKBACK_DAYS,       config)
        df_4h  = cache.get_with_indicators(coin, "4h",  config.LOOKBACK_DAYS,       config)
        df_1h  = cache.get_with_indicators(coin, "1h",  7,                          config)
        df_15m = cache.get_with_indicators(coin, "15m", config.SCALP_LOOKBACK_DAYS, config)
        snapshot = engine.compute(df_1d, df_4h, df_1h, df_15m)
        hl_name  = asset.get("hl", name)
        fr       = (funding_rates or {}).get(hl_name, 0.0)
        adj      = apply_funding_adjustment(snapshot.score, fr)

        # OI trend adjustment
        if oi_values and prev_oi:
            cur_oi  = oi_values.get(hl_name, 0.0)
            past_oi = prev_oi.get(hl_name, cur_oi)
            if past_oi > 0:
                oi_change = (cur_oi - past_oi) / past_oi
                adj = apply_oi_adjustment(adj, oi_change)

        return name, {
            "asset":        asset,
            "snapshot":     snapshot,
            "price":        snapshot.latest_price,
            "adj_score":    adj,
            "funding_rate": fr,
        }
    except Exception as e:
        logger.warning(f"{name}: {e}")
        return name, None


# ------------------------------------------------------------------
# Regime computation across all assets
# ------------------------------------------------------------------

def compute_all_regimes(
    engine: RegimeEngine,
    cache: DataCache,
    funding_rates: dict | None = None,
    oi_values: dict | None = None,
    prev_oi:   dict | None = None,
) -> list[dict]:
    """
    Loop over all config.ASSETS in parallel, fetch data, compute regime snapshot.

    funding_rates : optional {hl_name: hourly_rate} — funding crowding adjustment.
    oi_values     : optional {hl_name: open_interest} — current tick OI.
    prev_oi       : optional {hl_name: open_interest} — previous tick OI for delta.

    Returns list of dicts sorted by abs(adj_score) descending:
      [{"asset": ..., "snapshot": ..., "price": float,
        "adj_score": float, "funding_rate": float}, ...]
    """
    results = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_score_asset, asset, cache, engine, funding_rates, oi_values, prev_oi): asset
            for asset in config.ASSETS
        }
        for future in as_completed(futures):
            name, result = future.result()
            if result is not None:
                adj_score = result["adj_score"]
                sign = "+" if adj_score >= 0 else ""
                console.print(f"  {name}... [{'green' if adj_score >= 0 else 'red'}]{sign}{adj_score:.1f}[/{'green' if adj_score >= 0 else 'red'}]")
                results.append(result)
            else:
                console.print(f"  {name}... [dim]skip[/dim]")

    results.sort(key=lambda r: abs(r["adj_score"]), reverse=True)
    return results


# ------------------------------------------------------------------
# Position sizing
# ------------------------------------------------------------------

def target_position_size(score: float, equity: float, price: float) -> float:
    """
    Per-position half-Kelly sizing with portfolio notional cap.

    Each position is treated as an independent Kelly bet:
      risk_per_position = KELLY_FRACTION × equity × conviction
      notional          = risk_per_position / HARD_STOP_PCT

    A hard ceiling ensures total portfolio notional never exceeds
    MAX_NOTIONAL_FACTOR × equity regardless of conviction level.

    Example at 20% conviction, $10k, 10 positions:
      notional = $10k × 0.065 × 0.20 / 0.08 = $1,625/position (1.625× total)
      risk     = $1,300 total = 13% of equity

    Example at 80% conviction (cap kicks in):
      uncapped = $6,500/position → capped at $2,000 (2× total / 10 slots)
    """
    conviction  = abs(score) / 100.0
    notional    = equity * config.KELLY_FRACTION * conviction / HARD_STOP_PCT
    notional_cap = equity * config.MAX_NOTIONAL_FACTOR / config.MAX_POSITIONS
    return min(notional, notional_cap) / price if price > 0 else 0.0


# ------------------------------------------------------------------
# Correlation scaling
# ------------------------------------------------------------------

def _tier_correlation_factor(wallet: PaperWallet, asset_name: str, asset_tier: int, target_dir: str) -> float:
    """Scale position size down for each additional same-tier same-direction position."""
    count = sum(
        1 for name, pos in wallet.positions.items()
        if name != asset_name
        and pos["direction"] == target_dir
        and any(a["name"] == name and a["tier"] == asset_tier for a in config.ASSETS)
    )
    return config.TIER_CORR_FACTOR ** count


# ------------------------------------------------------------------
# Additional position-sizing factors
# ------------------------------------------------------------------

def _btc_size_factor(regimes: list[dict], asset_name: str, target_dir: str) -> float:
    """
    BTC regime gate: scale size up when BTC aligns, down when opposed.

    BTC is the market-wide risk barometer. Entering a long while BTC is
    bearish — or a short while BTC is bullish — materially reduces edge.

    Returns a multiplier in [0.5, 1.2].
    """
    if asset_name == "BTC":
        return 1.0
    btc_entry = next((r for r in regimes if r["asset"]["name"] == "BTC"), None)
    if btc_entry is None:
        return 1.0
    btc_score = btc_entry["adj_score"]
    aligned = (
        (target_dir == "long"  and btc_score >  config.MIN_CONVICTION_SCORE) or
        (target_dir == "short" and btc_score < -config.MIN_CONVICTION_SCORE)
    )
    opposed = (
        (target_dir == "long"  and btc_score < -config.MIN_CONVICTION_SCORE) or
        (target_dir == "short" and btc_score >  config.MIN_CONVICTION_SCORE)
    )
    if aligned:
        strength = min(abs(btc_score) / 100.0, 1.0)
        return 1.0 + 0.2 * strength   # up to 1.2
    if opposed:
        return 0.5
    return 1.0   # BTC flat


def _vol_size_factor(snapshot) -> float:
    """
    Low-volatility brake: reduce size when ATR percentile is unusually low.

    Thin-volatility markets have weak follow-through; high conviction in a
    quiet market often means the signal hasn't been tested yet.

    Returns multiplier in [VOL_REGIME_MIN, 1.0].
    """
    pct = snapshot.norm_atr_pct
    if pct <= 0 or (isinstance(pct, float) and np.isnan(pct)):
        return 1.0
    if pct >= config.VOL_REGIME_LOW:
        return 1.0
    # Linear: VOL_REGIME_MIN at pct=0, 1.0 at pct=VOL_REGIME_LOW
    return config.VOL_REGIME_MIN + (1.0 - config.VOL_REGIME_MIN) * (pct / config.VOL_REGIME_LOW)


def _ma200_size_factor(snapshot) -> float:
    """
    MA200 extension brake: penalise entries when price is stretched far from MA200.

    Price well beyond its 200-bar mean is statistically prone to mean reversion;
    reduce size to limit exposure to snap-backs.

    Returns multiplier in [0.5, 1.0].
    """
    dist = abs(snapshot.ma200_dist)
    if dist <= 0 or (isinstance(dist, float) and np.isnan(dist)):
        return 1.0
    if dist <= config.MA200_NEAR_BAND:
        return 1.0
    if dist >= config.MA200_FAR_BAND:
        return 0.5
    frac = (dist - config.MA200_NEAR_BAND) / (config.MA200_FAR_BAND - config.MA200_NEAR_BAND)
    return 1.0 - 0.5 * frac


# ------------------------------------------------------------------
# Hard stops
# ------------------------------------------------------------------

def check_hard_stops(wallet: PaperWallet, price_lookup: dict[str, float]):
    """
    Close any open position whose stored stop_loss price has been breached.

    price_lookup : {asset_name: current_price} — accepts output from either
                   fetch_latest_prices() (fast loop) or regime price data (slow loop).
    """
    for asset_name, pos in list(wallet.positions.items()):
        current_price = price_lookup.get(asset_name)
        if current_price is None:
            continue

        direction  = pos["direction"]
        trail_atr  = pos.get("trail_atr", 0.0)

        # Update trailing stop when price moves favourably
        if trail_atr > 0:
            activation_dist = trail_atr * config.TRAIL_ACTIVATION_ATR
            trail_dist      = trail_atr * config.TRAIL_ATR_MULTIPLIER
            entry_price     = pos["entry_price"]
            cur_trail       = pos.get("trailing_stop")
            if direction == "long":
                if current_price - entry_price >= activation_dist:
                    candidate = current_price - trail_dist
                    if cur_trail is None or candidate > cur_trail:
                        wallet.update_trailing_stop(asset_name, candidate)
            else:
                if entry_price - current_price >= activation_dist:
                    candidate = current_price + trail_dist
                    if cur_trail is None or candidate < cur_trail:
                        wallet.update_trailing_stop(asset_name, candidate)

        pos = wallet.get_position(asset_name)
        if pos is None:
            continue

        hard_stop     = pos["stop_loss"]
        trailing_stop = pos.get("trailing_stop")
        if trailing_stop is not None:
            effective_stop = (max(hard_stop, trailing_stop) if direction == "long"
                              else min(hard_stop, trailing_stop))
        else:
            effective_stop = hard_stop

        hit = (
            (direction == "long"  and current_price <= effective_stop) or
            (direction == "short" and current_price >= effective_stop)
        )
        if not hit:
            continue

        is_trail    = trailing_stop is not None and effective_stop == trailing_stop
        exit_reason = "trail_stop" if is_trail else "hard_stop"
        fee         = pos["size"] * current_price * config.EXCHANGE_FEE
        trade       = wallet.close_position(asset_name, current_price, exit_reason, fee)
        if trade:
            entry    = pos["entry_price"]
            move_pct = ((current_price - entry) / entry * 100 if direction == "long"
                        else (entry - current_price) / entry * 100)
            pnl_color = "green" if trade["net_pnl"] >= 0 else "red"
            style     = "red bold" if exit_reason == "hard_stop" else pnl_color
            msg = (
                f"{exit_reason.upper()} {asset_name} {direction.upper()} "
                f"entered @ ${entry:,.2f}, now ${current_price:,.2f} ({move_pct:.1f}%)  "
                f"PnL: ${trade['net_pnl']:+,.2f}"
            )
            console.print(f"  [{style}]{msg}[/{style}]")
            logger.info(msg)


# ------------------------------------------------------------------
# Portfolio rebalancing
# ------------------------------------------------------------------

def _open_new_position(
    wallet: PaperWallet,
    asset_name: str,
    target_dir: str,
    current_price: float,
    want_size: float,
    trail_atr: float,
) -> float:
    """Open a paper position with slippage, fees, ATR-based stop, and trail metadata. Returns fill price."""
    fill = current_price * (1 + config.SLIPPAGE if target_dir == "long" else 1 - config.SLIPPAGE)
    fee  = want_size * fill * config.EXCHANGE_FEE

    # ATR-based stop: tighter on low-volatility assets, capped at HARD_STOP_PCT on wild alts.
    # TP at 3× stop distance — trailing stop and regime exit do the real work in trends.
    if trail_atr > 0:
        stop_dist = min(trail_atr * config.ATR_STOP_MULTIPLIER, fill * HARD_STOP_PCT)
    else:
        stop_dist = fill * HARD_STOP_PCT
    tp_dist = stop_dist * 3

    wallet.open_position(
        strategy    = asset_name,
        direction   = target_dir,
        entry_price = fill,
        size        = want_size,
        stop_loss   = (fill - stop_dist) if target_dir == "long" else (fill + stop_dist),
        take_profit = (fill + tp_dist)   if target_dir == "long" else (fill - tp_dist),
        fee         = fee,
        trail_atr   = trail_atr,
    )
    return fill


def rebalance_portfolio(wallet: PaperWallet, regimes: list[dict], fng_value: int = -1):
    """
    1. Select top MAX_POSITIONS assets where abs(score) > MIN_CONVICTION_SCORE.
    2. Close positions not in that active set.
    3. Open / flip / resize positions for each active asset.

    fng_value : Fear & Greed Index (0–100), or -1 if unavailable.
                ≥ FNG_EXTREME_GREED → cut max positions and penalise longs.
    """
    # Apply F&G position cap
    effective_max = config.MAX_POSITIONS
    if fng_value >= config.FNG_EXTREME_GREED:
        effective_max = max(1, config.MAX_POSITIONS - config.FNG_POSITIONS_REDUCE)

    # Build active set from sorted regimes
    active_set: dict[str, dict] = {}
    for entry in regimes:
        if len(active_set) >= effective_max:
            break
        score = entry["adj_score"]
        # Apply F&G long penalty during extreme greed
        if fng_value >= config.FNG_EXTREME_GREED and score > 0:
            score = score - config.FNG_GREED_LONG_PENALTY
        if abs(score) > config.MIN_CONVICTION_SCORE:
            asset_name = entry["asset"]["name"]
            active_set[asset_name] = entry

    # Close positions whose asset is no longer in the active set
    for asset_name in list(wallet.positions.keys()):
        if asset_name not in active_set:
            pos           = wallet.get_position(asset_name)
            current_price = pos["entry_price"]  # fallback; ideally from regimes
            # Try to get a real price from any available regime entry
            for r in regimes:
                if r["asset"]["name"] == asset_name:
                    current_price = r["price"]
                    break
            fee   = pos["size"] * current_price * config.EXCHANGE_FEE
            trade = wallet.close_position(asset_name, current_price, "regime_exit", fee)
            if trade:
                pnl_color = "green" if trade["net_pnl"] >= 0 else "red"
                msg = f"CLOSED {asset_name} (not in active set)  PnL: ${trade['net_pnl']:+,.2f}"
                console.print(f"  [{pnl_color}]{msg}[/{pnl_color}]")
                logger.info(msg)

    # Compute equity for position sizing
    price_lookup = {r["asset"]["name"]: r["price"] for r in regimes}
    upnl   = wallet.total_unrealized_pnl(price_lookup)
    equity = wallet.capital + upnl

    # Manage each active position
    for asset_name, entry in active_set.items():
        snapshot      = entry["snapshot"]
        current_price = entry["price"]
        score         = entry["adj_score"]
        target_dir    = _direction_from_score(score)

        if target_dir == "flat":
            pos = wallet.get_position(asset_name)
            if pos is not None:
                fee   = pos["size"] * current_price * config.EXCHANGE_FEE
                trade = wallet.close_position(asset_name, current_price, "regime_flat", fee)
                if trade:
                    msg = f"CLOSED {asset_name} (regime flat)  PnL: ${trade['net_pnl']:+,.2f}"
                    console.print(f"  [yellow]{msg}[/yellow]")
                    logger.info(msg)
            continue

        corr_factor  = _tier_correlation_factor(wallet, asset_name, entry["asset"]["tier"], target_dir)
        btc_factor   = _btc_size_factor(regimes, asset_name, target_dir)
        vol_factor   = _vol_size_factor(snapshot)
        ma200_factor = _ma200_size_factor(snapshot)
        want_size = (
            target_position_size(score, equity, current_price)
            * corr_factor * btc_factor * vol_factor * ma200_factor
        )
        if want_size <= 0:
            continue

        pos = wallet.get_position(asset_name)

        # No position — open one
        if pos is None:
            if want_size <= 0:
                continue
            trail_atr = snapshot.latest_atr
            fill = _open_new_position(wallet, asset_name, target_dir, current_price, want_size, trail_atr)
            dir_color = "cyan" if target_dir == "long" else "magenta"
            msg = (
                f"OPENED {asset_name} {target_dir.upper()} "
                f"@ ${fill:,.2f}  size={want_size:.4f}  score={score:+.1f}"
                + (f"  corr={corr_factor:.2f}" if corr_factor < 1.0 else "")
            )
            console.print(f"  [{dir_color}]{msg}[/{dir_color}]")
            logger.info(msg)
            continue

        current_dir  = pos["direction"]
        current_size = pos["size"]

        # Wrong direction — flip
        if current_dir != target_dir:
            fee   = current_size * current_price * config.EXCHANGE_FEE
            trade = wallet.close_position(asset_name, current_price, "regime_flip", fee)
            if trade:
                pnl_color = "green" if trade["net_pnl"] >= 0 else "red"
                flip_msg = (
                    f"FLIPPED {asset_name} "
                    f"{current_dir.upper()} → {target_dir.upper()} "
                    f"@ ${current_price:,.2f}  PnL: ${trade['net_pnl']:+,.2f}"
                )
                console.print(f"  [{pnl_color}]{flip_msg}[/{pnl_color}]")
                logger.info(flip_msg)
            trail_atr = snapshot.latest_atr
            fill = _open_new_position(wallet, asset_name, target_dir, current_price, want_size, trail_atr)
            dir_color = "cyan" if target_dir == "long" else "magenta"
            open_msg = (
                f"OPENED {asset_name} {target_dir.upper()} "
                f"@ ${fill:,.2f}  size={want_size:.4f}  score={score:+.1f}"
            )
            console.print(f"  [{dir_color}]{open_msg}[/{dir_color}]")
            logger.info(open_msg)
            continue

        # Same direction — resize if materially different
        size_diff_pct = abs(want_size - current_size) / max(current_size, 1e-9)
        if size_diff_pct > REBALANCE_THRESHOLD:
            fee = current_size * current_price * config.EXCHANGE_FEE
            wallet.close_position(asset_name, current_price, "resize", fee)
            trail_atr = snapshot.latest_atr
            fill = _open_new_position(wallet, asset_name, target_dir, current_price, want_size, trail_atr)
            resize_msg = (
                f"RESIZED {asset_name} {target_dir.upper()}  "
                f"{current_size:.4f} → {want_size:.4f}  score={score:+.1f}"
            )
            console.print(f"  [dim]{resize_msg}[/dim]")
            logger.info(resize_msg)


# ------------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------------

def score_bar(score: float, width: int = 20) -> str:
    """Render a centred ASCII bar showing score magnitude and direction."""
    half   = width // 2
    filled = min(int(abs(score) / 100 * half), half)
    empty  = half - filled
    if score >= 0:
        left  = " " * half
        right = (f"[green]{'█' * filled}[/green]" if filled else "") + " " * empty
    else:
        left  = " " * empty + (f"[red]{'█' * filled}[/red]" if filled else "")
        right = " " * half
    return left + "|" + right


def print_portfolio(regimes: list[dict], wallet: PaperWallet):
    """Print the multi-asset regime table and open positions table."""

    # ---- Table 1: All Assets Regime Scores -----------------------
    t = Table(
        title="[bold cyan]All Assets — Regime Scores[/bold cyan]",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title_style="bold cyan",
    )
    t.add_column("Asset",      min_width=6)
    t.add_column("Tier",       min_width=5,  justify="center")
    t.add_column("Price",      min_width=12, justify="right")
    t.add_column("Score",      min_width=8,  justify="right")
    t.add_column("Direction",  min_width=10, justify="center")
    t.add_column("Conviction", min_width=10, justify="right")
    t.add_column("Bar",        min_width=22)
    t.add_column("Funding/hr", min_width=10, justify="right")
    t.add_column("Position",   min_width=8,  justify="center")

    open_position_keys = set(wallet.positions.keys())

    for i, entry in enumerate(regimes):
        # Visual separator after 10th row to mark MAX_POSITIONS cutoff
        if i == config.MAX_POSITIONS:
            t.add_section()

        asset     = entry["asset"]
        price     = entry["price"]
        score     = entry["adj_score"]
        direction = _direction_from_score(score)
        conv      = abs(score) / 100.0
        fr        = entry.get("funding_rate", 0.0)

        name = asset["name"]
        tier = str(asset["tier"])

        score_str = (
            f"[green]{score:+.1f}[/green]" if score > 0
            else f"[red]{score:+.1f}[/red]" if score < 0
            else f"[dim]{score:+.1f}[/dim]"
        )

        if direction == "long":
            dir_str = "[cyan]LONG[/cyan]"
        elif direction == "short":
            dir_str = "[magenta]SHORT[/magenta]"
        else:
            dir_str = "[dim yellow]FLAT[/dim yellow]"

        has_pos  = name in open_position_keys
        pos_str  = "[bold green]OPEN[/bold green]" if has_pos else "[dim]—[/dim]"

        price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"

        fr_pct = fr * 100
        if abs(fr_pct) < 0.001:
            fr_str = "[dim]—[/dim]"
        elif fr_pct > 0:
            fr_str = f"[yellow]+{fr_pct:.3f}%[/yellow]"
        else:
            fr_str = f"[cyan]{fr_pct:.3f}%[/cyan]"

        t.add_row(
            f"[bold]{name}[/bold]",
            tier,
            price_str,
            score_str,
            dir_str,
            f"{conv*100:.0f}%",
            score_bar(score, 20),
            fr_str,
            pos_str,
        )

    console.print()
    console.print(t)

    # ---- Table 2: Open Positions ----------------------------------
    positions = wallet.positions
    if positions:
        price_lookup = {r["asset"]["name"]: r["price"] for r in regimes}

        pt = Table(
            title="[bold cyan]Open Positions[/bold cyan]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            title_style="bold cyan",
        )
        pt.add_column("Asset",      min_width=6)
        pt.add_column("Dir",        min_width=6, justify="center")
        pt.add_column("Entry $",    min_width=12, justify="right")
        pt.add_column("Current $",  min_width=12, justify="right")
        pt.add_column("Size",       min_width=10, justify="right")
        pt.add_column("Unreal PnL", min_width=12, justify="right")
        pt.add_column("Stop",       min_width=12, justify="right")
        pt.add_column("Entry Time", min_width=17)

        for asset_name, pos in positions.items():
            dir_color    = "cyan" if pos["direction"] == "long" else "magenta"
            current_price = price_lookup.get(asset_name, pos["entry_price"])

            if pos["direction"] == "long":
                upnl = (current_price - pos["entry_price"]) * pos["size"]
            else:
                upnl = (pos["entry_price"] - current_price) * pos["size"]

            upnl_color = "green" if upnl >= 0 else "red"
            entry_price = pos["entry_price"]
            cur_str = (
                f"${current_price:,.4f}" if current_price < 1
                else f"${current_price:,.2f}"
            )
            entry_str = (
                f"${entry_price:,.4f}" if entry_price < 1
                else f"${entry_price:,.2f}"
            )

            entry_time_str = fmt_ts(pos.get("entry_time"))

            trailing_stop = pos.get("trailing_stop")
            if trailing_stop is not None:
                stop_cell = f"[yellow]~${trailing_stop:,.2f}[/yellow]"
            else:
                stop_cell = f"${pos['stop_loss']:,.2f}"

            pt.add_row(
                f"[bold]{asset_name}[/bold]",
                f"[{dir_color}]{pos['direction'].upper()}[/{dir_color}]",
                entry_str,
                cur_str,
                f"{pos['size']:.4f}",
                f"[{upnl_color}]${upnl:+,.2f}[/{upnl_color}]",
                stop_cell,
                entry_time_str,
            )

        console.print()
        console.print(pt)

    # ---- Summary line --------------------------------------------
    price_map = {r["asset"]["name"]: r["price"] for r in regimes}
    total_upnl = wallet.total_unrealized_pnl(price_map)
    equity     = wallet.capital + total_upnl
    ic         = wallet.state.get("initial_capital", config.PAPER_CAPITAL)
    ret        = (equity - ic) / ic * 100
    ret_color  = "green" if ret >= 0 else "red"
    upnl_color = "green" if total_upnl >= 0 else "red"

    console.print(
        f"\n  Equity: [bold]${equity:,.2f}[/bold]  "
        f"Return: [{ret_color}]{ret:+.2f}%[/{ret_color}]  "
        f"Unrealised: [{upnl_color}]${total_upnl:+,.2f}[/{upnl_color}]  "
        f"Closed trades: [bold]{len(wallet.trades)}[/bold]  "
        f"Open positions: [bold]{len(wallet.positions)}[/bold]"
    )
    console.print()


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def _charge_funding(wallet: PaperWallet, regimes: list[dict], funding_rates: dict):
    """Charge pro-rated hourly funding on all open positions once per scan tick."""
    if not funding_rates:
        return
    hours_elapsed = config.POLL_INTERVAL_SEC / 3600
    price_map   = {r["asset"]["name"]: r["price"]        for r in regimes}
    hl_name_map = {r["asset"]["name"]: r["asset"]["hl"]  for r in regimes}
    for asset_name in list(wallet.positions.keys()):
        pos = wallet.get_position(asset_name)
        if pos is None:
            continue
        hl_name      = hl_name_map.get(asset_name, asset_name)
        hourly_rate  = funding_rates.get(hl_name, 0.0)
        current_price = price_map.get(asset_name, pos["entry_price"])
        notional     = pos["size"] * current_price
        charge       = notional * abs(hourly_rate) * hours_elapsed
        pays = (
            (pos["direction"] == "long"  and hourly_rate > 0) or
            (pos["direction"] == "short" and hourly_rate < 0)
        )
        if pays and charge > 0:
            wallet.charge_funding(asset_name, charge)


def _save_equity_snapshot(wallet: "PaperWallet", regimes: list[dict]):
    """Append current equity to equity_history.jsonl for the dashboard chart."""
    price_map = {r["asset"]["name"]: r["price"] for r in regimes}
    upnl      = wallet.total_unrealized_pnl(price_map)
    equity    = wallet.capital + upnl
    entry     = {"t": datetime.now(tz=timezone.utc).isoformat(), "v": round(equity, 2)}
    with open(config.EQUITY_HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _save_scores(regimes: list[dict], tick: int):
    """Persist latest regime scores to scores.json for the dashboard."""
    data = {
        "tick":          tick,
        "timestamp":     fmt_now(),
        "timestamp_iso": datetime.now(tz=timezone.utc).isoformat(),
        "assets": [
            {
                "name":         r["asset"]["name"],
                "tier":         r["asset"]["tier"],
                "price":        r["price"],
                "score":        round(r["adj_score"], 2),
                "direction":    _direction_from_score(r["adj_score"]),
                "conviction":   round(abs(r["adj_score"]) / 100, 4),
                "funding_rate": round(r.get("funding_rate", 0.0), 6),
            }
            for r in regimes
        ],
    }
    tmp = config.SCORES_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, config.SCORES_FILE)


def run():
    console.print()
    console.print(Panel(
        "[bold cyan]Multi-Asset Regime Trader — Paper Mode[/bold cyan]\n"
        "[dim]Always-in conviction-weighted portfolio[/dim]\n"
        f"[dim]Capital: ${config.PAPER_CAPITAL:,.0f}  |  "
        f"Kelly: {config.KELLY_FRACTION*100:.1f}% (half)  |  "
        f"Hard stop: {HARD_STOP_PCT*100:.0f}%  |  "
        f"Max positions: {config.MAX_POSITIONS}  |  "
        f"Assets: {len(config.ASSETS)}  |  "
        f"Regime scan: {config.POLL_INTERVAL_SEC//60}min  |  "
        f"Stop check: {config.FAST_POLL_SEC}s[/dim]",
        border_style="cyan", expand=False,
    ))
    console.print()

    wallet = PaperWallet(config.REGIME_STATE_FILE, config.PAPER_CAPITAL)
    engine = RegimeEngine(config)
    cache  = DataCache()

    console.print(
        f"  Wallet loaded — cash: [bold]${wallet.capital:,.2f}[/bold]  "
        f"closed trades: [bold]{len(wallet.trades)}[/bold]  "
        f"open positions: [bold]{len(wallet.positions)}[/bold]"
    )
    console.print()

    # ---- Warm cache (full history fetch, done once at startup) -----------
    console.print("  [dim]Warming data cache — fetching full history from Hyperliquid...[/dim]")
    cache.warm(config.ASSETS, {
        "1d":  config.LOOKBACK_DAYS,
        "4h":  config.LOOKBACK_DAYS,
        "1h":  7,
        "15m": config.SCALP_LOOKBACK_DAYS,
    }, config)
    console.print("  [green]Cache warmed.[/green]")
    console.print()

    tick             = 0
    regimes:   list  = []    # kept in scope so fast loop can reference open positions
    prev_oi:   dict  = {}    # OI snapshot from previous tick for delta computation
    fng_value: int   = -1    # cached F&G index
    fng_last_refresh = 0.0   # time.time() of last F&G fetch

    while _running:
        # ---- Full regime scan ----------------------------------------
        tick += 1
        console.rule(f"[dim]Tick {tick}  —  {fmt_now()}[/dim]")
        console.print()

        console.print("  [dim]Refreshing data, funding rates, and OI...[/dim]")
        funding_rates, oi_values = fetch_funding_and_oi()

        # Refresh Fear & Greed (at most once per FNG_CACHE_TTL_SEC)
        now_ts = time.time()
        if now_ts - fng_last_refresh >= config.FNG_CACHE_TTL_SEC:
            fng_value        = fetch_fear_and_greed()
            fng_last_refresh = now_ts
            if fng_value >= 0:
                if fng_value <= config.FNG_EXTREME_FEAR:
                    fng_label = f"[green]F&G={fng_value} (extreme fear — contrarian open)[/green]"
                elif fng_value >= config.FNG_EXTREME_GREED:
                    fng_label = f"[red]F&G={fng_value} (extreme greed — reducing longs)[/red]"
                else:
                    fng_label = f"[dim]F&G={fng_value}[/dim]"
                console.print(f"  Fear & Greed Index: {fng_label}")

        regimes = compute_all_regimes(engine, cache, funding_rates, oi_values, prev_oi)
        prev_oi = oi_values   # store for next tick's delta

        if not regimes:
            console.print("  [red]No asset data available this tick — skipping.[/red]")
        else:
            price_lookup = {r["asset"]["name"]: r["price"] for r in regimes}
            check_hard_stops(wallet, price_lookup)
            rebalance_portfolio(wallet, regimes, fng_value)
            print_portfolio(regimes, wallet)
            _save_scores(regimes, tick)
            _save_equity_snapshot(wallet, regimes)
            _charge_funding(wallet, regimes, funding_rates)

        if not _running:
            break

        console.print(
            f"  [dim]Next regime scan in {config.POLL_INTERVAL_SEC // 60} min  |  "
            f"Stop checks every {config.FAST_POLL_SEC} s  |  Ctrl+C to stop.[/dim]\n"
        )

        # ---- Fast stop-check loop between regime scans ---------------
        scan_start = time.time()
        while _running and (time.time() - scan_start) < config.POLL_INTERVAL_SEC:
            time.sleep(config.FAST_POLL_SEC)
            if not _running:
                break

            prices = fetch_latest_prices()
            if not prices:
                continue

            if wallet.positions:
                check_hard_stops(wallet, prices)

            time_remaining = int(config.POLL_INTERVAL_SEC - (time.time() - scan_start))
            if wallet.positions:
                pos_strs = [
                    f"{name} [bold]${prices[name]:,.2f}[/bold]"
                    for name in wallet.positions
                    if name in prices
                ]
                pos_info = "  ".join(pos_strs) if pos_strs else "—"
            else:
                pos_info = "no open positions"
            console.print(
                f"  [dim]{fmt_now()}  ·  next scan "
                f"{time_remaining // 60}m{time_remaining % 60:02d}s  ·  {pos_info}[/dim]"
            )

    console.print("[yellow]Multi-asset regime trader stopped.[/yellow]")
