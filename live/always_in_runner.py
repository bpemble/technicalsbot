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

import time
import signal
import sys
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd

from utils import fmt_ts, fmt_now
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

import config
from data.fetcher import fetch_funding_rates, fetch_latest_prices, DataCache
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
    return score


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

def fetch_asset(asset: dict, cache: DataCache) -> dict | None:
    """Return all 4 timeframes for one asset from the cache."""
    try:
        coin = asset["hl"]
        return {
            "1d":  cache.get(coin, "1d",  config.LOOKBACK_DAYS),
            "4h":  cache.get(coin, "4h",  config.LOOKBACK_DAYS),
            "1h":  cache.get(coin, "1h",  7),
            "15m": cache.get(coin, "15m", config.SCALP_LOOKBACK_DAYS),
        }
    except Exception:
        return None


# ------------------------------------------------------------------
# Regime computation across all assets
# ------------------------------------------------------------------

def compute_all_regimes(engine: RegimeEngine, cache: DataCache, funding_rates: dict | None = None) -> list[dict]:
    """
    Loop over all config.ASSETS in parallel, fetch data, compute regime snapshot.

    funding_rates : optional {hl_name: hourly_rate} from fetch_funding_rates().
                    When provided, each asset's score is adjusted for funding crowding.

    Returns list of dicts sorted by abs(adj_score) descending:
      [{"asset": ..., "snapshot": ..., "price": float,
        "adj_score": float, "funding_rate": float}, ...]
    """
    results = []

    def process_asset(asset):
        name = asset["name"]
        frames = fetch_asset(asset, cache)
        if frames is None:
            return name, None
        try:
            df_1d  = add_indicators(frames["1d"],  config)
            df_4h  = add_indicators(frames["4h"],  config)
            df_1h  = add_indicators(frames["1h"],  config)
            df_15m = add_indicators(frames["15m"], config)
            snapshot = engine.compute(df_1d, df_4h, df_1h, df_15m)
            hl_name  = asset.get("hl", asset["name"])
            fr       = (funding_rates or {}).get(hl_name, 0.0)
            adj      = apply_funding_adjustment(snapshot.score, fr)
            return name, {
                "asset":        asset,
                "snapshot":     snapshot,
                "price":        snapshot.latest_price,
                "adj_score":    adj,
                "funding_rate": fr,
            }
        except Exception as e:
            logger.warning(f"{name}: indicator/regime error: {e}")
            return name, None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_asset, asset): asset for asset in config.ASSETS}
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

        direction = pos["direction"]

        if direction == "long" and current_price <= pos["stop_loss"]:
            fee   = pos["size"] * current_price * config.EXCHANGE_FEE
            trade = wallet.close_position(asset_name, current_price, "hard_stop", fee)
            if trade:
                entry = pos["entry_price"]
                move_pct = (current_price - entry) / entry * 100
                msg = (
                    f"HARD STOP {asset_name} LONG entered @ ${entry:,.2f}, "
                    f"now ${current_price:,.2f} ({move_pct:.1f}%)  "
                    f"PnL: ${trade['net_pnl']:+,.2f}"
                )
                console.print(f"  [red bold]{msg}[/red bold]")
                logger.info(msg)
        elif direction == "short" and current_price >= pos["stop_loss"]:
            fee   = pos["size"] * current_price * config.EXCHANGE_FEE
            trade = wallet.close_position(asset_name, current_price, "hard_stop", fee)
            if trade:
                entry = pos["entry_price"]
                move_pct = (entry - current_price) / entry * 100
                msg = (
                    f"HARD STOP {asset_name} SHORT entered @ ${entry:,.2f}, "
                    f"now ${current_price:,.2f} ({move_pct:.1f}%)  "
                    f"PnL: ${trade['net_pnl']:+,.2f}"
                )
                console.print(f"  [red bold]{msg}[/red bold]")
                logger.info(msg)


# ------------------------------------------------------------------
# Portfolio rebalancing
# ------------------------------------------------------------------

def rebalance_portfolio(wallet: PaperWallet, regimes: list[dict]):
    """
    1. Select top MAX_POSITIONS assets where abs(score) > MIN_CONVICTION_SCORE.
    2. Close positions not in that active set.
    3. Open / flip / resize positions for each active asset.
    """
    # Build active set from sorted regimes
    active_set: dict[str, dict] = {}
    for entry in regimes:
        if len(active_set) >= config.MAX_POSITIONS:
            break
        score = entry["adj_score"]
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

        want_size = target_position_size(score, equity, current_price)
        if want_size <= 0:
            continue

        pos = wallet.get_position(asset_name)

        # No position — open one
        if pos is None:
            fill = current_price * (
                1 + config.SLIPPAGE if target_dir == "long" else 1 - config.SLIPPAGE
            )
            fee = want_size * fill * config.EXCHANGE_FEE
            wallet.open_position(
                strategy    = asset_name,
                direction   = target_dir,
                entry_price = fill,
                size        = want_size,
                stop_loss   = fill * (1 - HARD_STOP_PCT) if target_dir == "long"
                              else fill * (1 + HARD_STOP_PCT),
                take_profit = fill * (1 + HARD_STOP_PCT * 3) if target_dir == "long"
                              else fill * (1 - HARD_STOP_PCT * 3),
                fee         = fee,
            )
            dir_color = "cyan" if target_dir == "long" else "magenta"
            msg = (
                f"OPENED {asset_name} {target_dir.upper()} "
                f"@ ${fill:,.2f}  size={want_size:.4f}  score={score:+.1f}"
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
            fill = current_price * (
                1 + config.SLIPPAGE if target_dir == "long" else 1 - config.SLIPPAGE
            )
            fee = want_size * fill * config.EXCHANGE_FEE
            wallet.open_position(
                strategy    = asset_name,
                direction   = target_dir,
                entry_price = fill,
                size        = want_size,
                stop_loss   = fill * (1 - HARD_STOP_PCT) if target_dir == "long"
                              else fill * (1 + HARD_STOP_PCT),
                take_profit = fill * (1 + HARD_STOP_PCT * 3) if target_dir == "long"
                              else fill * (1 - HARD_STOP_PCT * 3),
                fee         = fee,
            )
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
            fill = current_price * (
                1 + config.SLIPPAGE if target_dir == "long" else 1 - config.SLIPPAGE
            )
            fee  = want_size * fill * config.EXCHANGE_FEE
            wallet.open_position(
                strategy    = asset_name,
                direction   = target_dir,
                entry_price = fill,
                size        = want_size,
                stop_loss   = fill * (1 - HARD_STOP_PCT) if target_dir == "long"
                              else fill * (1 + HARD_STOP_PCT),
                take_profit = fill * (1 + HARD_STOP_PCT * 3) if target_dir == "long"
                              else fill * (1 - HARD_STOP_PCT * 3),
                fee         = fee,
            )
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

            pt.add_row(
                f"[bold]{asset_name}[/bold]",
                f"[{dir_color}]{pos['direction'].upper()}[/{dir_color}]",
                entry_str,
                cur_str,
                f"{pos['size']:.4f}",
                f"[{upnl_color}]${upnl:+,.2f}[/{upnl_color}]",
                f"${pos['stop_loss']:,.2f}",
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

def _save_equity_snapshot(wallet: "PaperWallet", regimes: list[dict]):
    """Append current equity to equity_history.jsonl for the dashboard chart."""
    import json
    price_map = {r["asset"]["name"]: r["price"] for r in regimes}
    upnl      = wallet.total_unrealized_pnl(price_map)
    equity    = wallet.capital + upnl
    entry     = {"t": datetime.now(tz=timezone.utc).isoformat(), "v": round(equity, 2)}
    with open(config.EQUITY_HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _save_scores(regimes: list[dict], tick: int):
    """Persist latest regime scores to scores.json for the dashboard."""
    import json, os
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
    })
    console.print("  [green]Cache warmed.[/green]")
    console.print()

    tick    = 0
    regimes = []   # kept in scope so fast loop can reference open positions

    while _running:
        # ---- Full regime scan ----------------------------------------
        tick += 1
        console.rule(f"[dim]Tick {tick}  —  {fmt_now()}[/dim]")
        console.print()

        console.print("  [dim]Refreshing data and funding rates...[/dim]")
        funding_rates = fetch_funding_rates()
        regimes = compute_all_regimes(engine, cache, funding_rates)

        if not regimes:
            console.print("  [red]No asset data available this tick — skipping.[/red]")
        else:
            price_lookup = {r["asset"]["name"]: r["price"] for r in regimes}
            check_hard_stops(wallet, price_lookup)
            rebalance_portfolio(wallet, regimes)
            print_portfolio(regimes, wallet)
            _save_scores(regimes, tick)

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
