# ============================================================
# live/always_in_runner.py — Always-in regime-based paper trader
#
# Philosophy:
#   The bot always has an opinion. It computes a conviction score
#   every tick and sizes its position proportionally. When the
#   regime flips, it flips. When conviction fades, it trims.
#   It is never idle — it is always either long, short, or at
#   minimum conviction (flat only in the deadzone ±15).
#
# Position sizing:
#   notional = capital * leverage * (|score| / 100)
#   size_eth = notional / current_price
#
# Hard stop circuit breaker:
#   If price moves > HARD_STOP_PCT against entry, close regardless
#   of regime score. Protects against runaway losses.
# ============================================================

import time
import signal
import sys
from datetime import datetime, timezone

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

import config
from data.fetcher import fetch_ohlcv
from indicators.compute import add_indicators
from strategy.regime import RegimeEngine
from live.paper_wallet import PaperWallet

console = Console()
_running = True

STRATEGY_NAME = "regime"
HARD_STOP_PCT = 0.08        # 8% hard stop — circuit breaker regardless of score
REBALANCE_THRESHOLD = 0.10  # only rebalance if target size differs by >10%


def _handle_sigint(sig, frame):
    global _running
    console.print("\n[yellow]Shutting down...[/yellow]")
    _running = False


signal.signal(signal.SIGINT, _handle_sigint)


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

def fetch_all() -> dict:
    timeframes = {
        "1d":  config.LOOKBACK_DAYS,
        "4h":  config.LOOKBACK_DAYS,
        "1h":  7,
        "15m": config.SCALP_LOOKBACK_DAYS,
    }
    result = {}
    for tf, days in timeframes.items():
        result[tf] = fetch_ohlcv(config.SYMBOL, tf, days)
    return result


# ------------------------------------------------------------------
# Position helpers
# ------------------------------------------------------------------

def target_size(score: float, capital: float, price: float) -> float:
    """ETH size for a given regime score and capital."""
    conviction = abs(score) / 100.0
    notional   = capital * config.LEVERAGE * conviction
    return notional / price if price > 0 else 0.0


def check_hard_stop(wallet: PaperWallet, current_price: float):
    """Close position if hard stop % is breached."""
    pos = wallet.get_position(STRATEGY_NAME)
    if pos is None:
        return

    entry = pos["entry_price"]
    direction = pos["direction"]

    if direction == "long":
        move_pct = (current_price - entry) / entry
    else:
        move_pct = (entry - current_price) / entry

    if move_pct < -HARD_STOP_PCT:
        fee = pos["size"] * current_price * config.EXCHANGE_FEE
        trade = wallet.close_position(STRATEGY_NAME, current_price, "hard_stop", fee)
        if trade:
            console.print(
                f"  [red bold]HARD STOP triggered: {direction.upper()} entered "
                f"@ ${entry:,.2f}, now ${current_price:,.2f} "
                f"({move_pct*100:.1f}%)  PnL: ${trade['net_pnl']:+,.2f}[/red bold]"
            )


def rebalance(wallet: PaperWallet, snapshot, current_price: float):
    """
    Compare current position to target regime position.
    Open, flip, resize, or close as needed.
    """
    target_dir  = snapshot.direction
    score       = snapshot.score
    capital     = wallet.capital
    pos         = wallet.get_position(STRATEGY_NAME)

    # ---- Determine what we want ----------------------------------
    if target_dir == "flat":
        # Close any open position
        if pos is not None:
            fee = pos["size"] * current_price * config.EXCHANGE_FEE
            trade = wallet.close_position(STRATEGY_NAME, current_price, "regime_flat", fee)
            if trade:
                console.print(f"  [yellow]CLOSED (regime flat)  PnL: ${trade['net_pnl']:+,.2f}[/yellow]")
        return

    want_size = target_size(score, capital, current_price)
    if want_size <= 0:
        return

    # ---- No position — open one ----------------------------------
    if pos is None:
        fill = current_price * (1 + config.SLIPPAGE if target_dir == "long" else 1 - config.SLIPPAGE)
        fee  = want_size * fill * config.EXCHANGE_FEE
        wallet.open_position(
            strategy   = STRATEGY_NAME,
            direction  = target_dir,
            entry_price= fill,
            size       = want_size,
            stop_loss  = fill * (1 - HARD_STOP_PCT) if target_dir == "long" else fill * (1 + HARD_STOP_PCT),
            take_profit= fill * (1 + HARD_STOP_PCT * 3) if target_dir == "long" else fill * (1 - HARD_STOP_PCT * 3),
            fee        = fee,
        )
        dir_color = "cyan" if target_dir == "long" else "magenta"
        console.print(
            f"  [{dir_color}]OPENED {target_dir.upper()} @ ${fill:,.2f}  "
            f"size={want_size:.4f} ETH  score={score:+.1f}[/{dir_color}]"
        )
        return

    current_dir  = pos["direction"]
    current_size = pos["size"]

    # ---- Wrong direction — flip ----------------------------------
    if current_dir != target_dir:
        fee = current_size * current_price * config.EXCHANGE_FEE
        trade = wallet.close_position(STRATEGY_NAME, current_price, "regime_flip", fee)
        if trade:
            pnl_color = "green" if trade["net_pnl"] >= 0 else "red"
            console.print(
                f"  [{pnl_color}]FLIPPED {current_dir.upper()} → {target_dir.upper()} "
                f"@ ${current_price:,.2f}  PnL: ${trade['net_pnl']:+,.2f}[/{pnl_color}]"
            )
        # Open new position in opposite direction
        fill = current_price * (1 + config.SLIPPAGE if target_dir == "long" else 1 - config.SLIPPAGE)
        fee  = want_size * fill * config.EXCHANGE_FEE
        wallet.open_position(
            strategy   = STRATEGY_NAME,
            direction  = target_dir,
            entry_price= fill,
            size       = want_size,
            stop_loss  = fill * (1 - HARD_STOP_PCT) if target_dir == "long" else fill * (1 + HARD_STOP_PCT),
            take_profit= fill * (1 + HARD_STOP_PCT * 3) if target_dir == "long" else fill * (1 - HARD_STOP_PCT * 3),
            fee        = fee,
        )
        dir_color = "cyan" if target_dir == "long" else "magenta"
        console.print(
            f"  [{dir_color}]OPENED {target_dir.upper()} @ ${fill:,.2f}  "
            f"size={want_size:.4f} ETH  score={score:+.1f}[/{dir_color}]"
        )
        return

    # ---- Same direction — resize if materially different ----------
    size_diff_pct = abs(want_size - current_size) / max(current_size, 1e-9)
    if size_diff_pct > REBALANCE_THRESHOLD:
        # Update size in wallet (close and reopen at same price to track correctly)
        fee = current_size * current_price * config.EXCHANGE_FEE
        wallet.close_position(STRATEGY_NAME, current_price, "resize", fee)
        fill = current_price
        fee  = want_size * fill * config.EXCHANGE_FEE
        wallet.open_position(
            strategy   = STRATEGY_NAME,
            direction  = target_dir,
            entry_price= fill,
            size       = want_size,
            stop_loss  = fill * (1 - HARD_STOP_PCT) if target_dir == "long" else fill * (1 + HARD_STOP_PCT),
            take_profit= fill * (1 + HARD_STOP_PCT * 3) if target_dir == "long" else fill * (1 - HARD_STOP_PCT * 3),
            fee        = fee,
        )
        console.print(
            f"  [dim]RESIZED {target_dir.upper()}  "
            f"{current_size:.4f} → {want_size:.4f} ETH  score={score:+.1f}[/dim]"
        )


# ------------------------------------------------------------------
# Display
# ------------------------------------------------------------------

def score_bar(score: float, width: int = 30) -> str:
    """Render a coloured ASCII bar showing score magnitude and direction."""
    half = width // 2
    filled = int(abs(score) / 100 * half)
    filled = min(filled, half)

    if score >= 0:
        bar = " " * half + "[green]" + "█" * filled + "[/green]" + " " * (half - filled)
    else:
        bar = " " * (half - filled) + "[red]" + "█" * filled + "[/red]" + " " * half

    mid = "|"
    return bar[:half] + mid + bar[half:]


def print_regime(snapshot, wallet: PaperWallet):
    score     = snapshot.score
    direction = snapshot.direction
    price     = snapshot.latest_price
    upnl      = wallet.total_unrealized_pnl({STRATEGY_NAME: price})
    equity    = wallet.capital + upnl
    ret       = (equity - config.PAPER_CAPITAL) / config.PAPER_CAPITAL * 100

    dir_color = {"long": "cyan", "short": "magenta", "flat": "yellow"}[direction]
    ret_color = "green" if ret >= 0 else "red"
    ts        = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Main status table
    t = Table(box=box.SIMPLE, show_header=False, expand=False, min_width=52)
    t.add_column("K", style="dim",  min_width=20)
    t.add_column("V", style="bold", min_width=28)

    t.add_row("ETH/USD",         f"${price:,.2f}")
    t.add_row("Regime score",    f"[{dir_color}]{score:+.1f}[/{dir_color}]  {score_bar(score, 24)}")
    t.add_row("Direction",       f"[{dir_color} bold]{direction.upper()}[/{dir_color} bold]")
    t.add_row("Conviction",      f"{snapshot.conviction*100:.0f} %")
    t.add_row("",                "")
    t.add_row("Cash capital",    f"${wallet.capital:,.2f}")
    t.add_row("Unrealised PnL",  f"[{'green' if upnl >= 0 else 'red'}]${upnl:+,.2f}[/{'green' if upnl >= 0 else 'red'}]")
    t.add_row("Total equity",    f"${equity:,.2f}")
    t.add_row("Return",          f"[{ret_color}]{ret:+.2f} %[/{ret_color}]")
    t.add_row("Closed trades",   str(len(wallet.trades)))

    pos = wallet.get_position(STRATEGY_NAME)
    if pos:
        entry     = pos["entry_price"]
        move_pct  = ((price - entry) / entry * 100) if pos["direction"] == "long" \
                    else ((entry - price) / entry * 100)
        mc        = "green" if move_pct >= 0 else "red"
        t.add_row("Position",
                  f"[{dir_color}]{pos['direction'].upper()} {pos['size']:.4f} ETH "
                  f"@ ${entry:,.2f}  [{mc}]{move_pct:+.2f}%[/{mc}][/{dir_color}]")
        t.add_row("Hard stop",   f"${pos['stop_loss']:,.2f}")

    console.print(Panel(t, title=f"[bold cyan]Regime Trader — {ts}[/bold cyan]",
                        border_style="cyan", expand=False))

    # Per-timeframe breakdown
    tf_table = Table(box=box.SIMPLE, show_header=True, expand=False,
                     title="[dim]Score Breakdown[/dim]", title_style="dim")
    tf_table.add_column("TF",   min_width=6,  style="dim")
    tf_table.add_column("Score", min_width=7, justify="right")
    tf_table.add_column("RSI",   min_width=7, justify="right")
    tf_table.add_column("EMA",   min_width=7, justify="right")
    tf_table.add_column("MACD",  min_width=7, justify="right")
    tf_table.add_column("BB",    min_width=7, justify="right")

    tf_weights = {"daily": "40%", "4h": "30%", "1h": "20%", "15m": "10%"}

    for tf in ["daily", "4h", "1h", "15m"]:
        cs   = snapshot.component_scores.get(tf, 0)
        inds = snapshot.indicator_scores.get(tf, {})
        c    = "green" if cs > 0.05 else "red" if cs < -0.05 else "dim"

        def fmt_ind(key):
            v = inds.get(key)
            if v is None:
                return "[dim]—[/dim]"
            ic = "green" if v > 0.05 else "red" if v < -0.05 else "dim"
            return f"[{ic}]{v:+.2f}[/{ic}]"

        tf_table.add_row(
            f"{tf} ({tf_weights[tf]})",
            f"[{c}]{cs:+.2f}[/{c}]",
            fmt_ind("rsi"),
            fmt_ind("ema_trend"),
            fmt_ind("macd"),
            fmt_ind("bb_reversion"),
        )

    console.print(tf_table)
    console.print()


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def run():
    console.print()
    console.print(Panel(
        "[bold cyan]ETH/USD Regime Trader — Paper Mode[/bold cyan]\n"
        "[dim]Always-in conviction-weighted positioning[/dim]\n"
        f"[dim]Capital: ${config.PAPER_CAPITAL:,.0f}  |  "
        f"Leverage: {config.LEVERAGE}x  |  "
        f"Hard stop: {HARD_STOP_PCT*100:.0f}%  |  "
        f"Poll: {config.POLL_INTERVAL_SEC//60}min[/dim]",
        border_style="cyan", expand=False,
    ))
    console.print()

    wallet = PaperWallet(config.PAPER_STATE_FILE, config.PAPER_CAPITAL)
    engine = RegimeEngine(config)

    console.print(f"  Wallet loaded — cash: [bold]${wallet.capital:,.2f}[/bold]  "
                  f"closed trades: [bold]{len(wallet.trades)}[/bold]")
    console.print()

    tick = 0
    while _running:
        tick += 1
        console.rule(f"[dim]Tick {tick}[/dim]")

        # Fetch
        try:
            with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                          console=console, transient=True) as p:
                p.add_task("Fetching market data…", total=None)
                frames = fetch_all()
        except Exception as exc:
            console.print(f"  [red]Fetch error: {exc}[/red] — retrying next tick")
            time.sleep(config.POLL_INTERVAL_SEC)
            continue

        # Compute indicators
        df_daily = add_indicators(frames["1d"],  config)
        df_4h    = add_indicators(frames["4h"],  config)
        df_1h    = add_indicators(frames["1h"],  config)
        df_15m   = add_indicators(frames["15m"], config)

        # Compute regime
        try:
            snapshot = engine.compute(df_daily, df_4h, df_1h, df_15m)
        except Exception as exc:
            console.print(f"  [red]Regime error: {exc}[/red]")
            time.sleep(config.POLL_INTERVAL_SEC)
            continue

        current_price = snapshot.latest_price

        # Check hard stop first
        check_hard_stop(wallet, current_price)

        # Rebalance to target regime
        rebalance(wallet, snapshot, current_price)

        # Display
        print_regime(snapshot, wallet)

        if not _running:
            break

        console.print(f"  [dim]Next update in {config.POLL_INTERVAL_SEC // 60} min. Ctrl+C to stop.[/dim]\n")
        time.sleep(config.POLL_INTERVAL_SEC)

    console.print("[yellow]Regime trader stopped.[/yellow]")
