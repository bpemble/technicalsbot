"""
Paper trading runner.
Polls Kraken every POLL_INTERVAL_SEC seconds, generates signals from both
the swing strategy (daily+4h) and scalp strategy (1h+15m), and executes
paper trades via PaperWallet.
"""
import time
import signal
import sys
from datetime import datetime, timezone

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

import config
from data.fetcher import fetch_ohlcv
from indicators.compute import add_indicators
from strategy.multi_tf import MultiTFStrategy
from strategy.scalp import ScalpStrategy
from live.paper_wallet import PaperWallet

console = Console()
_running = True


def _handle_sigint(sig, frame):
    global _running
    console.print("\n[yellow]Shutting down paper trader...[/yellow]")
    _running = False


signal.signal(signal.SIGINT, _handle_sigint)


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------

def fetch_all() -> dict:
    """Fetch all required timeframes. Returns dict of DataFrames."""
    frames = {}
    timeframes = {
        "1d":  config.LOOKBACK_DAYS,
        "4h":  config.LOOKBACK_DAYS,
        "1h":  7,     # Kraken 1h: ~7 days is enough for indicators
        "15m": config.SCALP_LOOKBACK_DAYS,
    }
    for tf, days in timeframes.items():
        frames[tf] = fetch_ohlcv(config.SYMBOL, tf, days)
    return frames


# ------------------------------------------------------------------
# Position management helpers
# ------------------------------------------------------------------

def check_stops(wallet: PaperWallet, current_price: float, exchange_fee: float):
    """Check if any open position's stop or TP has been hit."""
    closed = []
    for strategy, pos in list(wallet.positions.items()):
        direction   = pos["direction"]
        stop_loss   = pos["stop_loss"]
        take_profit = pos["take_profit"]
        size        = pos["size"]

        exit_price  = None
        exit_reason = None

        if direction == "long":
            if current_price <= stop_loss:
                exit_price, exit_reason = stop_loss, "stop_loss"
            elif current_price >= take_profit:
                exit_price, exit_reason = take_profit, "take_profit"
        else:
            if current_price >= stop_loss:
                exit_price, exit_reason = stop_loss, "stop_loss"
            elif current_price <= take_profit:
                exit_price, exit_reason = take_profit, "take_profit"

        if exit_reason:
            fee   = size * exit_price * exchange_fee
            trade = wallet.close_position(strategy, exit_price, exit_reason, fee)
            if trade:
                pnl_color = "green" if trade["net_pnl"] >= 0 else "red"
                console.print(
                    f"  [{pnl_color}]CLOSED {strategy} {direction.upper()} "
                    f"@ ${exit_price:,.2f} ({exit_reason})  "
                    f"PnL: ${trade['net_pnl']:+,.2f}[/{pnl_color}]"
                )
            closed.append(strategy)
    return closed


def try_entry(
    wallet: PaperWallet,
    strategy_name: str,
    signals: pd.DataFrame,
    current_price: float,
    risk_pct: float,
    atr_stop: float,
    exchange_fee: float,
):
    """Check latest signal row and open a paper position if signalled."""
    if wallet.has_position(strategy_name):
        return

    last = signals.iloc[-1]
    sig  = int(last["signal"])
    if sig == 0:
        return

    direction   = "long" if sig == 1 else "short"
    atr_val     = float(last["atr_at_signal"]) if not pd.isna(last["atr_at_signal"]) else 0
    stop_dist   = atr_val * atr_stop
    if stop_dist <= 0:
        return

    # Use slippage-adjusted fill price
    fill = current_price * (1 + config.SLIPPAGE if sig == 1 else 1 - config.SLIPPAGE)

    stop_loss   = fill - stop_dist if sig == 1 else fill + stop_dist
    take_profit = (
        fill + atr_val * (config.ATR_TP_MULTIPLIER if "swing" in strategy_name else config.SCALP_ATR_TP)
        if sig == 1
        else fill - atr_val * (config.ATR_TP_MULTIPLIER if "swing" in strategy_name else config.SCALP_ATR_TP)
    )

    dollar_risk = wallet.capital * risk_pct
    size        = min(dollar_risk / stop_dist, (wallet.capital * config.LEVERAGE) / fill)
    if size <= 0:
        return

    fee = size * fill * exchange_fee
    wallet.open_position(strategy_name, direction, fill, size, stop_loss, take_profit, fee)

    dir_color = "cyan" if direction == "long" else "magenta"
    console.print(
        f"  [{dir_color}]OPENED {strategy_name} {direction.upper()} "
        f"@ ${fill:,.2f}  size={size:.4f} ETH  "
        f"SL=${stop_loss:,.2f}  TP=${take_profit:,.2f}[/{dir_color}]"
    )


# ------------------------------------------------------------------
# Status display
# ------------------------------------------------------------------

def print_status(wallet: PaperWallet, current_price: float):
    upnl   = wallet.total_unrealized_pnl(
        {s: current_price for s in wallet.positions}
    )
    equity = wallet.capital + upnl
    ret    = ((equity - wallet.initial_capital) / wallet.initial_capital) * 100

    table = Table(box=box.SIMPLE, show_header=False, expand=False)
    table.add_column("K", style="dim", min_width=22)
    table.add_column("V", style="bold", min_width=16)

    ret_color = "green" if ret >= 0 else "red"
    table.add_row("ETH/USD",          f"${current_price:,.2f}")
    table.add_row("Cash capital",     f"${wallet.capital:,.2f}")
    table.add_row("Unrealised PnL",   f"[{'green' if upnl >= 0 else 'red'}]${upnl:+,.2f}[/{'green' if upnl >= 0 else 'red'}]")
    table.add_row("Total equity",     f"${equity:,.2f}")
    table.add_row("Return",           f"[{ret_color}]{ret:+.2f} %[/{ret_color}]")
    table.add_row("Closed trades",    str(len(wallet.trades)))
    table.add_row("Open positions",   str(len(wallet.positions)))

    for name, pos in wallet.positions.items():
        upnl_pos = wallet.unrealized_pnl(name, current_price)
        c = "green" if upnl_pos >= 0 else "red"
        table.add_row(
            f"  {name}",
            f"[{c}]{pos['direction'].upper()} ${pos['entry_price']:,.2f} → "
            f"${upnl_pos:+,.2f}[/{c}]"
        )

    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    console.print(Panel(table, title=f"[bold cyan]Paper Wallet — {ts}[/bold cyan]", border_style="cyan", expand=False))


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def run():
    console.print()
    console.print(Panel(
        "[bold cyan]ETH/USD Paper Trader[/bold cyan]\n"
        "[dim]Swing (daily+4h) + Scalp (1h+15m) strategies[/dim]\n"
        f"[dim]Starting capital: ${config.PAPER_CAPITAL:,.0f} USDC  |  "
        f"Poll interval: {config.POLL_INTERVAL_SEC // 60} min[/dim]",
        border_style="cyan", expand=False,
    ))

    wallet = PaperWallet(config.PAPER_STATE_FILE, config.PAPER_CAPITAL)
    swing  = MultiTFStrategy(config)
    scalp  = ScalpStrategy(config)

    console.print(f"  Wallet loaded — capital: [bold]${wallet.capital:,.2f}[/bold]  "
                  f"closed trades: [bold]{len(wallet.trades)}[/bold]")
    console.print()

    tick = 0
    while _running:
        tick += 1
        console.rule(f"[dim]Tick {tick} — {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}[/dim]")

        # ---- Fetch data ------------------------------------------
        try:
            console.print("  Fetching data...", end=" ")
            frames = fetch_all()
            console.print("[green]done[/green]")
        except Exception as exc:
            console.print(f"[red]fetch error: {exc}[/red]  (will retry next tick)")
            time.sleep(config.POLL_INTERVAL_SEC)
            continue

        current_price = float(frames["15m"]["close"].iloc[-1])

        # ---- Compute indicators ----------------------------------
        df_daily_ind = add_indicators(frames["1d"], config)
        df_4h_ind    = add_indicators(frames["4h"], config)
        df_1h_ind    = add_indicators(frames["1h"], config)
        df_15m_ind   = add_indicators(frames["15m"], config)

        # ---- Check stops on open positions ----------------------
        check_stops(wallet, current_price, config.EXCHANGE_FEE)

        # ---- Generate signals ------------------------------------
        try:
            swing_signals = swing.generate_signals(df_daily_ind, df_4h_ind)
            scalp_signals = scalp.generate_signals(df_1h_ind, df_15m_ind)
        except Exception as exc:
            console.print(f"[red]signal error: {exc}[/red]")
            time.sleep(config.POLL_INTERVAL_SEC)
            continue

        # ---- Try entries ----------------------------------------
        try_entry(wallet, "swing", swing_signals, current_price,
                  config.RISK_PER_TRADE, config.ATR_STOP_MULTIPLIER, config.EXCHANGE_FEE)
        try_entry(wallet, "scalp", scalp_signals, current_price,
                  config.SCALP_RISK_PER_TRADE, config.SCALP_ATR_STOP, config.EXCHANGE_FEE)

        # ---- Print status ----------------------------------------
        print_status(wallet, current_price)
        console.print()

        if not _running:
            break

        console.print(f"  [dim]Next tick in {config.POLL_INTERVAL_SEC // 60} min. Ctrl+C to stop.[/dim]")
        time.sleep(config.POLL_INTERVAL_SEC)

    console.print("[yellow]Paper trader stopped.[/yellow]")
