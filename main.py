"""
ETH/USDC Perpetuals Trading Bot — Backtester
Usage: python main.py

Runs a 2-year multi-timeframe EMA+RSI+MACD+ATR backtest using historical
Binance Futures data and reports performance metrics in the terminal.
"""

import sys
import math

import pandas as pd
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich import box

import config
from utils import fmt_ts
from data.fetcher import fetch_ohlcv
from indicators.compute import add_indicators
from strategy.multi_tf import MultiTFStrategy
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics


console = Console()


# ================================================================
# Helpers
# ================================================================

def draw_equity_curve(equity_curve: pd.Series, width: int = 70, height: int = 14) -> None:
    """
    Draw a simple ASCII equity curve in the terminal using rich markup.

    The chart fills `height` rows and `width` columns using block characters.
    """
    if equity_curve.empty:
        console.print("[yellow]No equity data to plot.[/yellow]")
        return

    values = equity_curve.resample("D").last().dropna().values.astype(float)
    if len(values) < 2:
        console.print("[yellow]Not enough equity data points to plot.[/yellow]")
        return

    # Downsample to `width` columns
    if len(values) > width:
        indices = np.linspace(0, len(values) - 1, width, dtype=int)
        values = values[indices]

    min_v = values.min()
    max_v = values.max()
    span  = max_v - min_v if max_v != min_v else 1.0

    # Normalise to [0, height-1]
    normed = ((values - min_v) / span * (height - 1)).astype(int)

    lines = []
    for row in range(height - 1, -1, -1):
        line_chars = []
        for col, col_val in enumerate(normed):
            if col_val == row:
                # Colour by position relative to initial
                color = "bright_green" if values[col] >= values[0] else "bright_red"
                line_chars.append(f"[{color}]█[/{color}]")
            elif col_val > row:
                color = "green" if values[col] >= values[0] else "red"
                line_chars.append(f"[{color}]│[/{color}]")
            else:
                line_chars.append(" ")
        lines.append("".join(line_chars))

    # Y-axis labels
    y_labels = [
        f"${max_v:>10,.0f}",
        f"${(min_v + span / 2):>10,.0f}",
        f"${min_v:>10,.0f}",
    ]

    label_rows = {height - 1: y_labels[0], height // 2: y_labels[1], 0: y_labels[2]}

    console.print()
    console.print(Panel(
        Text("Equity Curve (daily resampled)", justify="center"),
        style="bold cyan",
        expand=False,
    ))

    for i, line in enumerate(lines):
        row_idx = height - 1 - i
        label   = label_rows.get(row_idx, " " * 12)
        console.print(f"  [dim]{label}[/dim]  {line}")

    # X-axis
    x_left  = equity_curve.index[0].strftime("%Y-%m-%d")
    x_right = equity_curve.index[-1].strftime("%Y-%m-%d")
    padding = " " * 14
    x_axis  = f"{padding}[dim]{x_left}[/dim]{'':>{width - len(x_left) - 2}}[dim]{x_right}[/dim]"
    console.print(x_axis)
    console.print()


def print_metrics_table(metrics: dict) -> None:
    """Render all metrics in a nicely formatted rich table."""
    table = Table(
        title="Backtest Performance Metrics",
        box=box.DOUBLE_EDGE,
        title_style="bold cyan",
        show_header=True,
        header_style="bold magenta",
        expand=False,
        min_width=55,
    )
    table.add_column("Metric", style="bold", justify="left", min_width=28)
    table.add_column("Value", justify="right", min_width=18)

    def row(label: str, val_str: str):
        table.add_row(label, val_str)

    cap_init  = metrics["initial_capital"]
    cap_final = metrics["final_capital"]
    tot_ret   = metrics["total_return"]
    ann_ret   = metrics["annualized_return"]
    mdd       = metrics["max_drawdown"]
    sharpe    = metrics["sharpe_ratio"]
    sortino   = metrics["sortino_ratio"]
    pf        = metrics["profit_factor"]

    # Handle infinite profit factor
    pf_str = "∞" if pf == float("inf") else f"{pf:.3f}"

    row("Initial Capital",       f"[bold]${cap_init:,.2f}[/bold]")
    row("Final Capital",         f"[bold {'green' if cap_final >= cap_init else 'red'}]${cap_final:,.2f}[/bold {'green' if cap_final >= cap_init else 'red'}]")
    table.add_section()
    row("Total Return",          f"[{'green' if tot_ret >= 0 else 'red'}]{tot_ret:.2f} %[/{'green' if tot_ret >= 0 else 'red'}]")
    row("Annualised Return",     f"[{'green' if ann_ret >= 0 else 'red'}]{ann_ret:.2f} %[/{'green' if ann_ret >= 0 else 'red'}]")
    row("Max Drawdown",          f"[red]{mdd:.2f} %[/red]")
    table.add_section()
    row("Sharpe Ratio",          f"[{'green' if sharpe >= 1 else 'yellow' if sharpe >= 0 else 'red'}]{sharpe:.3f}[/{'green' if sharpe >= 1 else 'yellow' if sharpe >= 0 else 'red'}]")
    row("Sortino Ratio",         f"[{'green' if sortino >= 1 else 'yellow' if sortino >= 0 else 'red'}]{sortino:.3f}[/{'green' if sortino >= 1 else 'yellow' if sortino >= 0 else 'red'}]")
    row("Profit Factor",         f"[{'green' if pf == float('inf') or pf >= 1 else 'red'}]{pf_str}[/{'green' if pf == float('inf') or pf >= 1 else 'red'}]")
    table.add_section()
    row("Total Trades",          str(metrics["total_trades"]))
    row("Winning Trades",        f"[green]{metrics['winning_trades']}[/green]")
    row("Losing Trades",         f"[red]{metrics['losing_trades']}[/red]")
    row("Win Rate",              f"[{'green' if metrics['win_rate'] >= 50 else 'red'}]{metrics['win_rate']:.2f} %[/{'green' if metrics['win_rate'] >= 50 else 'red'}]")
    table.add_section()
    row("Avg Win",               f"[green]+${metrics['avg_win']:,.2f}[/green]")
    row("Avg Loss",              f"[red]-${metrics['avg_loss']:,.2f}[/red]")
    row("Avg Win %",             f"[green]+{metrics['avg_win_pct']:.3f} %[/green]")
    row("Avg Loss %",            f"[red]-{abs(metrics['avg_loss_pct']):.3f} %[/red]")
    row("Largest Win",           f"[green]+${metrics['largest_win']:,.2f}[/green]")
    row("Largest Loss",          f"[red]-${abs(metrics['largest_loss']):,.2f}[/red]")
    row("Expectancy (per trade)",f"[{'green' if metrics['expectancy'] >= 0 else 'red'}]${metrics['expectancy']:,.2f}[/{'green' if metrics['expectancy'] >= 0 else 'red'}]")
    table.add_section()
    row("Gross Profit",          f"[green]+${metrics['gross_profit']:,.2f}[/green]")
    row("Gross Loss",            f"[red]-${metrics['gross_loss']:,.2f}[/red]")
    row("Total Fees Paid",       f"[yellow]-${metrics['total_fees_paid']:,.2f}[/yellow]")
    row("Total Funding Paid",    f"[yellow]-${metrics['total_funding_paid']:,.2f}[/yellow]")
    table.add_section()
    row("Avg Trade Duration",    f"{metrics['avg_trade_duration']:.1f} h")

    console.print(table)


def print_recent_trades(trades: list, n: int = 20) -> None:
    """Print the last N trades as a rich table."""
    if not trades:
        console.print("[yellow]No trades to display.[/yellow]")
        return

    recent = trades[-n:]

    table = Table(
        title=f"Last {len(recent)} Trades",
        box=box.SIMPLE_HEAVY,
        title_style="bold cyan",
        show_header=True,
        header_style="bold magenta",
        expand=False,
    )
    table.add_column("#",           justify="right",  style="dim",   min_width=4)
    table.add_column("Dir",         justify="center",                min_width=5)
    table.add_column("Entry Time",  justify="center",                min_width=17)
    table.add_column("Exit Time",   justify="center",                min_width=17)
    table.add_column("Entry $",     justify="right",                 min_width=9)
    table.add_column("Exit $",      justify="right",                 min_width=9)
    table.add_column("Size (ETH)",  justify="right",                 min_width=10)
    table.add_column("PnL (USDC)",  justify="right",                 min_width=11)
    table.add_column("PnL %",       justify="right",                 min_width=8)
    table.add_column("Exit Reason", justify="left",                  min_width=14)
    table.add_column("Fees",        justify="right",  style="yellow", min_width=8)

    start_idx = len(trades) - len(recent) + 1

    for idx, t in enumerate(recent, start=start_idx):
        direction   = t["direction"]
        pnl         = t["pnl"]
        pnl_pct     = t["pnl_pct"]
        color       = "green" if pnl >= 0 else "red"
        dir_markup  = "[cyan]LONG[/cyan]" if direction == "long" else "[magenta]SHORT[/magenta]"

        entry_str = fmt_ts(t["entry_time"])
        exit_str  = fmt_ts(t["exit_time"])

        reason_colors = {
            "stop_loss":      "red",
            "take_profit":    "green",
            "signal_reverse": "yellow",
            "end_of_data":    "dim",
        }
        reason = t.get("exit_reason", "")
        rc     = reason_colors.get(reason, "white")

        table.add_row(
            str(idx),
            dir_markup,
            entry_str,
            exit_str,
            f"${t['entry_price']:,.2f}",
            f"${t['exit_price']:,.2f}",
            f"{t['size']:.4f}",
            f"[{color}]{'+' if pnl >= 0 else ''}{pnl:,.2f}[/{color}]",
            f"[{color}]{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%[/{color}]",
            f"[{rc}]{reason}[/{rc}]",
            f"${t.get('fees_paid', 0):,.2f}",
        )

    console.print(table)


# ================================================================
# Main
# ================================================================

def main() -> None:
    console.print()
    console.print(
        Panel(
            "[bold cyan]ETH/USDC Perpetuals Trading Bot — Backtester[/bold cyan]\n"
            "[dim]Multi-timeframe EMA + RSI + MACD + ATR Strategy[/dim]\n"
            "[dim]Exchange: Hyperliquid  |  Data source: Binance Futures[/dim]",
            expand=False,
            border_style="cyan",
        )
    )
    console.print()

    # ---- Configuration summary ------------------------------------
    cfg_table = Table(box=box.MINIMAL, show_header=False, expand=False)
    cfg_table.add_column("Key",   style="dim",  min_width=22)
    cfg_table.add_column("Value", style="bold", min_width=12)
    cfg_pairs = [
        ("Symbol",           config.SYMBOL),
        ("Timeframes",       config.PRIMARY_TF + " (trend) + " + config.ENTRY_TF + " (entry)"),
        ("Lookback",         f"{config.LOOKBACK_DAYS} days"),
        ("Initial Capital",  f"${config.INITIAL_CAPITAL:,.0f} USDC"),
        ("Leverage",         f"{config.LEVERAGE}x"),
        ("Risk / Trade",     f"{config.RISK_PER_TRADE * 100:.0f} %"),
        ("ATR Stop Mult",    str(config.ATR_STOP_MULTIPLIER)),
        ("ATR TP Mult",      str(config.ATR_TP_MULTIPLIER)),
        ("EMA Fast / Slow",  f"{config.EMA_FAST} / {config.EMA_SLOW}"),
        ("RSI Period",       str(config.RSI_PERIOD)),
        ("Exchange Fee",     f"{config.EXCHANGE_FEE * 100:.4f} %"),
    ]
    for k, v in cfg_pairs:
        cfg_table.add_row(k, str(v))
    console.print(Panel(cfg_table, title="[bold]Configuration[/bold]", border_style="dim"))
    console.print()

    # ---- Fetch data -----------------------------------------------
    console.print("[bold]Step 1 / 4 — Fetching historical data…[/bold]")
    df_daily: pd.DataFrame
    df_4h: pd.DataFrame

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        t1 = progress.add_task("Daily (trend) — fetching OHLCV…", total=None)
        try:
            df_daily = fetch_ohlcv(config.SYMBOL, "1d", config.LOOKBACK_DAYS)
        except Exception as exc:
            console.print(f"[red]Failed to fetch daily data: {exc}[/red]")
            sys.exit(1)
        progress.update(t1, description=f"[green]Daily data fetched ({len(df_daily)} bars)[/green]", completed=True)

        t2 = progress.add_task("4h (entry) — fetching OHLCV…", total=None)
        try:
            df_4h = fetch_ohlcv(config.SYMBOL, "4h", config.LOOKBACK_DAYS)
        except Exception as exc:
            console.print(f"[red]Failed to fetch 4h data: {exc}[/red]")
            sys.exit(1)
        progress.update(t2, description=f"[green]4h data fetched ({len(df_4h)} bars)[/green]", completed=True)

    console.print(
        f"  [green]Daily:[/green] {len(df_daily):,} bars  "
        f"({df_daily.index[0].strftime('%Y-%m-%d')} → {df_daily.index[-1].strftime('%Y-%m-%d')})\n"
        f"  [green]4h:[/green] {len(df_4h):,} bars  "
        f"({df_4h.index[0].strftime('%Y-%m-%d')} → {df_4h.index[-1].strftime('%Y-%m-%d')})"
    )
    console.print()

    # ---- Compute indicators ---------------------------------------
    console.print("[bold]Step 2 / 4 — Computing indicators…[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        ti = progress.add_task("Adding indicators…", total=None)
        df_daily_ind = add_indicators(df_daily, config)
        df_4h_ind    = add_indicators(df_4h, config)
        progress.update(ti, description="[green]Indicators computed[/green]", completed=True)

    console.print(f"  Indicator columns added: {[c for c in df_daily_ind.columns if c not in ['open','high','low','close','volume']]}")
    console.print()

    # ---- Generate signals -----------------------------------------
    console.print("[bold]Step 3 / 4 — Generating signals…[/bold]")
    strategy = MultiTFStrategy(config)
    signals  = strategy.generate_signals(df_daily_ind, df_4h_ind)

    n_long  = int((signals["signal"] ==  1).sum())
    n_short = int((signals["signal"] == -1).sum())
    console.print(f"  Long signals: [cyan]{n_long}[/cyan]   Short signals: [magenta]{n_short}[/magenta]")

    # ---- Run backtest ---------------------------------------------
    console.print("[bold]Step 4 / 4 — Running backtest engine…[/bold]")
    engine  = BacktestEngine(config)
    results = engine.run(signals, df_4h_ind)

    trades       = results["trades"]
    equity_curve = results["equity_curve"]
    final_cap    = results["final_capital"]

    console.print(f"  Total trades executed: [bold]{len(trades)}[/bold]")
    console.print(f"  Final capital: [bold]${final_cap:,.2f}[/bold]")
    console.print()

    # ---- Metrics --------------------------------------------------
    metrics = compute_metrics(trades, equity_curve, config.INITIAL_CAPITAL)

    console.rule("[bold cyan]Performance Report[/bold cyan]")
    console.print()
    print_metrics_table(metrics)
    console.print()

    # ---- Trade log ------------------------------------------------
    console.rule("[bold cyan]Recent Trades[/bold cyan]")
    console.print()
    print_recent_trades(trades, n=20)
    console.print()

    # ---- Equity curve --------------------------------------------
    console.rule("[bold cyan]Equity Curve[/bold cyan]")
    draw_equity_curve(equity_curve, width=70, height=14)

    # ---- Footer ---------------------------------------------------
    console.print(
        Panel(
            "[dim]Backtest complete. "
            "Results are for simulation purposes only. "
            "Past performance does not guarantee future results.[/dim]",
            border_style="dim",
            expand=False,
        )
    )
    console.print()


if __name__ == "__main__":
    main()
