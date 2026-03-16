"""
Regime strategy backtester.
Usage: python -m backtest.regime_backtest

Fetches historical data, computes regime scores bar-by-bar (no lookahead),
simulates always-in positioning, and reports performance metrics.
"""
import sys
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box

import config
from utils import fmt_ts
from data.fetcher import fetch_hl_candles
from indicators.compute import add_indicators
from strategy.regime import RegimeEngine
from backtest.metrics import compute_metrics

# ETH on Hyperliquid (full paginated history — used instead of Kraken)
_BT_COIN = "ETH"

def _fetch(interval: str, days: int) -> pd.DataFrame:
    """Fetch from Hyperliquid with proper pagination for full history."""
    from datetime import datetime, timedelta, timezone
    since_ms = int(
        (datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )
    return fetch_hl_candles(_BT_COIN, interval, since_ms)

console = Console()

HARD_STOP_PCT = 0.08


def run_regime_backtest():
    """
    Bar-by-bar backtest of the RegimeEngine always-in strategy.

    Uses 4h bars as the entry/exit timeframe. For each 4h bar, slices all
    timeframes to data up to that point (no lookahead bias), computes the
    regime score, and manages a single position with conviction-weighted
    sizing and an 8% hard stop.
    """
    console.print()
    console.print(Panel(
        "[bold cyan]Regime Strategy Backtester[/bold cyan]\n"
        "[dim]Always-in conviction-weighted positioning — no lookahead[/dim]\n"
        f"[dim]Symbol: {config.SYMBOL}  |  Entry TF: 4h  |  Trend TF: 1d[/dim]",
        expand=False,
        border_style="cyan",
    ))
    console.print()

    # ----------------------------------------------------------------
    # Step 1: Fetch data
    # ----------------------------------------------------------------
    console.print("[bold]Step 1 / 4 — Fetching historical data…[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        t_1d = progress.add_task("Daily bars…", total=None)
        try:
            df_1d_raw = _fetch("1d", config.LOOKBACK_DAYS)
        except Exception as exc:
            console.print(f"[red]Failed to fetch daily data: {exc}[/red]")
            sys.exit(1)
        progress.update(t_1d, description=f"[green]Daily: {len(df_1d_raw)} bars[/green]", completed=True)

        t_4h = progress.add_task("4h bars…", total=None)
        try:
            df_4h_raw = _fetch("4h", config.LOOKBACK_DAYS)
        except Exception as exc:
            console.print(f"[red]Failed to fetch 4h data: {exc}[/red]")
            sys.exit(1)
        progress.update(t_4h, description=f"[green]4h: {len(df_4h_raw)} bars[/green]", completed=True)

        t_1h = progress.add_task("1h bars…", total=None)
        try:
            df_1h_raw = _fetch("1h", 730)
        except Exception as exc:
            console.print(f"[yellow]1h fetch failed ({exc}), using empty frame[/yellow]")
            df_1h_raw = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        progress.update(t_1h, description=f"[green]1h: {len(df_1h_raw)} bars[/green]", completed=True)

        t_15m = progress.add_task("15m bars…", total=None)
        try:
            df_15m_raw = _fetch("15m", config.SCALP_LOOKBACK_DAYS)
        except Exception as exc:
            console.print(f"[yellow]15m fetch failed ({exc}), using empty frame[/yellow]")
            df_15m_raw = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        progress.update(t_15m, description=f"[green]15m: {len(df_15m_raw)} bars[/green]", completed=True)

    console.print(
        f"  Daily: [green]{len(df_1d_raw):,}[/green] bars  |  "
        f"4h: [green]{len(df_4h_raw):,}[/green] bars  |  "
        f"1h: [green]{len(df_1h_raw):,}[/green] bars  |  "
        f"15m: [green]{len(df_15m_raw):,}[/green] bars"
    )
    console.print()

    # ----------------------------------------------------------------
    # Step 2: Compute indicators on full history (then slice per bar)
    # ----------------------------------------------------------------
    console.print("[bold]Step 2 / 4 — Computing indicators on full history…[/bold]")

    df_1d_ind  = add_indicators(df_1d_raw,  config) if not df_1d_raw.empty  else df_1d_raw
    df_4h_ind  = add_indicators(df_4h_raw,  config) if not df_4h_raw.empty  else df_4h_raw
    df_1h_ind  = add_indicators(df_1h_raw,  config) if not df_1h_raw.empty  else df_1h_raw
    df_15m_ind = add_indicators(df_15m_raw, config) if not df_15m_raw.empty else df_15m_raw

    console.print("  [green]Done.[/green]")
    console.print()

    # ----------------------------------------------------------------
    # Step 3: Bar-by-bar simulation
    # ----------------------------------------------------------------
    console.print("[bold]Step 3 / 4 — Running bar-by-bar regime simulation…[/bold]")
    console.print(f"  [dim]Processing {len(df_4h_ind)} 4h bars — this may take a moment…[/dim]")

    engine = RegimeEngine(config)

    # Simulation state
    capital       = config.PAPER_CAPITAL
    initial_cap   = config.PAPER_CAPITAL
    equity_curve  = {}
    trades        = []

    in_position   = False
    direction     = None        # 'long' or 'short'
    entry_price   = 0.0
    entry_time    = None
    position_size = 0.0         # asset units
    stop_price    = 0.0
    fees_entry    = 0.0
    funding_acc   = 0.0
    candles_held  = 0

    bars_4h = list(df_4h_ind.index)
    n_bars  = len(bars_4h)

    # Minimum bars needed for indicators to warm up before we start trading
    WARMUP_BARS = 60

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        sim_task = progress.add_task("Simulating…", total=n_bars)

        for i, ts in enumerate(bars_4h):
            progress.advance(sim_task)

            bar = df_4h_ind.loc[ts]
            candle_open  = float(bar["open"])
            candle_high  = float(bar["high"])
            candle_low   = float(bar["low"])
            candle_close = float(bar["close"])

            # ---- Manage open position --------------------------------
            if in_position:
                candles_held += 1
                notional_now = position_size * candle_open

                # Funding every 2 candles on 4h data (= 8h period)
                if candles_held % 2 == 0:
                    funding_charge = notional_now * (config.FUNDING_RATE_DAILY / 3)
                    funding_acc   += funding_charge

                # Hard stop check
                stop_hit = False
                if direction == "long" and candle_low <= stop_price:
                    stop_hit = True
                elif direction == "short" and candle_high >= stop_price:
                    stop_hit = True

                if stop_hit:
                    exit_p = stop_price
                    exit_fee = position_size * exit_p * config.EXCHANGE_FEE
                    if direction == "long":
                        gross = (exit_p - entry_price) * position_size
                    else:
                        gross = (entry_price - exit_p) * position_size
                    net = gross - exit_fee - funding_acc
                    capital += gross - exit_fee
                    trades.append({
                        "entry_time":     entry_time,
                        "exit_time":      ts,
                        "direction":      direction,
                        "entry_price":    entry_price,
                        "exit_price":     exit_p,
                        "size":           position_size,
                        "pnl":            net,
                        "pnl_pct":        (net / (capital - net)) * 100 if (capital - net) > 0 else 0.0,
                        "exit_reason":    "hard_stop",
                        "fees_paid":      fees_entry + exit_fee,
                        "funding_paid":   funding_acc,
                        "duration_hours": (pd.Timestamp(ts) - pd.Timestamp(entry_time)).total_seconds() / 3600 if entry_time else 0.0,
                    })
                    in_position  = False
                    direction    = None
                    candles_held = 0
                    funding_acc  = 0.0
                    equity_curve[ts] = capital
                    continue

            # ---- Compute regime (no lookahead: slice to current bar) -----
            if i < WARMUP_BARS:
                equity_curve[ts] = capital
                continue

            # Slice each timeframe to timestamps <= current 4h bar's ts
            slice_1d  = df_1d_ind[df_1d_ind.index <= ts]
            slice_4h  = df_4h_ind.iloc[:i + 1]
            slice_1h  = df_1h_ind[df_1h_ind.index <= ts]  if not df_1h_ind.empty  else df_1h_ind
            slice_15m = df_15m_ind[df_15m_ind.index <= ts] if not df_15m_ind.empty else df_15m_ind

            if slice_1d.empty or slice_4h.empty:
                equity_curve[ts] = capital
                continue

            try:
                snapshot = engine.compute(slice_1d, slice_4h, slice_1h, slice_15m)
            except Exception:
                equity_curve[ts] = capital
                continue

            target_dir = snapshot.direction
            score      = snapshot.score

            # ---- Position management ---------------------------------
            if in_position:
                # Flip or close if regime changes
                if target_dir == "flat" or (target_dir != direction):
                    exit_p   = candle_close
                    exit_fee = position_size * exit_p * config.EXCHANGE_FEE
                    if direction == "long":
                        gross = (exit_p - entry_price) * position_size
                    else:
                        gross = (entry_price - exit_p) * position_size
                    net = gross - exit_fee - funding_acc
                    capital += gross - exit_fee
                    trades.append({
                        "entry_time":     entry_time,
                        "exit_time":      ts,
                        "direction":      direction,
                        "entry_price":    entry_price,
                        "exit_price":     exit_p,
                        "size":           position_size,
                        "pnl":            net,
                        "pnl_pct":        (net / (capital - net)) * 100 if (capital - net) > 0 else 0.0,
                        "exit_reason":    "regime_flip" if target_dir != "flat" else "regime_flat",
                        "fees_paid":      fees_entry + exit_fee,
                        "funding_paid":   funding_acc,
                        "duration_hours": (pd.Timestamp(ts) - pd.Timestamp(entry_time)).total_seconds() / 3600 if entry_time else 0.0,
                    })
                    in_position  = False
                    direction    = None
                    candles_held = 0
                    funding_acc  = 0.0

                    if target_dir == "flat":
                        equity_curve[ts] = capital
                        continue
                else:
                    # Same direction — mark to market
                    if direction == "long":
                        unrealised = (candle_close - entry_price) * position_size
                    else:
                        unrealised = (entry_price - candle_close) * position_size
                    equity_curve[ts] = capital + unrealised - funding_acc
                    continue

            # ---- Open new position -----------------------------------
            if target_dir in ("long", "short") and abs(score) > config.MIN_CONVICTION_SCORE:
                conviction = abs(score) / 100.0
                notional   = capital * config.LEVERAGE * conviction
                fill       = candle_close * (
                    1 + config.SLIPPAGE if target_dir == "long" else 1 - config.SLIPPAGE
                )
                if fill <= 0:
                    equity_curve[ts] = capital
                    continue

                size = notional / fill
                if size <= 0:
                    equity_curve[ts] = capital
                    continue

                fee = size * fill * config.EXCHANGE_FEE
                capital -= fee

                stop_price    = fill * (1 - HARD_STOP_PCT) if target_dir == "long" else fill * (1 + HARD_STOP_PCT)
                in_position   = True
                direction     = target_dir
                entry_price   = fill
                entry_time    = ts
                position_size = size
                fees_entry    = fee
                funding_acc   = 0.0
                candles_held  = 0

            if in_position:
                if direction == "long":
                    unrealised = (candle_close - entry_price) * position_size
                else:
                    unrealised = (entry_price - candle_close) * position_size
                equity_curve[ts] = capital + unrealised - funding_acc
            else:
                equity_curve[ts] = capital

        progress.update(sim_task, completed=n_bars)

    # Close any open position at end of data
    if in_position:
        last_ts    = bars_4h[-1]
        last_close = float(df_4h_ind.loc[last_ts, "close"])
        exit_fee   = position_size * last_close * config.EXCHANGE_FEE
        if direction == "long":
            gross = (last_close - entry_price) * position_size
        else:
            gross = (entry_price - last_close) * position_size
        net = gross - exit_fee - funding_acc
        capital += gross - exit_fee
        trades.append({
            "entry_time":     entry_time,
            "exit_time":      last_ts,
            "direction":      direction,
            "entry_price":    entry_price,
            "exit_price":     last_close,
            "size":           position_size,
            "pnl":            net,
            "pnl_pct":        (net / (capital - net)) * 100 if (capital - net) > 0 else 0.0,
            "exit_reason":    "end_of_data",
            "fees_paid":      fees_entry + exit_fee,
            "funding_paid":   funding_acc,
            "duration_hours": (pd.Timestamp(last_ts) - pd.Timestamp(entry_time)).total_seconds() / 3600 if entry_time else 0.0,
        })
        equity_curve[last_ts] = capital

    console.print(f"  Simulation complete — [bold]{len(trades)}[/bold] trades executed.")
    console.print()

    # ----------------------------------------------------------------
    # Step 4: Metrics & report
    # ----------------------------------------------------------------
    console.print("[bold]Step 4 / 4 — Computing metrics…[/bold]")

    equity_series = pd.Series(equity_curve)
    equity_series.index = pd.to_datetime(equity_series.index, utc=True)
    equity_series.sort_index(inplace=True)

    metrics = compute_metrics(trades, equity_series, initial_cap)

    console.rule("[bold cyan]Regime Backtest — Performance Report[/bold cyan]")
    console.print()

    # Metrics table
    table = Table(
        title="Regime Strategy Performance Metrics",
        box=box.DOUBLE_EDGE,
        title_style="bold cyan",
        show_header=True,
        header_style="bold magenta",
        expand=False,
        min_width=55,
    )
    table.add_column("Metric", style="bold", justify="left", min_width=28)
    table.add_column("Value",  justify="right", min_width=18)

    def row(label, val_str):
        table.add_row(label, val_str)

    pf  = metrics["profit_factor"]
    pf_str = "∞" if pf == float("inf") else f"{pf:.3f}"

    cap_init  = metrics["initial_capital"]
    cap_final = metrics["final_capital"]
    tot_ret   = metrics["total_return"]
    ann_ret   = metrics["annualized_return"]
    mdd       = metrics["max_drawdown"]
    sharpe    = metrics["sharpe_ratio"]
    sortino   = metrics["sortino_ratio"]

    row("Initial Capital",        f"[bold]${cap_init:,.2f}[/bold]")
    row("Final Capital",          f"[bold {'green' if cap_final >= cap_init else 'red'}]${cap_final:,.2f}[/bold {'green' if cap_final >= cap_init else 'red'}]")
    table.add_section()
    row("Total Return",           f"[{'green' if tot_ret >= 0 else 'red'}]{tot_ret:.2f} %[/{'green' if tot_ret >= 0 else 'red'}]")
    row("Annualised Return",      f"[{'green' if ann_ret >= 0 else 'red'}]{ann_ret:.2f} %[/{'green' if ann_ret >= 0 else 'red'}]")
    row("Max Drawdown",           f"[red]{mdd:.2f} %[/red]")
    table.add_section()
    row("Sharpe Ratio",           f"[{'green' if sharpe >= 1 else 'yellow' if sharpe >= 0 else 'red'}]{sharpe:.3f}[/{'green' if sharpe >= 1 else 'yellow' if sharpe >= 0 else 'red'}]")
    row("Sortino Ratio",          f"[{'green' if sortino >= 1 else 'yellow' if sortino >= 0 else 'red'}]{sortino:.3f}[/{'green' if sortino >= 1 else 'yellow' if sortino >= 0 else 'red'}]")
    row("Profit Factor",          f"[{'green' if pf == float('inf') or pf >= 1 else 'red'}]{pf_str}[/{'green' if pf == float('inf') or pf >= 1 else 'red'}]")
    table.add_section()
    row("Total Trades",           str(metrics["total_trades"]))
    row("Win Rate",               f"[{'green' if metrics['win_rate'] >= 50 else 'red'}]{metrics['win_rate']:.2f} %[/{'green' if metrics['win_rate'] >= 50 else 'red'}]")
    row("Avg Win",                f"[green]+${metrics['avg_win']:,.2f}[/green]")
    row("Avg Loss",               f"[red]-${metrics['avg_loss']:,.2f}[/red]")
    row("Expectancy (per trade)", f"[{'green' if metrics['expectancy'] >= 0 else 'red'}]${metrics['expectancy']:,.2f}[/{'green' if metrics['expectancy'] >= 0 else 'red'}]")
    table.add_section()
    row("Largest Win",            f"[green]+${metrics['largest_win']:,.2f}[/green]")
    row("Largest Loss",           f"[red]-${abs(metrics['largest_loss']):,.2f}[/red]")
    row("Avg Trade Duration",     f"{metrics['avg_trade_duration']:.1f} h")
    table.add_section()
    row("Total Fees Paid",        f"[yellow]-${metrics['total_fees_paid']:,.2f}[/yellow]")
    row("Total Funding Paid",     f"[yellow]-${metrics['total_funding_paid']:,.2f}[/yellow]")

    console.print(table)
    console.print()

    # Last N trades
    if trades:
        n_show = min(15, len(trades))
        trade_table = Table(
            title=f"Last {n_show} Trades",
            box=box.SIMPLE_HEAVY,
            title_style="bold cyan",
            show_header=True,
            header_style="bold magenta",
            expand=False,
        )
        trade_table.add_column("#",       justify="right", style="dim", min_width=4)
        trade_table.add_column("Dir",     justify="center", min_width=6)
        trade_table.add_column("Entry",   justify="center", min_width=17)
        trade_table.add_column("Exit",    justify="center", min_width=17)
        trade_table.add_column("Entry $", justify="right",  min_width=10)
        trade_table.add_column("Exit $",  justify="right",  min_width=10)
        trade_table.add_column("PnL",     justify="right",  min_width=12)
        trade_table.add_column("Reason",  justify="left",   min_width=12)

        reason_colors = {
            "hard_stop":   "red",
            "regime_flat": "yellow",
            "regime_flip": "yellow",
            "end_of_data": "dim",
        }

        start_i = len(trades) - n_show + 1
        for idx, t in enumerate(trades[-n_show:], start=start_i):
            pnl      = t["pnl"]
            color    = "green" if pnl >= 0 else "red"
            dir_str  = "[cyan]LONG[/cyan]" if t["direction"] == "long" else "[magenta]SHORT[/magenta]"
            reason   = t.get("exit_reason", "—")
            rc       = reason_colors.get(reason, "white")
            entry_str = fmt_ts(t["entry_time"])
            exit_str  = fmt_ts(t["exit_time"])

            trade_table.add_row(
                str(idx),
                dir_str,
                entry_str,
                exit_str,
                f"${t['entry_price']:,.2f}",
                f"${t['exit_price']:,.2f}",
                f"[{color}]{'+' if pnl >= 0 else ''}{pnl:,.2f}[/{color}]",
                f"[{rc}]{reason}[/{rc}]",
            )

        console.print(trade_table)
        console.print()

    console.print(Panel(
        "[dim]Regime backtest complete. "
        "Bar-by-bar simulation with no lookahead bias. "
        "Results are for simulation purposes only.[/dim]",
        border_style="dim",
        expand=False,
    ))
    console.print()


if __name__ == "__main__":
    run_regime_backtest()
