# ============================================================
# backtest/regime_backtest.py — Regime strategy backtester
#
# Two public surfaces:
#
#   simulate(df_1d, df_4h, df_1h, df_15m, cfg) -> dict
#       Pure simulation on pre-fetched, pre-indicator DataFrames.
#       Used by both main.py and optimize.py so the logic is DRY.
#
#   run(coin, verbose) -> dict
#       Full pipeline: fetch → indicators → simulate → report.
#       Called by main.py.
# ============================================================

import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
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

console = Console()

HARD_STOP_PCT   = 0.08   # hard cap on ATR-based stop (matches always_in_runner)
MAX_NOTIONAL_X  = 2.0    # max position = 2× current equity (single-asset)
WARMUP_BARS_4H  = 80     # bars before we start scoring (indicator warmup)


# ── Sizing helpers (mirror live/always_in_runner without imports) ─────────────

def _vol_size_factor_bt(snap, cfg) -> float:
    """Reduce size in low-vol regimes (mirrors _vol_size_factor in always_in_runner)."""
    pct  = snap.norm_atr_pct
    low  = getattr(cfg, "VOL_REGIME_LOW",  0.70)
    minf = getattr(cfg, "VOL_REGIME_MIN",  0.60)
    if pct <= 0 or (isinstance(pct, float) and np.isnan(pct)):
        return 1.0
    if pct >= low:
        return 1.0
    return minf + (1.0 - minf) * (pct / low)


def _ma200_size_factor_bt(snap, cfg) -> float:
    """Penalise size when price is stretched far from MA200 (mirrors always_in_runner)."""
    dist = abs(snap.ma200_dist)
    near = getattr(cfg, "MA200_NEAR_BAND", 0.15)
    far  = getattr(cfg, "MA200_FAR_BAND",  0.35)
    if dist <= 0 or (isinstance(dist, float) and np.isnan(dist)):
        return 1.0
    if dist <= near:
        return 1.0
    if dist >= far:
        return 0.5
    frac = (dist - near) / (far - near)
    return 1.0 - 0.5 * frac


# ── Pure simulation ───────────────────────────────────────────────────────────

def simulate(
    df_1d:  pd.DataFrame,
    df_4h:  pd.DataFrame,
    df_1h:  pd.DataFrame,
    df_15m: pd.DataFrame,
    cfg,
    initial_capital: float = None,
) -> dict:
    """
    Bar-by-bar regime backtest on pre-computed indicator DataFrames.

    Matches the live always_in_runner logic:
      - Regime score drives direction (long/short/flat).
      - Kelly sizing scaled by conviction, capped at MAX_NOTIONAL_X × equity.
      - ATR-based stop, capped at HARD_STOP_PCT.
      - Trailing stop activated after TRAIL_ACTIVATION_ATR gain.
      - Exits: hard/trail stop, regime flip, regime flat.

    All DataFrames must have indicators already applied (add_indicators).
    Returns the same dict shape as BacktestEngine.run().
    """
    if initial_capital is None:
        initial_capital = getattr(cfg, "PAPER_CAPITAL", 10_000.0)

    engine    = RegimeEngine(cfg)
    capital   = initial_capital
    trades:   list[dict] = []
    eq_curve: dict = {}

    pos: Optional[dict] = None  # current open position

    for i in range(WARMUP_BARS_4H, len(df_4h)):
        ts    = df_4h.index[i]
        row4h = df_4h.iloc[i]

        c_open  = float(row4h["open"])
        c_high  = float(row4h["high"])
        c_low   = float(row4h["low"])
        c_close = float(row4h["close"])

        # ── Slice all frames to bars up to ts (no lookahead) ──────────
        d1d  = df_1d[df_1d.index   <= ts]
        d4h  = df_4h.iloc[:i + 1]
        d1h  = df_1h[df_1h.index   <= ts] if not df_1h.empty  else df_1h
        d15m = df_15m[df_15m.index <= ts] if not df_15m.empty else df_15m

        if d1d.empty or d4h.empty:
            eq_curve[ts] = capital
            continue

        # ── Regime score ───────────────────────────────────────────────
        try:
            snap = engine.compute(d1d, d4h, d1h, d15m)
        except Exception:
            eq_curve[ts] = capital
            continue

        score = snap.score
        price = float(c_close) if snap.latest_price <= 0 else snap.latest_price
        atr   = snap.latest_atr

        if price <= 0:
            eq_curve[ts] = capital
            continue

        # ── Manage open position ───────────────────────────────────────
        if pos is not None:
            # Update trailing stop from this candle's intraday range
            t_atr = pos["trail_atr"]
            if t_atr > 0:
                act  = t_atr * cfg.TRAIL_ACTIVATION_ATR
                dist = t_atr * cfg.TRAIL_ATR_MULTIPLIER
                if pos["direction"] == "long":
                    if c_high - pos["entry_price"] >= act:
                        cand = c_high - dist
                        if pos["trailing_stop"] is None or cand > pos["trailing_stop"]:
                            pos["trailing_stop"] = cand
                else:
                    if pos["entry_price"] - c_low >= act:
                        cand = c_low + dist
                        if pos["trailing_stop"] is None or cand < pos["trailing_stop"]:
                            pos["trailing_stop"] = cand

            # Effective stop = tighter of hard or trail
            eff_stop = pos["stop_loss"]
            if pos["trailing_stop"] is not None:
                eff_stop = (
                    max(eff_stop, pos["trailing_stop"]) if pos["direction"] == "long"
                    else min(eff_stop, pos["trailing_stop"])
                )

            stop_hit = (
                (pos["direction"] == "long"  and c_low  <= eff_stop) or
                (pos["direction"] == "short" and c_high >= eff_stop)
            )
            if stop_hit:
                reason = "trail_stop" if pos["trailing_stop"] is not None else "hard_stop"
                trade  = _close(pos, eff_stop, reason, ts, cfg, capital)
                capital += trade["pnl"]
                trades.append(trade)
                pos = None

        # ── Target direction ───────────────────────────────────────────
        if score > cfg.MIN_CONVICTION_SCORE:
            target = "long"
        elif score < -cfg.MIN_CONVICTION_SCORE:
            target = "short"
        else:
            target = "flat"

        # ── Close on regime flip / flat ────────────────────────────────
        if pos is not None and pos["direction"] != target:
            trade   = _close(pos, price, "regime_exit", ts, cfg, capital)
            capital += trade["pnl"]
            trades.append(trade)
            pos = None

        # ── Open new position ──────────────────────────────────────────
        if pos is None and target != "flat":
            slip = cfg.SLIPPAGE if target == "long" else -cfg.SLIPPAGE
            fill = price * (1 + slip)

            stop_dist = (
                min(atr * cfg.ATR_STOP_MULTIPLIER, fill * HARD_STOP_PCT)
                if atr > 0 else fill * HARD_STOP_PCT
            )

            conviction = abs(score) / 100.0
            notional   = capital * cfg.KELLY_FRACTION * conviction / HARD_STOP_PCT
            notional   = min(notional, capital * MAX_NOTIONAL_X)

            # Vol regime and MA200 size factors (same logic as live runner)
            notional  *= _vol_size_factor_bt(snap, cfg)
            notional  *= _ma200_size_factor_bt(snap, cfg)

            size       = notional / fill if fill > 0 else 0.0

            if size <= 0:
                eq_curve[ts] = capital
                continue

            fee      = size * fill * cfg.EXCHANGE_FEE
            capital -= fee

            pos = {
                "direction":    target,
                "entry_price":  fill,
                "entry_time":   ts,
                "size":         size,
                "stop_loss":    (fill - stop_dist) if target == "long" else (fill + stop_dist),
                "trail_atr":    atr,
                "trailing_stop": None,
                "entry_fee":    fee,
            }

        # ── Mark to market ─────────────────────────────────────────────
        if pos is not None:
            upnl = (
                (price - pos["entry_price"]) * pos["size"]
                if pos["direction"] == "long"
                else (pos["entry_price"] - price) * pos["size"]
            )
            eq_curve[ts] = capital + upnl
        else:
            eq_curve[ts] = capital

    # Close any still-open position at last bar
    if pos is not None and len(df_4h) > 0:
        last_ts    = df_4h.index[-1]
        last_close = float(df_4h.iloc[-1]["close"])
        trade      = _close(pos, last_close, "end_of_data", last_ts, cfg, capital)
        capital   += trade["pnl"]
        trades.append(trade)
        eq_curve[last_ts] = capital

    eq_series = pd.Series(eq_curve)
    eq_series.index = pd.to_datetime(eq_series.index, utc=True)
    eq_series.sort_index(inplace=True)

    return {
        "trades":        trades,
        "equity_curve":  eq_series,
        "final_capital": capital,
    }


def _close(pos: dict, exit_price: float, reason: str, ts, cfg, capital: float) -> dict:
    size     = pos["size"]
    fee_exit = size * exit_price * cfg.EXCHANGE_FEE
    gross    = (
        (exit_price - pos["entry_price"]) * size if pos["direction"] == "long"
        else (pos["entry_price"] - exit_price) * size
    )
    net_pnl  = gross - fee_exit
    pnl_pct  = (net_pnl / capital * 100) if capital > 0 else 0.0
    try:
        dur_h = (pd.Timestamp(ts) - pd.Timestamp(pos["entry_time"])).total_seconds() / 3600
    except Exception:
        dur_h = 0.0
    return {
        "entry_time":     pos["entry_time"],
        "exit_time":      ts,
        "direction":      pos["direction"],
        "entry_price":    pos["entry_price"],
        "exit_price":     exit_price,
        "size":           size,
        "pnl":            net_pnl,
        "pnl_pct":        pnl_pct,
        "exit_reason":    reason,
        "fees_paid":      pos.get("entry_fee", 0.0) + fee_exit,
        "funding_paid":   0.0,
        "duration_hours": dur_h,
    }


# ── Full pipeline (fetch → indicators → simulate → report) ───────────────────

def run(coin: str = None, verbose: bool = True) -> dict:
    """
    Fetch data, compute indicators, simulate, and print a full report.
    Returns the results dict from simulate().
    """
    if coin is None:
        coin = getattr(config, "BACKTEST_COIN", "ETH")

    if verbose:
        console.print()
        console.print(Panel(
            "[bold cyan]Regime Strategy Backtester[/bold cyan]\n"
            "[dim]Always-in conviction-weighted positioning — ATR stops + trailing stops[/dim]\n"
            f"[dim]Asset: {coin}  |  Entry TF: 4h  |  Trend TF: 1d  |  "
            f"Lookback: {config.LOOKBACK_DAYS} days[/dim]",
            border_style="cyan", expand=False,
        ))
        console.print()

    # ── Step 1: Fetch ─────────────────────────────────────────────────
    if verbose:
        console.print("[bold]Step 1 / 4 — Fetching historical data…[/bold]")

    def _fetch(interval: str, days: int) -> pd.DataFrame:
        since_ms = int(
            (datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp() * 1000
        )
        return fetch_hl_candles(coin, interval, since_ms)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), TimeElapsedColumn(),
                  console=console, transient=True) as prog:

        def fetch_tf(interval, days, label):
            t = prog.add_task(f"Fetching {label}…", total=None)
            try:
                df = _fetch(interval, days)
            except Exception as exc:
                console.print(f"[yellow]{label} fetch failed ({exc}), using empty frame[/yellow]")
                df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            prog.update(t, description=f"[green]{label}: {len(df):,} bars[/green]", completed=True)
            return df

        df_1d_raw  = fetch_tf("1d",  config.LOOKBACK_DAYS, "Daily")
        df_4h_raw  = fetch_tf("4h",  config.LOOKBACK_DAYS, "4h")
        # HL 1h history is typically ~90 days; try full lookback, fall back to 90d
        df_1h_raw  = fetch_tf("1h",  config.LOOKBACK_DAYS, "1h")
        if df_1h_raw.empty:
            df_1h_raw = fetch_tf("1h", 90, "1h (90d fallback)")
        df_15m_raw = fetch_tf("15m", min(config.LOOKBACK_DAYS, 60), "15m")

    if verbose:
        console.print(
            f"  [green]Daily:[/green] {len(df_1d_raw):,} bars  "
            f"({df_1d_raw.index[0].date() if not df_1d_raw.empty else '?'} → "
            f"{df_1d_raw.index[-1].date() if not df_1d_raw.empty else '?'})\n"
            f"  [green]4h:[/green]    {len(df_4h_raw):,} bars  |  "
            f"[green]1h:[/green] {len(df_1h_raw):,} bars  |  "
            f"[green]15m:[/green] {len(df_15m_raw):,} bars"
        )
        console.print()

    # ── Step 2: Indicators ────────────────────────────────────────────
    if verbose:
        console.print("[bold]Step 2 / 4 — Computing indicators…[/bold]")

    def ind(df):
        return add_indicators(df, config) if not df.empty else df

    df_1d  = ind(df_1d_raw)
    df_4h  = ind(df_4h_raw)
    df_1h  = ind(df_1h_raw)
    df_15m = ind(df_15m_raw)

    if verbose:
        console.print("  [green]Done.[/green]")
        console.print()

    # ── Step 3: Simulate ──────────────────────────────────────────────
    if verbose:
        console.print("[bold]Step 3 / 4 — Running bar-by-bar simulation…[/bold]")
        console.print(f"  [dim]{len(df_4h):,} 4h bars to process…[/dim]")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), TextColumn("{task.percentage:>3.0f}%"),
                  TimeElapsedColumn(), console=console, transient=True) as prog:

        task = prog.add_task("Simulating…", total=len(df_4h) - WARMUP_BARS_4H)

        # Wrap simulate() with a progress hook by monkey-patching isn't clean —
        # instead, run it and just show a spinner (it's fast enough, ~10-30s).
        prog.update(task, description="Simulating…")
        results = simulate(df_1d, df_4h, df_1h, df_15m, config)
        prog.update(task, completed=len(df_4h) - WARMUP_BARS_4H)

    trades      = results["trades"]
    equity      = results["equity_curve"]
    final_cap   = results["final_capital"]

    if verbose:
        console.print(f"  [green]Done.[/green]  {len(trades)} trades executed.")
        console.print()

    # ── Step 4: Report ────────────────────────────────────────────────
    if verbose:
        console.print("[bold]Step 4 / 4 — Computing metrics…[/bold]")
        console.print()

    metrics = compute_metrics(trades, equity, config.PAPER_CAPITAL)

    if verbose:
        _print_report(metrics, trades, equity, coin)

    return results


# ── Reporting helpers ─────────────────────────────────────────────────────────

def _print_report(metrics: dict, trades: list, equity: pd.Series, coin: str):
    console.rule("[bold cyan]Regime Backtest — Performance Report[/bold cyan]")
    console.print()

    # ── Metrics table ──────────────────────────────────────────────────
    tbl = Table(
        title=f"Regime Strategy Performance — {coin}",
        box=box.DOUBLE_EDGE, title_style="bold cyan",
        show_header=True, header_style="bold magenta",
        expand=False, min_width=55,
    )
    tbl.add_column("Metric", style="bold", justify="left",  min_width=28)
    tbl.add_column("Value",                justify="right", min_width=18)

    def row(label, val):
        tbl.add_row(label, val)

    ic  = metrics["initial_capital"]
    fc  = metrics["final_capital"]
    tr  = metrics["total_return"]
    ar  = metrics["annualized_return"]
    mdd = metrics["max_drawdown"]
    sh  = metrics["sharpe_ratio"]
    so  = metrics["sortino_ratio"]
    pf  = metrics["profit_factor"]
    pf_str = "∞" if pf == float("inf") else f"{pf:.3f}"

    row("Initial Capital",        f"[bold]${ic:,.2f}[/bold]")
    row("Final Capital",          f"[bold {'green' if fc >= ic else 'red'}]${fc:,.2f}[/bold {'green' if fc >= ic else 'red'}]")
    tbl.add_section()
    row("Total Return",           f"[{'green' if tr >= 0 else 'red'}]{tr:+.2f}%[/{'green' if tr >= 0 else 'red'}]")
    row("Annualised Return",      f"[{'green' if ar >= 0 else 'red'}]{ar:+.2f}%[/{'green' if ar >= 0 else 'red'}]")
    row("Max Drawdown",           f"[red]{mdd:.2f}%[/red]")
    tbl.add_section()
    row("Sharpe Ratio",           f"[{'green' if sh >= 1 else 'yellow' if sh >= 0 else 'red'}]{sh:.3f}[/{'green' if sh >= 1 else 'yellow' if sh >= 0 else 'red'}]")
    row("Sortino Ratio",          f"[{'green' if so >= 1 else 'yellow' if so >= 0 else 'red'}]{so:.3f}[/{'green' if so >= 1 else 'yellow' if so >= 0 else 'red'}]")
    row("Profit Factor",          f"[{'green' if pf == float('inf') or pf >= 1 else 'red'}]{pf_str}[/{'green' if pf == float('inf') or pf >= 1 else 'red'}]")
    tbl.add_section()
    row("Total Trades",           str(metrics["total_trades"]))
    row("Win Rate",               f"[{'green' if metrics['win_rate'] >= 50 else 'red'}]{metrics['win_rate']:.1f}%[/{'green' if metrics['win_rate'] >= 50 else 'red'}]")
    row("Avg Win",                f"[green]+${metrics['avg_win']:,.2f}[/green]")
    row("Avg Loss",               f"[red]-${metrics['avg_loss']:,.2f}[/red]")
    row("Expectancy (per trade)", f"[{'green' if metrics['expectancy'] >= 0 else 'red'}]${metrics['expectancy']:+,.2f}[/{'green' if metrics['expectancy'] >= 0 else 'red'}]")
    tbl.add_section()
    row("Largest Win",            f"[green]+${metrics['largest_win']:,.2f}[/green]")
    row("Largest Loss",           f"[red]-${abs(metrics['largest_loss']):,.2f}[/red]")
    row("Avg Trade Duration",     f"{metrics['avg_trade_duration']:.1f} h")
    tbl.add_section()
    row("Total Fees Paid",        f"[yellow]-${metrics['total_fees_paid']:,.2f}[/yellow]")
    row("Total Funding Paid",     f"[yellow]-${metrics['total_funding_paid']:,.2f}[/yellow]")

    console.print(tbl)
    console.print()

    # ── Exit reason breakdown ──────────────────────────────────────────
    if trades:
        from collections import Counter
        reasons = Counter(t.get("exit_reason", "?") for t in trades)
        parts = []
        colors = {"hard_stop": "red", "trail_stop": "yellow",
                  "regime_exit": "cyan", "end_of_data": "dim"}
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            c = colors.get(reason, "white")
            pct = count / len(trades) * 100
            parts.append(f"[{c}]{reason}[/{c}]: {count} ({pct:.0f}%)")
        console.print("  Exit reasons: " + "  |  ".join(parts))
        console.print()

    # ── Recent trades ──────────────────────────────────────────────────
    if trades:
        n_show = min(20, len(trades))
        tt = Table(
            title=f"Last {n_show} Trades",
            box=box.SIMPLE_HEAVY, title_style="bold cyan",
            show_header=True, header_style="bold magenta", expand=False,
        )
        tt.add_column("#",        justify="right",  style="dim", min_width=4)
        tt.add_column("Dir",      justify="center",              min_width=6)
        tt.add_column("Entry",    justify="center",              min_width=17)
        tt.add_column("Exit",     justify="center",              min_width=17)
        tt.add_column("Entry $",  justify="right",               min_width=10)
        tt.add_column("Exit $",   justify="right",               min_width=10)
        tt.add_column("PnL",      justify="right",               min_width=12)
        tt.add_column("PnL %",    justify="right",               min_width=8)
        tt.add_column("Reason",   justify="left",                min_width=12)

        reason_colors = {
            "hard_stop":   "red",
            "trail_stop":  "yellow",
            "regime_exit": "cyan",
            "end_of_data": "dim",
        }
        start_i = len(trades) - n_show + 1
        for idx, t in enumerate(trades[-n_show:], start=start_i):
            pnl    = t["pnl"]
            col    = "green" if pnl >= 0 else "red"
            dir_s  = "[cyan]LONG[/cyan]" if t["direction"] == "long" else "[magenta]SHORT[/magenta]"
            reason = t.get("exit_reason", "—")
            rc     = reason_colors.get(reason, "white")
            tt.add_row(
                str(idx), dir_s,
                fmt_ts(t["entry_time"]), fmt_ts(t["exit_time"]),
                f"${t['entry_price']:,.2f}", f"${t['exit_price']:,.2f}",
                f"[{col}]{pnl:+,.2f}[/{col}]",
                f"[{col}]{t['pnl_pct']:+.2f}%[/{col}]",
                f"[{rc}]{reason}[/{rc}]",
            )

        console.print(tt)
        console.print()

    # ── Equity curve ───────────────────────────────────────────────────
    _draw_equity_curve(equity)

    console.print(Panel(
        "[dim]Regime backtest complete. Bar-by-bar simulation — no lookahead bias.\n"
        "Results are for simulation purposes only.[/dim]",
        border_style="dim", expand=False,
    ))
    console.print()


def _draw_equity_curve(equity: pd.Series, width: int = 70, height: int = 14):
    if equity.empty:
        return
    values = equity.resample("D").last().dropna().values.astype(float)
    if len(values) < 2:
        return
    if len(values) > width:
        idx    = np.linspace(0, len(values) - 1, width, dtype=int)
        values = values[idx]

    mn, mx = values.min(), values.max()
    span   = mx - mn if mx != mn else 1.0
    normed = ((values - mn) / span * (height - 1)).astype(int)

    console.print(Panel("Equity Curve (daily resampled)", style="bold cyan", expand=False))
    label_rows = {height - 1: f"${mx:>10,.0f}", height // 2: f"${(mn + span / 2):>10,.0f}", 0: f"${mn:>10,.0f}"}

    for row_i in range(height - 1, -1, -1):
        chars = []
        for col, col_val in enumerate(normed):
            color = "bright_green" if values[col] >= values[0] else "bright_red"
            dim   = "green"        if values[col] >= values[0] else "red"
            if col_val == row_i:
                chars.append(f"[{color}]█[/{color}]")
            elif col_val > row_i:
                chars.append(f"[{dim}]│[/{dim}]")
            else:
                chars.append(" ")
        label = label_rows.get(row_i, " " * 12)
        console.print(f"  [dim]{label}[/dim]  {''.join(chars)}")

    x_l = equity.index[0].strftime("%Y-%m-%d")
    x_r = equity.index[-1].strftime("%Y-%m-%d")
    console.print(f"  {'':14}{x_l}{'':>{width - len(x_l) - 2}}{x_r}")
    console.print()


if __name__ == "__main__":
    run()
