#!/usr/bin/env python3
# ============================================================
# optimize.py — Walk-forward Bayesian parameter optimizer
#
# Uses Optuna to search the parameter space against real
# Hyperliquid historical data, then reports the best params
# found on out-of-sample windows and writes them to
# config_optimized.py for human review before deployment.
#
# Architecture
# ------------
# 1. Fetch full price history for one or more anchor assets.
# 2. Precompute indicators once on each full dataset.
# 3. Split into overlapping walk-forward windows:
#      train: 9 months  →  test: 3 months  (slides 3 months each step)
# 4. Each Optuna trial runs the regime backtest on every window's
#    train set across all assets. Objective = mean score (fast).
# 5. After the study, evaluate the best params on every window's
#    *test* set (true out-of-sample performance).
# 6. Write results to config_optimized.py for review.
#
# Usage
# -----
#   python optimize.py                              # 150 trials, ETH
#   python optimize.py --assets BTC,ETH,SOL        # multi-asset objective
#   python optimize.py --trials 300                # more trials
#   python optimize.py --objective calmar          # calmar ratio objective
#   python optimize.py --no-write                  # report only
#
# Safety note
# -----------
# Never auto-deploy config_optimized.py to config.py without review.
# Optuna optimises for in-sample score; always check the OOS column
# and the per-window breakdown before accepting new params.
# ============================================================

import argparse
import types
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import optuna

import config
from data.fetcher import fetch_ohlcv_hl
from indicators.compute import add_indicators
from backtest.metrics import compute_metrics
from backtest.regime_backtest import simulate as _run_regime_backtest, HARD_STOP_PCT, WARMUP_BARS_4H

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Walk-forward window settings ─────────────────────────────────────────────

TRAIN_MONTHS = 9
TEST_MONTHS  = 3
SLIDE_MONTHS = 3

INITIAL_CAPITAL = config.PAPER_CAPITAL

# ── Parameter search space ────────────────────────────────────────────────────
# (name, lo, hi, step)  — step=None → continuous float; step=float → grid step

PARAM_SPACE = [
    # Core signal thresholds
    ("MIN_CONVICTION_SCORE",  12.0,  40.0,  None),
    ("ADX_TREND_THRESHOLD",   12.0,  30.0,  1.0),

    # Stop / TP / trail
    ("ATR_STOP_MULTIPLIER",    1.0,   4.5,  None),   # upper raised (was 3.5)
    ("ATR_TP_MULTIPLIER",      1.5,   5.0,  None),
    ("TRAIL_ACTIVATION_ATR",   1.0,   3.0,  None),   # floor 1.0 — must gain ≥1× ATR before trailing
    ("TRAIL_ATR_MULTIPLIER",   0.5,   2.5,  None),

    # Sizing
    ("KELLY_FRACTION",         0.03,  0.20, None),   # upper raised (was 0.12)
    ("TIER_CORR_FACTOR",       0.5,   0.95, None),

    # Volatility regime sizing (new)
    ("VOL_REGIME_LOW",         0.40,  0.90, None),   # pct threshold below which size shrinks
    ("VOL_REGIME_MIN",         0.30,  0.85, None),   # minimum size multiplier in low-vol

    # MA200 extension brake (new)
    ("MA200_NEAR_BAND",        0.05,  0.35, None),   # within this % → full size
    ("MA200_FAR_BAND",         0.20,  0.65, None),   # beyond this % → half size
]


# ── Config proxy ──────────────────────────────────────────────────────────────

def _make_cfg(**overrides) -> types.SimpleNamespace:
    """Build a config-like namespace from base config plus trial overrides."""
    base = {k: getattr(config, k) for k in dir(config) if not k.startswith("_")}
    base.update(overrides)
    return types.SimpleNamespace(**base)


# ── Objective scoring ─────────────────────────────────────────────────────────

def _sharpe(equity: pd.Series) -> float:
    """Annualised Sharpe from a 4h equity curve."""
    rets = equity.pct_change().dropna()
    if len(rets) < 10:
        return -9.0
    std = rets.std(ddof=1)
    if std <= 0:
        return -9.0
    return float(rets.mean() / std * np.sqrt(2190))   # 6 bars/day × 365


def _calmar(equity: pd.Series) -> float:
    """Calmar ratio: annualised return / max drawdown."""
    rets = equity.pct_change().dropna()
    if len(rets) < 10:
        return -9.0
    rolling_max = equity.cummax()
    max_dd = abs(float(((equity - rolling_max) / rolling_max).min()))
    if max_dd < 1e-6:
        return _sharpe(equity) * 2.0   # no drawdown — reward generously
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1)
    n_years   = len(rets) / 2190
    ann_ret   = float((1 + total_ret) ** (1 / max(n_years, 1e-3)) - 1)
    return ann_ret / max_dd


def _score(equity: pd.Series, obj_type: str) -> float:
    if obj_type == "calmar":
        return _calmar(equity)
    return _sharpe(equity)


# ── Walk-forward window generation ────────────────────────────────────────────

def _make_windows(df_4h: pd.DataFrame) -> list[dict]:
    start  = df_4h.index[0]
    end    = df_4h.index[-1]
    windows = []
    t = start
    while True:
        train_end  = t + pd.DateOffset(months=TRAIN_MONTHS)
        test_start = train_end
        test_end   = test_start + pd.DateOffset(months=TEST_MONTHS)
        if test_end > end:
            break
        windows.append({
            "train_start": t,
            "train_end":   train_end,
            "test_start":  test_start,
            "test_end":    test_end,
        })
        t += pd.DateOffset(months=SLIDE_MONTHS)
    return windows


def _slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
    return df[(df.index >= start) & (df.index < end)]


# ── Optuna objective ──────────────────────────────────────────────────────────

def _make_objective(
    assets_data: dict,   # {coin: (df_1d, df_4h, df_1h, df_15m)}
    windows: list[dict],
    obj_type: str,
):
    def objective(trial: optuna.Trial) -> float:
        # ── Sample parameters ──────────────────────────────────────────
        params = {}
        for name, lo, hi, step in PARAM_SPACE:
            if step is None:
                params[name] = trial.suggest_float(name, lo, hi)
            else:
                params[name] = float(trial.suggest_float(name, lo, hi, step=step))

        # ── Structural constraints ──────────────────────────────────────
        # MA200: far band must be wider than near band (otherwise sizing logic inverts)
        if params["MA200_FAR_BAND"] <= params["MA200_NEAR_BAND"] + 0.05:
            return -9.0
        # Vol regime: floor must be below the threshold (otherwise no scaling range)
        if params["VOL_REGIME_MIN"] >= params["VOL_REGIME_LOW"] - 0.05:
            return -9.0

        # ── Anti-overfit behavioural guards ────────────────────────────
        # Trail must activate after meaningful gain — prevents the optimizer
        # discovering "scalp everything immediately" as a degenerate solution.
        # TRAIL_ACTIVATION_ATR < 1.0 is already blocked by the search space floor,
        # but belt-and-suspenders here in case the space is ever widened.
        if params["TRAIL_ACTIVATION_ATR"] < 1.0:
            return -9.0
        # Trailing stop must be tighter than the initial ATR stop; if trail ≥ stop
        # the trailing stop can never improve on the original and the param is wasted.
        if params["TRAIL_ATR_MULTIPLIER"] >= params["ATR_STOP_MULTIPLIER"]:
            return -9.0
        # TP target must offer positive risk/reward vs the ATR stop.
        # Requiring TP > stop keeps the system in trend-following territory
        # and prevents the optimizer collapsing to a pure stop-loss strategy.
        if params["ATR_TP_MULTIPLIER"] < params["ATR_STOP_MULTIPLIER"]:
            return -9.0

        cfg = _make_cfg(**params)

        # ── Run on each asset × each training window ────────────────────
        all_scores = []
        for coin, (df_1d, df_4h, df_1h, df_15m) in assets_data.items():
            for w in windows:
                d1d  = _slice(df_1d,  w["train_start"], w["train_end"])
                d4h  = _slice(df_4h,  w["train_start"], w["train_end"])
                d1h  = _slice(df_1h,  w["train_start"], w["train_end"])
                d15m = _slice(df_15m, w["train_start"], w["train_end"])

                if len(d4h) < WARMUP_BARS_4H + 60:
                    continue

                result = _run_regime_backtest(d1d, d4h, d1h, d15m, cfg)
                if len(result["trades"]) < 5:
                    all_scores.append(-5.0)
                    continue

                all_scores.append(_score(result["equity_curve"], obj_type))

        return float(np.mean(all_scores)) if all_scores else -9.0

    return objective


# ── Out-of-sample evaluation ──────────────────────────────────────────────────

def _evaluate_oos(
    assets_data: dict,
    windows: list[dict],
    best_cfg,
) -> list[dict]:
    """
    Evaluate best params on each window's test set across all assets.
    Returns one row per window with metrics averaged across assets.
    """
    rows = []
    for w in windows:
        window_scores = []
        for coin, (df_1d, df_4h, df_1h, df_15m) in assets_data.items():
            d1d  = _slice(df_1d,  w["test_start"], w["test_end"])
            d4h  = _slice(df_4h,  w["test_start"], w["test_end"])
            d1h  = _slice(df_1h,  w["test_start"], w["test_end"])
            d15m = _slice(df_15m, w["test_start"], w["test_end"])

            if len(d4h) < WARMUP_BARS_4H + 20:
                continue

            result  = _run_regime_backtest(d1d, d4h, d1h, d15m, best_cfg)
            metrics = compute_metrics(result["trades"], result["equity_curve"], INITIAL_CAPITAL)
            window_scores.append(metrics)

        if not window_scores:
            continue

        # Average across assets for this window
        rows.append({
            "window":    f"{w['test_start'].strftime('%Y-%m')} → {w['test_end'].strftime('%Y-%m')}",
            "trades":    int(np.mean([m["total_trades"] for m in window_scores])),
            "sharpe":    float(np.mean([m["sharpe_ratio"]    for m in window_scores])),
            "return":    float(np.mean([m["total_return"]    for m in window_scores])),
            "drawdown":  float(np.mean([m["max_drawdown"]    for m in window_scores])),
            "win_rate":  float(np.mean([m["win_rate"]        for m in window_scores])),
            "n_assets":  len(window_scores),
        })
    return rows


# ── Baseline score ────────────────────────────────────────────────────────────

def _baseline_score(assets_data: dict, obj_type: str) -> float:
    scores = []
    for coin, (df_1d, df_4h, df_1h, df_15m) in assets_data.items():
        result = _run_regime_backtest(df_1d, df_4h, df_1h, df_15m, config)
        scores.append(_score(result["equity_curve"], obj_type))
    return float(np.mean(scores)) if scores else 0.0


# ── Report / output ───────────────────────────────────────────────────────────

def _print_results(
    best_params: dict,
    oos_rows: list[dict],
    current_score: float,
    obj_type: str,
    asset_names: list[str],
):
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    console = Console()
    console.print()

    obj_label = "Sharpe" if obj_type == "sharpe" else "Calmar"

    # ── OOS performance table ──────────────────────────────────────────
    multi = len(asset_names) > 1
    title_suffix = f"  [dim](avg across {', '.join(asset_names)})[/dim]" if multi else ""
    t = Table(
        title=f"[bold cyan]Out-of-Sample Performance (best params)[/bold cyan]{title_suffix}",
        box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta",
    )
    t.add_column("Window",    min_width=20)
    t.add_column("Trades",    min_width=7,  justify="right")
    t.add_column("Sharpe",    min_width=8,  justify="right")
    t.add_column("Return %",  min_width=9,  justify="right")
    t.add_column("Max DD %",  min_width=9,  justify="right")
    t.add_column("Win Rate",  min_width=9,  justify="right")

    for r in oos_rows:
        sc = "green" if r["sharpe"] >= 1 else ("yellow" if r["sharpe"] >= 0 else "red")
        rc = "green" if r["return"] >= 0 else "red"
        t.add_row(
            r["window"],
            str(r["trades"]),
            f"[{sc}]{r['sharpe']:.3f}[/{sc}]",
            f"[{rc}]{r['return']:+.1f}%[/{rc}]",
            f"[red]{r['drawdown']:.1f}%[/red]",
            f"{r['win_rate']:.1f}%",
        )

    if oos_rows:
        mean_sharpe = float(np.mean([r["sharpe"] for r in oos_rows]))
        mean_ret    = float(np.mean([r["return"]  for r in oos_rows]))
        sc = "green" if mean_sharpe >= 1 else ("yellow" if mean_sharpe >= 0 else "red")
        rc = "green" if mean_ret    >= 0 else "red"
        t.add_section()
        t.add_row(
            "[bold]Mean OOS[/bold]", "",
            f"[bold {sc}]{mean_sharpe:.3f}[/bold {sc}]",
            f"[bold {rc}]{mean_ret:+.1f}%[/bold {rc}]",
            "", "",
        )

    console.print(t)
    console.print()

    # ── Best params table ──────────────────────────────────────────────
    p = Table(
        title="[bold cyan]Optimised Parameters[/bold cyan]",
        box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta",
    )
    p.add_column("Parameter",  min_width=24)
    p.add_column("Current",    min_width=10, justify="right")
    p.add_column("Optimised",  min_width=10, justify="right")
    p.add_column("Change",     min_width=10, justify="right")

    for name, *_ in PARAM_SPACE:
        current_val = getattr(config, name, "N/A")
        new_val     = best_params[name]
        if isinstance(current_val, (int, float)):
            delta     = new_val - float(current_val)
            delta_str = f"[{'green' if delta >= 0 else 'red'}]{delta:+.4f}[/{'green' if delta >= 0 else 'red'}]"
            p.add_row(name, f"{current_val:.4f}", f"[bold]{new_val:.4f}[/bold]", delta_str)
        else:
            p.add_row(name, str(current_val), f"[bold]{new_val}[/bold]", "")

    console.print(p)
    console.print()

    # ── Summary panel ──────────────────────────────────────────────────
    if oos_rows:
        mean_s  = float(np.mean([r["sharpe"] for r in oos_rows]))
        verdict = (
            "[bold green]LOOKS GOOD — consider deploying[/bold green]"
            if mean_s > current_score
            else "[bold yellow]MARGINAL — review carefully before deploying[/bold yellow]"
        )
        console.print(Panel(
            f"  Current (live) config {obj_label} baseline: [bold]{current_score:.3f}[/bold]\n"
            f"  Mean OOS Sharpe (optimised):           [bold]{mean_s:.3f}[/bold]\n\n"
            f"  {verdict}\n\n"
            f"  [dim]Results written to config_optimized.py — review before copying to config.py[/dim]",
            title="[bold]Verdict[/bold]", border_style="cyan", expand=False,
        ))
    console.print()


def _write_config(
    best_params: dict,
    oos_rows: list[dict],
    asset_names: list[str],
    n_trials: int,
    obj_type: str,
):
    ts       = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    mean_oos = float(np.mean([r["sharpe"] for r in oos_rows])) if oos_rows else 0.0
    assets_s = ",".join(asset_names)

    lines = [
        "# ============================================================",
        "# config_optimized.py — Auto-generated by optimize.py",
        f"# Generated  : {ts}",
        f"# Assets     : {assets_s}",
        f"# Trials     : {n_trials}",
        f"# Objective  : {obj_type}",
        f"# Mean OOS Sharpe : {mean_oos:.4f}",
        "#",
        "# REVIEW BEFORE DEPLOYING — copy desired values into config.py",
        "# ============================================================",
        "",
        "# Walk-forward OOS window results:",
    ]
    for r in oos_rows:
        lines.append(
            f"#   {r['window']}  trades={r['trades']}  "
            f"sharpe={r['sharpe']:.3f}  return={r['return']:+.1f}%  "
            f"dd={r['drawdown']:.1f}%"
        )

    lines += ["", "# ── Optimised parameters ──────────────────────────────────────"]
    for name, *_ in PARAM_SPACE:
        current_val = getattr(config, name, None)
        new_val     = best_params[name]
        suffix      = f"  # was {current_val}"
        if isinstance(new_val, float):
            lines.append(f"{name:<28} = {new_val:.6f}{suffix}")
        else:
            lines.append(f"{name:<28} = {new_val}{suffix}")

    lines.append("")
    with open("config_optimized.py", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  Written to config_optimized.py")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Walk-forward parameter optimiser")
    parser.add_argument("--trials",    type=int,   default=150,
                        help="Number of Optuna trials (default: 150)")
    parser.add_argument("--assets",    type=str,   default="ETH",
                        help="Comma-separated anchor assets, e.g. BTC,ETH,SOL (default: ETH)")
    parser.add_argument("--objective", type=str,   default="sharpe",
                        choices=["sharpe", "calmar"],
                        help="Optimisation objective: sharpe or calmar (default: sharpe)")
    parser.add_argument("--no-write",  action="store_true",
                        help="Skip writing config_optimized.py")
    # Backward-compat alias
    parser.add_argument("--coin",      type=str,   default=None,
                        help="Alias for --assets (single coin)")
    args = parser.parse_args()

    if args.coin:
        asset_names = [args.coin.upper()]
    else:
        asset_names = [a.strip().upper() for a in args.assets.split(",") if a.strip()]

    obj_type   = args.objective
    obj_label  = "Sharpe" if obj_type == "sharpe" else "Calmar"
    multi      = len(asset_names) > 1
    assets_str = ", ".join(asset_names)

    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    console = Console()

    console.print()
    console.print(Panel(
        f"[bold cyan]Walk-Forward Parameter Optimiser[/bold cyan]\n"
        f"[dim]Assets: {assets_str}  |  "
        f"Trials: {args.trials}  |  "
        f"Objective: {obj_label}  |  "
        f"Windows: {TRAIN_MONTHS}m train / {TEST_MONTHS}m test / {SLIDE_MONTHS}m slide[/dim]",
        border_style="cyan", expand=False,
    ))
    console.print()

    # ── Step 1: Fetch data for all assets ─────────────────────────────
    console.print("[bold]Step 1 / 4 — Fetching historical data…[/bold]")

    assets_data: dict = {}

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), TimeElapsedColumn(), console=console, transient=True) as prog:

        def fetch(coin, tf, days, label):
            t = prog.add_task(f"[{coin}] Fetching {label}…", total=None)
            try:
                df = fetch_ohlcv_hl(coin, tf, days)
            except Exception as exc:
                console.print(f"[yellow][{coin}] {label} fetch failed ({exc}), using empty frame[/yellow]")
                df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            prog.update(t, description=f"[green][{coin}] {label}: {len(df):,} bars[/green]", completed=True)
            return df

        for coin in asset_names:
            df_1d_raw  = fetch(coin, "1d",  config.LOOKBACK_DAYS,       "Daily")
            df_4h_raw  = fetch(coin, "4h",  config.LOOKBACK_DAYS,       "4h")
            df_1h_raw  = fetch(coin, "1h",  60,                         "1h")
            df_15m_raw = fetch(coin, "15m", config.SCALP_LOOKBACK_DAYS, "15m")
            assets_data[coin] = (df_1d_raw, df_4h_raw, df_1h_raw, df_15m_raw)

    for coin, (d1d, d4h, d1h, d15m) in assets_data.items():
        if not d4h.empty:
            console.print(
                f"  [bold]{coin}[/bold]  "
                f"[green]Daily:[/green] {len(d1d):,}  "
                f"[green]4h:[/green] {len(d4h):,}  "
                f"[green]1h:[/green] {len(d1h):,}  "
                f"[green]15m:[/green] {len(d15m):,}"
            )
    console.print()

    # ── Step 2: Pre-compute indicators ────────────────────────────────
    console.print("[bold]Step 2 / 4 — Pre-computing indicators…[/bold]")

    ind_data: dict = {}
    for coin, (d1d, d4h, d1h, d15m) in assets_data.items():
        def ind(df):
            return add_indicators(df, config) if not df.empty else df
        ind_data[coin] = (ind(d1d), ind(d4h), ind(d1h), ind(d15m))

    console.print("  [green]Done.[/green]")
    console.print()

    # ── Step 3: Walk-forward windows (use first asset's 4h as reference) ──
    ref_4h = ind_data[asset_names[0]][1]
    windows = _make_windows(ref_4h)
    if not windows:
        console.print("[red]Not enough data to form walk-forward windows. Need ≥12 months.[/red]")
        return

    n_backtests = args.trials * len(windows) * len(asset_names)
    console.print(
        f"[bold]Step 3 / 4 — Running Optuna ({args.trials} trials, "
        f"{len(windows)} windows, {len(asset_names)} asset(s))…[/bold]"
    )
    console.print(
        f"  [dim]Total backtests: {n_backtests:,}  |  "
        f"Objective: {obj_label}[/dim]"
    )
    console.print(f"  [dim]Baseline {obj_label} (current config)… computing…[/dim]", end="")
    current_score = _baseline_score(ind_data, obj_type)
    console.print(f"\r  Baseline {obj_label} (current config): [bold]{current_score:.3f}[/bold]")
    console.print()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
    )

    objective = _make_objective(ind_data, windows, obj_type)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Optimising…", total=args.trials)

        def _callback(study, trial):
            prog.advance(task)
            prog.update(
                task,
                description=(
                    f"Optimising…  best {obj_label} = "
                    f"[bold green]{study.best_value:.3f}[/bold green]"
                ),
            )

        study.optimize(objective, n_trials=args.trials, callbacks=[_callback], show_progress_bar=False)

    console.print()
    best_params = study.best_params
    console.print(f"  [green]Best in-sample mean {obj_label}: {study.best_value:.3f}[/green]")
    console.print()

    # ── Step 4: Out-of-sample evaluation ──────────────────────────────
    console.print("[bold]Step 4 / 4 — Evaluating best params out-of-sample…[/bold]")
    best_cfg = _make_cfg(**best_params)
    oos_rows = _evaluate_oos(ind_data, windows, best_cfg)
    console.print("  [green]Done.[/green]")
    console.print()

    # ── Print and write results ────────────────────────────────────────
    _print_results(best_params, oos_rows, current_score, obj_type, asset_names)

    if not args.no_write:
        _write_config(best_params, oos_rows, asset_names, args.trials, obj_type)

    console.print("[dim]Run with --no-write to skip writing config_optimized.py[/dim]\n")


if __name__ == "__main__":
    main()
