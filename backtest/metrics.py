# ============================================================
# backtest/metrics.py — Performance metric computation
# ============================================================

import pandas as pd
import numpy as np
from typing import List


def compute_metrics(
    trades: List[dict],
    equity_curve: pd.Series,
    initial_capital: float,
) -> dict:
    """
    Compute comprehensive performance metrics from backtest results.

    Parameters
    ----------
    trades          : list of trade dicts produced by BacktestEngine.run()
    equity_curve    : pd.Series of portfolio value over time
    initial_capital : float — starting capital in USDC

    Returns
    -------
    dict with all metrics (see inline comments for descriptions)
    """

    # ----------------------------------------------------------------
    # Guard: empty trades
    # ----------------------------------------------------------------
    if not trades:
        return _empty_metrics(initial_capital)

    pnls     = [t["pnl"] for t in trades]
    pnl_pcts = [t["pnl_pct"] for t in trades]

    wins  = [t for t in trades if t["pnl"] > 0]
    loses = [t for t in trades if t["pnl"] <= 0]

    total_trades   = len(trades)
    winning_trades = len(wins)
    losing_trades  = len(loses)
    win_rate       = (winning_trades / total_trades * 100) if total_trades else 0.0

    # ----------------------------------------------------------------
    # Return metrics
    # ----------------------------------------------------------------
    final_capital = equity_curve.iloc[-1] if len(equity_curve) else initial_capital
    total_return  = (final_capital - initial_capital) / initial_capital * 100

    # Annualized return (CAGR)
    if len(equity_curve) >= 2:
        start_dt = equity_curve.index[0]
        end_dt   = equity_curve.index[-1]
        years    = (end_dt - start_dt).total_seconds() / (365.25 * 86400)
        if years > 0 and initial_capital > 0:
            ann_return = ((final_capital / initial_capital) ** (1 / years) - 1) * 100
        else:
            ann_return = 0.0
    else:
        ann_return = 0.0

    # ----------------------------------------------------------------
    # Drawdown
    # ----------------------------------------------------------------
    equity_arr    = equity_curve.values.astype(float)
    peak          = np.maximum.accumulate(equity_arr)
    drawdown      = (equity_arr - peak) / np.where(peak > 0, peak, 1)
    max_drawdown  = float(drawdown.min() * 100)  # negative number → most negative

    # ----------------------------------------------------------------
    # Sharpe ratio (annualised, 0 risk-free rate)
    # Using equity-curve returns rather than trade returns for a more
    # accurate hourly sampling.
    # ----------------------------------------------------------------
    equity_returns = equity_curve.pct_change().dropna()
    if len(equity_returns) > 1:
        # Hourly data → 24*365 periods per year
        periods_per_year = 24 * 365
        mean_ret = equity_returns.mean()
        std_ret  = equity_returns.std(ddof=1)
        sharpe   = (mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0.0
    else:
        sharpe = 0.0

    # ----------------------------------------------------------------
    # Sortino ratio (downside deviation)
    # ----------------------------------------------------------------
    if len(equity_returns) > 1:
        neg_returns    = equity_returns[equity_returns < 0]
        downside_std   = neg_returns.std(ddof=1) if len(neg_returns) > 1 else 0.0
        sortino        = (equity_returns.mean() / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0.0
    else:
        sortino = 0.0

    # ----------------------------------------------------------------
    # Profit factor
    # ----------------------------------------------------------------
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss   = abs(sum(t["pnl"] for t in loses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # ----------------------------------------------------------------
    # Win / loss averages
    # ----------------------------------------------------------------
    avg_win      = (gross_profit / winning_trades) if winning_trades else 0.0
    avg_loss     = (gross_loss / losing_trades)    if losing_trades  else 0.0
    avg_win_pct  = float(np.mean([t["pnl_pct"] for t in wins]))  if wins  else 0.0
    avg_loss_pct = float(np.mean([t["pnl_pct"] for t in loses])) if loses else 0.0

    largest_win  = max(pnls) if pnls else 0.0
    largest_loss = min(pnls) if pnls else 0.0

    # ----------------------------------------------------------------
    # Duration
    # ----------------------------------------------------------------
    durations = [t.get("duration_hours", 0) for t in trades]
    avg_duration = float(np.mean(durations)) if durations else 0.0

    # ----------------------------------------------------------------
    # Fees & funding
    # ----------------------------------------------------------------
    total_fees    = sum(t.get("fees_paid", 0)    for t in trades)
    total_funding = sum(t.get("funding_paid", 0) for t in trades)

    # ----------------------------------------------------------------
    # Expectancy
    # ----------------------------------------------------------------
    expectancy = float(np.mean(pnls)) if pnls else 0.0

    return {
        "total_trades":      total_trades,
        "winning_trades":    winning_trades,
        "losing_trades":     losing_trades,
        "win_rate":          round(win_rate, 2),
        "total_return":      round(total_return, 2),
        "annualized_return": round(ann_return, 2),
        "max_drawdown":      round(max_drawdown, 2),
        "sharpe_ratio":      round(float(sharpe), 3),
        "sortino_ratio":     round(float(sortino), 3),
        "profit_factor":     round(profit_factor, 3),
        "avg_win":           round(avg_win, 2),
        "avg_loss":          round(avg_loss, 2),
        "avg_win_pct":       round(avg_win_pct, 4),
        "avg_loss_pct":      round(avg_loss_pct, 4),
        "largest_win":       round(largest_win, 2),
        "largest_loss":      round(largest_loss, 2),
        "avg_trade_duration": round(avg_duration, 1),
        "total_fees_paid":   round(total_fees, 2),
        "total_funding_paid": round(total_funding, 2),
        "expectancy":        round(expectancy, 2),
        "gross_profit":      round(gross_profit, 2),
        "gross_loss":        round(gross_loss, 2),
        "final_capital":     round(final_capital, 2),
        "initial_capital":   round(initial_capital, 2),
    }


def _empty_metrics(initial_capital: float) -> dict:
    return {
        "total_trades":       0,
        "winning_trades":     0,
        "losing_trades":      0,
        "win_rate":           0.0,
        "total_return":       0.0,
        "annualized_return":  0.0,
        "max_drawdown":       0.0,
        "sharpe_ratio":       0.0,
        "sortino_ratio":      0.0,
        "profit_factor":      0.0,
        "avg_win":            0.0,
        "avg_loss":           0.0,
        "avg_win_pct":        0.0,
        "avg_loss_pct":       0.0,
        "largest_win":        0.0,
        "largest_loss":       0.0,
        "avg_trade_duration": 0.0,
        "total_fees_paid":    0.0,
        "total_funding_paid": 0.0,
        "expectancy":         0.0,
        "gross_profit":       0.0,
        "gross_loss":         0.0,
        "final_capital":      initial_capital,
        "initial_capital":    initial_capital,
    }
