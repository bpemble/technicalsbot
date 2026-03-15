# ============================================================
# backtest/engine.py — Event-driven backtesting loop
# ============================================================

import pandas as pd
import numpy as np
from typing import Optional


class BacktestEngine:
    """
    Iterates through 1h candles and simulates order execution.

    Cost model
    ----------
    - Entry fee   : EXCHANGE_FEE (taker) applied to notional at entry
    - Exit fee    : EXCHANGE_FEE (taker) applied to notional at exit
    - Slippage    : SLIPPAGE fraction added to cost on entries (widens spread)
    - Funding     : FUNDING_RATE_DAILY / 3  per 8h interval, charged every
                    8 candles (once per 8h funding period)

    Intracandle stop / TP logic
    ---------------------------
    For each open position on each new candle:
      1. Use the candle's high/low to check if stop or TP was triggered.
      2. If both could be hit in the same candle → assume stop (conservative).
      3. For longs  : stop triggered if low  <= stop_loss
                      TP   triggered if high >= take_profit
      4. For shorts : stop triggered if high >= stop_loss
                      TP   triggered if low  <= take_profit

    Entry fill
    ----------
    Signals generated on candle[i] are filled at the *open* of candle[i+1]
    (next-bar-open execution, zero lookahead).
    """

    def __init__(self, config):
        self.config = config

    # ------------------------------------------------------------------
    def run(
        self,
        signals_df: pd.DataFrame,
        price_df: pd.DataFrame,
    ) -> dict:
        """
        Run the backtest.

        Parameters
        ----------
        signals_df : Output of MultiTFStrategy.generate_signals().
                     Must contain: signal, stop_loss, take_profit,
                                   atr_at_signal, close_at_signal
        price_df   : 1h OHLCV DataFrame (same index as signals_df ideally,
                     but can be the raw fetched frame).

        Returns
        -------
        dict with keys:
            trades        : list[dict]
            equity_curve  : pd.Series  (capital indexed by candle timestamp)
            final_capital : float
        """
        cfg = self.config

        # Align on common index
        common_idx = signals_df.index.intersection(price_df.index)
        signals_df = signals_df.loc[common_idx]
        price_df   = price_df.loc[common_idx]

        capital    = cfg.INITIAL_CAPITAL
        equity_curve: dict = {}
        trades: list = []

        # Open position state
        in_position       = False
        direction: Optional[str] = None   # 'long' or 'short'
        entry_price       = 0.0
        entry_time        = None
        position_size     = 0.0           # ETH amount
        stop_loss         = 0.0
        take_profit       = 0.0
        fees_paid_entry   = 0.0
        funding_accrued   = 0.0
        candles_in_trade  = 0

        timestamps = list(signals_df.index)
        n          = len(timestamps)

        for i, ts in enumerate(timestamps):
            price_row  = price_df.loc[ts]
            sig_row    = signals_df.loc[ts]

            candle_open  = float(price_row["open"])
            candle_high  = float(price_row["high"])
            candle_low   = float(price_row["low"])
            candle_close = float(price_row["close"])

            # ---- 1. Manage open position on this candle ---------------
            if in_position:
                candles_in_trade += 1
                notional_now = position_size * candle_open

                # Funding: charged every 8 candles (≈ every 8h)
                if candles_in_trade % 8 == 0:
                    funding_charge = notional_now * (cfg.FUNDING_RATE_DAILY / 3)
                    funding_accrued += funding_charge

                # Check stop/TP intracandle
                stop_hit = False
                tp_hit   = False

                if direction == "long":
                    stop_hit = candle_low  <= stop_loss
                    tp_hit   = candle_high >= take_profit
                else:  # short
                    stop_hit = candle_high >= stop_loss
                    tp_hit   = candle_low  <= take_profit

                # If both can be hit → conservative: assume stop first
                if stop_hit and tp_hit:
                    tp_hit = False

                exit_reason = None
                exit_price  = None

                if stop_hit:
                    exit_price  = stop_loss
                    exit_reason = "stop_loss"
                elif tp_hit:
                    exit_price  = take_profit
                    exit_reason = "take_profit"
                else:
                    # Check for signal reversal (signal on THIS candle means
                    # we entered on its open, so we close at this candle's
                    # open if an opposing signal was generated on the
                    # PREVIOUS candle).
                    # We handle this below after the entry-signal section.
                    pass

                if exit_reason is not None:
                    trade = self._close_trade(
                        cfg, direction, entry_price, exit_price,
                        position_size, entry_time, ts,
                        fees_paid_entry, funding_accrued, exit_reason, capital,
                    )
                    capital         += trade["pnl"]
                    trades.append(trade)
                    in_position      = False
                    direction        = None
                    candles_in_trade = 0
                    funding_accrued  = 0.0

            # ---- 2. Check for new signal (fills on NEXT candle open) --
            #    Signal on candle[i-1] → entry on candle[i].open
            #    We look at the PREVIOUS candle's signal.
            if i > 0:
                prev_ts     = timestamps[i - 1]
                prev_signal = int(signals_df.loc[prev_ts, "signal"])
                prev_sl     = float(signals_df.loc[prev_ts, "stop_loss"]) if not np.isnan(signals_df.loc[prev_ts, "stop_loss"]) else None
                prev_tp     = float(signals_df.loc[prev_ts, "take_profit"]) if not np.isnan(signals_df.loc[prev_ts, "take_profit"]) else None
                prev_atr    = float(signals_df.loc[prev_ts, "atr_at_signal"]) if "atr_at_signal" in signals_df.columns and not np.isnan(signals_df.loc[prev_ts, "atr_at_signal"]) else None

                if in_position and prev_signal != 0:
                    # Opposite signal reverses position
                    if (direction == "long" and prev_signal == -1) or \
                       (direction == "short" and prev_signal == 1):
                        exit_price = candle_open
                        trade = self._close_trade(
                            cfg, direction, entry_price, exit_price,
                            position_size, entry_time, ts,
                            fees_paid_entry, funding_accrued, "signal_reverse", capital,
                        )
                        capital         += trade["pnl"]
                        trades.append(trade)
                        in_position      = False
                        direction        = None
                        candles_in_trade = 0
                        funding_accrued  = 0.0

                if not in_position and prev_signal != 0 and prev_sl is not None:
                    # Enter new position at this candle's open
                    fill_price = candle_open * (
                        1 + cfg.SLIPPAGE if prev_signal == 1
                        else 1 - cfg.SLIPPAGE
                    )

                    # Recompute stop and TP relative to actual fill price
                    if prev_atr is not None:
                        atr_val = prev_atr
                    else:
                        # fallback: use stop distance from signal
                        if prev_signal == 1:
                            atr_val = (float(signals_df.loc[prev_ts, "close_at_signal"]) - prev_sl) / cfg.ATR_STOP_MULTIPLIER
                        else:
                            atr_val = (prev_sl - float(signals_df.loc[prev_ts, "close_at_signal"])) / cfg.ATR_STOP_MULTIPLIER

                    stop_dist = atr_val * cfg.ATR_STOP_MULTIPLIER
                    tp_dist   = atr_val * cfg.ATR_TP_MULTIPLIER

                    if prev_signal == 1:
                        stop_loss   = fill_price - stop_dist
                        take_profit = fill_price + tp_dist
                    else:
                        stop_loss   = fill_price + stop_dist
                        take_profit = fill_price - tp_dist

                    # Position size: risk a fixed % of current capital
                    dollar_risk   = capital * cfg.RISK_PER_TRADE
                    max_notional  = capital * cfg.LEVERAGE
                    raw_size      = dollar_risk / stop_dist if stop_dist > 0 else 0.0
                    max_size      = max_notional / fill_price if fill_price > 0 else 0.0
                    position_size = min(raw_size, max_size)

                    if position_size <= 0:
                        # Skip degenerate case
                        equity_curve[ts] = capital
                        continue

                    notional_entry = position_size * fill_price
                    fees_paid_entry = notional_entry * cfg.EXCHANGE_FEE

                    # Deduct entry fees immediately
                    capital -= fees_paid_entry

                    in_position      = True
                    direction        = "long" if prev_signal == 1 else "short"
                    entry_price      = fill_price
                    entry_time       = ts
                    candles_in_trade = 0
                    funding_accrued  = 0.0

            # ---- 3. Record equity at this timestamp ------------------
            if in_position:
                # Mark-to-market unrealised P&L
                if direction == "long":
                    unrealised = (candle_close - entry_price) * position_size
                else:
                    unrealised = (entry_price - candle_close) * position_size
                equity_curve[ts] = capital + unrealised - funding_accrued
            else:
                equity_curve[ts] = capital

        # ---- 4. Close any still-open position at last bar's close ----
        if in_position:
            last_ts    = timestamps[-1]
            last_close = float(price_df.loc[last_ts, "close"])
            trade = self._close_trade(
                cfg, direction, entry_price, last_close,
                position_size, entry_time, last_ts,
                fees_paid_entry, funding_accrued, "end_of_data", capital,
            )
            capital += trade["pnl"]
            trades.append(trade)
            equity_curve[last_ts] = capital

        equity_series = pd.Series(equity_curve)
        equity_series.index = pd.to_datetime(equity_series.index, utc=True)
        equity_series.sort_index(inplace=True)

        return {
            "trades":        trades,
            "equity_curve":  equity_series,
            "final_capital": capital,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _close_trade(
        cfg,
        direction: str,
        entry_price: float,
        exit_price: float,
        size: float,
        entry_time,
        exit_time,
        fees_entry: float,
        funding_paid: float,
        exit_reason: str,
        capital_before_exit: float,
    ) -> dict:
        """Compute trade P&L and return trade dict."""
        notional_exit = size * exit_price
        fees_exit     = notional_exit * cfg.EXCHANGE_FEE

        if direction == "long":
            gross_pnl = (exit_price - entry_price) * size
        else:
            gross_pnl = (entry_price - exit_price) * size

        # Net P&L: gross minus exit fee (entry fee already deducted from capital)
        net_pnl = gross_pnl - fees_exit - funding_paid

        # pnl_pct relative to capital before exit
        pnl_pct = (net_pnl / capital_before_exit) * 100 if capital_before_exit > 0 else 0.0

        # Duration
        if entry_time is not None and exit_time is not None:
            try:
                duration_hours = (
                    pd.Timestamp(exit_time) - pd.Timestamp(entry_time)
                ).total_seconds() / 3600
            except Exception:
                duration_hours = 0.0
        else:
            duration_hours = 0.0

        return {
            "entry_time":    entry_time,
            "exit_time":     exit_time,
            "direction":     direction,
            "entry_price":   entry_price,
            "exit_price":    exit_price,
            "size":          size,
            "pnl":           net_pnl,
            "pnl_pct":       pnl_pct,
            "exit_reason":   exit_reason,
            "fees_paid":     fees_entry + fees_exit,
            "funding_paid":  funding_paid,
            "duration_hours": duration_hours,
        }
