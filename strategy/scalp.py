"""
Scalp strategy: 1h trend bias + 15m entry timing.

LONG entry (all must be true):
  - 1h: EMA21 > EMA55 (uptrend)
  - 1h: MACD histogram > 0
  - 15m: RSI > 40 AND rising (momentum building)
  - 15m: close > EMA21

SHORT entry (all must be true):
  - 1h: EMA21 < EMA55 (downtrend)
  - 1h: MACD histogram < 0
  - 15m: RSI < 60 AND falling (momentum weakening)
  - 15m: close < EMA21

Exits:
  - Stop: entry ± ATR * SCALP_ATR_STOP
  - TP:   entry ± ATR * SCALP_ATR_TP
  - Opposite signal reverses
"""
import pandas as pd
import numpy as np


class ScalpStrategy:
    def __init__(self, config):
        self.config = config

    def generate_signals(
        self,
        df_trend: pd.DataFrame,   # 1h bars with indicators
        df_entry: pd.DataFrame,   # 15m bars with indicators
    ) -> pd.DataFrame:
        """
        Align 1h bias onto 15m bars (no lookahead).
        Shift 1h index forward by 1h so bar[i] only influences bars after its close.
        Returns DataFrame indexed by 15m timestamps with columns:
          signal, stop_loss, take_profit, position_size, atr_at_signal, close_at_signal
        """
        cfg = self.config

        # Build 1h bias, shifted forward 1h (no lookahead)
        bias = df_trend[["ema_bullish", "macd_hist", "atr"]].copy()
        bias.index = bias.index + pd.Timedelta(hours=1)
        bias = bias.rename(columns={
            "ema_bullish": "bias_bull",
            "macd_hist":   "bias_macd_hist",
            "atr":         "bias_atr",
        })

        # Merge onto 15m bars
        entry_r = df_entry.reset_index()
        bias_r  = bias.reset_index()
        entry_r["timestamp"] = entry_r["timestamp"].astype("datetime64[us, UTC]")
        bias_r["timestamp"]  = bias_r["timestamp"].astype("datetime64[us, UTC]")

        merged = pd.merge_asof(
            entry_r.sort_values("timestamp"),
            bias_r.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        ).set_index("timestamp")

        rsi      = merged["rsi"]
        rsi_prev = rsi.shift(1)
        rsi_rising  = rsi > rsi_prev
        rsi_falling = rsi < rsi_prev

        long_cond = (
            (merged["bias_bull"] == True)
            & (merged["bias_macd_hist"] > 0)
            & (rsi > 40)
            & rsi_rising
            & (merged["close"] > merged["ema_fast"])
        )
        short_cond = (
            (merged["bias_bull"] == False)
            & (merged["bias_macd_hist"] < 0)
            & (rsi < 60)
            & rsi_falling
            & (merged["close"] < merged["ema_fast"])
        )

        # Only trigger on first bar of a run
        long_cond  = long_cond  & ~long_cond.shift(1).fillna(False)
        short_cond = short_cond & ~short_cond.shift(1).fillna(False)

        atr   = merged["atr"]
        close = merged["close"]

        out = pd.DataFrame(index=merged.index)
        out["signal"]          = 0
        out["stop_loss"]       = np.nan
        out["take_profit"]     = np.nan
        out["position_size"]   = np.nan
        out["atr_at_signal"]   = atr
        out["close_at_signal"] = close

        out.loc[long_cond,  "signal"] =  1
        out.loc[short_cond, "signal"] = -1

        long_mask  = out["signal"] ==  1
        short_mask = out["signal"] == -1

        out.loc[long_mask,  "stop_loss"]   = close[long_mask]  - atr[long_mask]  * cfg.SCALP_ATR_STOP
        out.loc[long_mask,  "take_profit"] = close[long_mask]  + atr[long_mask]  * cfg.SCALP_ATR_TP
        out.loc[short_mask, "stop_loss"]   = close[short_mask] + atr[short_mask] * cfg.SCALP_ATR_STOP
        out.loc[short_mask, "take_profit"] = close[short_mask] - atr[short_mask] * cfg.SCALP_ATR_TP

        stop_dist   = atr * cfg.SCALP_ATR_STOP
        dollar_risk = cfg.PAPER_CAPITAL * cfg.SCALP_RISK_PER_TRADE
        raw_size    = dollar_risk / stop_dist.replace(0, np.nan)
        max_size    = (cfg.PAPER_CAPITAL * cfg.LEVERAGE) / close.replace(0, np.nan)
        sig_mask    = long_mask | short_mask
        out.loc[sig_mask, "position_size"] = raw_size[sig_mask].clip(upper=max_size[sig_mask])

        return out
