# ============================================================
# strategy/multi_tf.py — Multi-timeframe EMA+RSI+MACD+ATR strategy
# ============================================================

import pandas as pd
import numpy as np


class MultiTFStrategy:
    """
    Multi-timeframe trend-following + momentum strategy for ETH/USD perps.

    Trend bias is read from the daily chart; entries are timed on the 4h chart.

    LONG entry (all must be true):
      - 1d: EMA_FAST > EMA_SLOW          (uptrend confirmed)
      - 1d: MACD histogram > 0           (bullish momentum)
      - 4h: RSI < 60 AND rising          (momentum building, not overbought)
      - 4h: close > EMA_FAST             (price above fast MA)

    SHORT entry (all must be true):
      - 1d: EMA_FAST < EMA_SLOW          (downtrend confirmed)
      - 1d: MACD histogram < 0           (bearish momentum)
      - 4h: RSI > 40 AND falling         (momentum weakening, not oversold)
      - 4h: close < EMA_FAST             (price below fast MA)

    Exit rules:
      - Stop loss  : entry_price ± ATR * ATR_STOP_MULTIPLIER
      - Take profit: entry_price ± ATR * ATR_TP_MULTIPLIER
      - Opposite signal reverses the position
    """

    def __init__(self, config):
        self.config = config

    def generate_signals(
        self,
        df_primary: pd.DataFrame,   # daily bars with indicators
        df_entry: pd.DataFrame,     # 4h bars with indicators
    ) -> pd.DataFrame:
        """
        Align daily bias to 4h bars and generate entry/exit signals.

        No lookahead bias: a daily candle that closes at T is first usable
        on bars that open at T or later. We shift the daily index forward
        by 1 day before the merge so bar[i]'s signal only applies after
        its close.

        Returns a DataFrame indexed by 4h timestamps with columns:
            signal        : int  (1=long, -1=short, 0=none)
            stop_loss     : float
            take_profit   : float
            position_size : float (ETH, indicative — engine recomputes live)
            atr_at_signal : float
            close_at_signal: float
        """
        cfg = self.config

        # ---- 1. Build daily bias series (shift forward 1 day) -------
        bias = df_primary[["ema_bullish", "macd_hist", "atr"]].copy()
        bias.index = bias.index + pd.Timedelta(days=1)
        bias = bias.rename(columns={
            "ema_bullish": "bias_bull",
            "macd_hist":   "bias_macd_hist",
            "atr":         "bias_atr",
        })

        # ---- 2. Merge daily bias onto 4h bars (no lookahead) --------
        entry_r = df_entry.reset_index()
        bias_r  = bias.reset_index()

        # Normalise datetime precision so merge_asof doesn't complain
        entry_r["timestamp"] = entry_r["timestamp"].astype("datetime64[us, UTC]")
        bias_r["timestamp"]  = bias_r["timestamp"].astype("datetime64[us, UTC]")

        merged = pd.merge_asof(
            entry_r.sort_values("timestamp"),
            bias_r.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        merged.set_index("timestamp", inplace=True)

        # ---- 3. RSI threshold + direction conditions ----------------
        rsi      = merged["rsi"]
        rsi_prev = rsi.shift(1)
        rsi_rising  = rsi > rsi_prev   # RSI trending up
        rsi_falling = rsi < rsi_prev   # RSI trending down

        # ---- 4. Long / Short conditions -----------------------------
        long_cond = (
            (merged["bias_bull"] == True)           # daily uptrend
            & (merged["bias_macd_hist"] > 0)         # daily bullish MACD
            & (rsi < 60)                             # not overbought
            & rsi_rising                             # momentum building
            & (merged["close"] > merged["ema_fast"]) # 4h price above EMA
        )

        short_cond = (
            (merged["bias_bull"] == False)           # daily downtrend
            & (merged["bias_macd_hist"] < 0)         # daily bearish MACD
            & (rsi > 40)                             # not oversold
            & rsi_falling                            # momentum weakening
            & (merged["close"] < merged["ema_fast"]) # 4h price below EMA
        )

        # Avoid entering the same direction on consecutive bars
        # (only trigger on the first bar where conditions flip on)
        long_cond  = long_cond  & ~long_cond.shift(1).fillna(False)
        short_cond = short_cond & ~short_cond.shift(1).fillna(False)

        # ---- 5. Build output DataFrame ------------------------------
        atr   = merged["atr"]
        close = merged["close"]

        out = pd.DataFrame(index=merged.index)
        out["signal"]         = 0
        out["stop_loss"]      = np.nan
        out["take_profit"]    = np.nan
        out["position_size"]  = np.nan
        out["atr_at_signal"]  = atr
        out["close_at_signal"] = close

        out.loc[long_cond,  "signal"] =  1
        out.loc[short_cond, "signal"] = -1

        long_mask  = out["signal"] ==  1
        short_mask = out["signal"] == -1

        out.loc[long_mask,  "stop_loss"]   = close[long_mask]  - atr[long_mask]  * cfg.ATR_STOP_MULTIPLIER
        out.loc[long_mask,  "take_profit"] = close[long_mask]  + atr[long_mask]  * cfg.ATR_TP_MULTIPLIER
        out.loc[short_mask, "stop_loss"]   = close[short_mask] + atr[short_mask] * cfg.ATR_STOP_MULTIPLIER
        out.loc[short_mask, "take_profit"] = close[short_mask] - atr[short_mask] * cfg.ATR_TP_MULTIPLIER

        # Indicative position sizes (engine overrides with live capital)
        stop_dist   = atr * cfg.ATR_STOP_MULTIPLIER
        dollar_risk = cfg.INITIAL_CAPITAL * cfg.RISK_PER_TRADE
        raw_size    = dollar_risk / stop_dist.replace(0, np.nan)
        max_size    = (cfg.INITIAL_CAPITAL * cfg.LEVERAGE) / close.replace(0, np.nan)
        sig_mask    = long_mask | short_mask
        out.loc[sig_mask, "position_size"] = raw_size[sig_mask].clip(upper=max_size[sig_mask])

        return out
