# ============================================================
# strategy/regime.py — Multi-timeframe market regime engine
#
# Produces a composite conviction score from -100 to +100:
#   +100 = maximally bullish (strong long)
#   -100 = maximally bearish (strong short)
#      0 = no edge (flat / deadzone)
#
# Score is built from four indicator components per timeframe,
# then combined with timeframe weights (daily has most weight).
#
# Timeframe weights:
#   Daily : 40%  (slow, high-conviction trend)
#   4h    : 30%  (medium-term momentum)
#   1h    : 20%  (short-term momentum)
#   15m   : 10%  (immediate pressure)
# ============================================================

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RegimeSnapshot:
    """Full breakdown of the regime score at a point in time."""
    score: float                          # composite -100 to +100
    direction: str                        # 'long', 'short', or 'flat'
    conviction: float                     # abs(score) / 100, 0-1
    component_scores: dict = field(default_factory=dict)  # per-TF breakdown
    indicator_scores: dict = field(default_factory=dict)  # per-indicator breakdown
    latest_price: float = 0.0


# Minimum absolute score to hold any position (deadzone avoids thrashing)
MIN_CONVICTION_SCORE = 15.0


class RegimeEngine:
    """
    Computes a market regime score from multiple timeframes and indicators.

    Indicator components (each normalised to [-1, +1]):

    1. RSI component
       Oversold (RSI < 30) → strongly bullish (+1)
       Overbought (RSI > 70) → strongly bearish (-1)
       Neutral (RSI = 50) → 0
       Formula: clip((50 - RSI) / 35, -1, 1)

    2. EMA trend component
       Price above both EMAs AND fast > slow  → +1  (strong uptrend)
       Price below both EMAs AND fast < slow  → -1  (strong downtrend)
       Mixed                                  → ±0.5

    3. MACD momentum component
       MACD histogram normalised by a rolling ATR proxy.
       Positive & growing → bullish; negative & shrinking → bearish.

    4. Bollinger Band mean-reversion component
       Price near/below lower band → bullish (oversold stretch)
       Price near/above upper band → bearish (overbought stretch)
       Formula: clip((bb_mid - close) / (bb_upper - bb_mid), -1, 1)
    """

    def __init__(self, config):
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        df_daily: pd.DataFrame,
        df_4h:    pd.DataFrame,
        df_1h:    pd.DataFrame,
        df_15m:   pd.DataFrame,
    ) -> RegimeSnapshot:
        """
        Compute the current regime score from the latest bar of each timeframe.
        All DataFrames must have indicators already applied (add_indicators).
        """
        weights = {
            "daily": 0.40,
            "4h":    0.30,
            "1h":    0.20,
            "15m":   0.10,
        }
        frames = {
            "daily": df_daily,
            "4h":    df_4h,
            "1h":    df_1h,
            "15m":   df_15m,
        }

        component_scores = {}
        indicator_scores = {}

        for tf_name, df in frames.items():
            if df is None or df.empty:
                component_scores[tf_name] = 0.0
                continue
            row = df.iloc[-1]
            scores = self._score_row(row, tf_name)
            indicator_scores[tf_name] = scores
            component_scores[tf_name] = float(np.mean(list(scores.values())))

        # Weighted composite
        raw = sum(component_scores[tf] * weights[tf] for tf in weights)
        score = float(np.clip(raw * 100, -100, 100))

        if score > MIN_CONVICTION_SCORE:
            direction = "long"
        elif score < -MIN_CONVICTION_SCORE:
            direction = "short"
        else:
            direction = "flat"

        conviction = abs(score) / 100.0

        latest_price = float(df_15m["close"].iloc[-1]) if not df_15m.empty else 0.0

        return RegimeSnapshot(
            score=score,
            direction=direction,
            conviction=conviction,
            component_scores=component_scores,
            indicator_scores=indicator_scores,
            latest_price=latest_price,
        )

    # ------------------------------------------------------------------
    # Per-row indicator scoring
    # ------------------------------------------------------------------

    def _score_row(self, row: pd.Series, tf_name: str) -> dict:
        scores = {}

        # ---- 1. RSI -------------------------------------------------
        rsi = self._safe(row, "rsi")
        if rsi is not None:
            scores["rsi"] = float(np.clip((50.0 - rsi) / 35.0, -1.0, 1.0))

        # ---- 2. EMA trend -------------------------------------------
        close     = self._safe(row, "close")
        ema_fast  = self._safe(row, "ema_fast")
        ema_slow  = self._safe(row, "ema_slow")
        if close is not None and ema_fast is not None and ema_slow is not None:
            above_fast = close > ema_fast
            above_slow = close > ema_slow
            fast_above_slow = ema_fast > ema_slow
            if above_fast and above_slow and fast_above_slow:
                scores["ema_trend"] = 1.0
            elif not above_fast and not above_slow and not fast_above_slow:
                scores["ema_trend"] = -1.0
            elif above_fast and fast_above_slow:
                scores["ema_trend"] = 0.5
            elif not above_fast and not fast_above_slow:
                scores["ema_trend"] = -0.5
            else:
                scores["ema_trend"] = 0.0

        # ---- 3. MACD momentum ---------------------------------------
        macd_hist = self._safe(row, "macd_hist")
        atr       = self._safe(row, "atr")
        if macd_hist is not None and atr is not None and atr > 0:
            # Normalise histogram by ATR so it's comparable across price levels
            scores["macd"] = float(np.clip(macd_hist / atr, -1.0, 1.0))

        # ---- 4. Bollinger Band mean-reversion -----------------------
        bb_upper = self._safe(row, "bb_upper")
        bb_mid   = self._safe(row, "bb_mid")
        bb_lower = self._safe(row, "bb_lower")
        if close is not None and bb_upper is not None and bb_mid is not None and bb_lower is not None:
            band_half = bb_upper - bb_mid
            if band_half > 0:
                scores["bb_reversion"] = float(np.clip((bb_mid - close) / band_half, -1.0, 1.0))

        return scores

    @staticmethod
    def _safe(row: pd.Series, col: str) -> Optional[float]:
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)
