# ============================================================
# indicators/compute.py — Technical indicator computation
# Pure pandas/numpy — no external TA library required.
# ============================================================

import pandas as pd
import numpy as np


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=length - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=length - 1, adjust=False).mean()
    # Replace zero avg_loss with NaN to avoid division-by-zero
    avg_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # Where avg_loss was zero (no down moves) → RSI = 100
    rsi = rsi.fillna(100)
    return rsi


def _macd(series: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=length - 1, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int):
    """
    Wilder's Average Directional Index.
    Returns (adx, plus_di, minus_di) — all as pd.Series.
    ADX measures trend *strength* (not direction): >25 = trending, <20 = ranging.
    """
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    atr_s    = tr.ewm(com=length - 1, adjust=False).mean().replace(0, np.nan)
    plus_di  = 100 * plus_dm.ewm(com=length - 1, adjust=False).mean()  / atr_s
    minus_di = 100 * minus_dm.ewm(com=length - 1, adjust=False).mean() / atr_s

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx     = 100 * (plus_di - minus_di).abs() / di_sum
    adx    = dx.ewm(com=length - 1, adjust=False).mean()

    return adx, plus_di, minus_di


def _bbands(series: pd.Series, length: int, std: float):
    mid = series.rolling(length).mean()
    sigma = series.rolling(length).std(ddof=0)
    upper = mid + std * sigma
    lower = mid - std * sigma
    return upper, mid, lower


def add_indicators(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Add all technical indicators to an OHLCV DataFrame.

    Parameters
    ----------
    df     : DataFrame with columns open, high, low, close, volume
             and a DatetimeIndex (UTC).
    config : module or object exposing EMA_FAST, EMA_SLOW, RSI_PERIOD,
             MACD_FAST, MACD_SLOW, MACD_SIGNAL, ATR_PERIOD constants.

    Returns
    -------
    A copy of df with additional columns:
        ema_fast, ema_slow, rsi,
        macd, macd_signal, macd_hist,
        atr, adx, plus_di, minus_di,
        bb_upper, bb_mid, bb_lower,
        volume_sma,
        ema_bullish, ema_cross_up, ema_cross_down
    """
    df = df.copy()

    # ---- EMA -------------------------------------------------------
    df["ema_fast"] = _ema(df["close"], config.EMA_FAST)
    df["ema_slow"] = _ema(df["close"], config.EMA_SLOW)

    # ---- RSI -------------------------------------------------------
    df["rsi"] = _rsi(df["close"], config.RSI_PERIOD)
    # Momentum: direction RSI has moved over the last 5 bars, normalised to [-1, +1].
    # Rising RSI on a long → bullish; falling RSI on a short → bearish.
    # Avoids the level-trap where RSI stays "overbought" through entire strong trends.
    df["rsi_momentum"] = df["rsi"].diff(5).div(30).clip(-1.0, 1.0)

    # ---- MACD ------------------------------------------------------
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(
        df["close"], config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL
    )

    # ---- ATR -------------------------------------------------------
    df["atr"] = _atr(df["high"], df["low"], df["close"], config.ATR_PERIOD)

    # ---- ADX -------------------------------------------------------
    df["adx"], df["plus_di"], df["minus_di"] = _adx(
        df["high"], df["low"], df["close"], config.ADX_PERIOD
    )

    # ---- Bollinger Bands -------------------------------------------
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = _bbands(df["close"], 20, 2.0)

    # ---- BB bandwidth (self-calibrating via rolling percentile) ----
    _bw = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
    df["bb_bandwidth"]     = _bw
    df["bb_bandwidth_p90"] = _bw.rolling(config.BB_BW_LOOKBACK, min_periods=10).quantile(0.90)

    # ---- Volume SMA ------------------------------------------------
    df["volume_sma"] = df["volume"].rolling(20).mean()

    # ---- Derived EMA signal columns --------------------------------
    df["ema_bullish"] = df["ema_fast"] > df["ema_slow"]

    # ---- Normalised ATR (volatility regime) ------------------------
    # norm_atr     : ATR / close — comparable across price levels.
    # norm_atr_pct : rolling percentile rank — 0 = historically low vol, 1 = high.
    _norm_atr = df["atr"] / df["close"].replace(0, np.nan)
    df["norm_atr"]     = _norm_atr
    df["norm_atr_pct"] = _norm_atr.rolling(
        config.NORM_ATR_LOOKBACK, min_periods=10
    ).rank(pct=True)

    # ---- MA200 distance (extension brake) --------------------------
    # ma200_dist: (close - MA200) / MA200 — positive = above, negative = below.
    # NaN until MA200_PERIOD bars are available (no penalty applied upstream).
    df["ma200"] = df["close"].rolling(
        config.MA200_PERIOD, min_periods=config.MA200_PERIOD
    ).mean()
    df["ma200_dist"] = (df["close"] - df["ma200"]) / df["ma200"].replace(0, np.nan)

    return df
