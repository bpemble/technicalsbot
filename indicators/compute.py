# ============================================================
# indicators/compute.py — Technical indicator computation
# Pure pandas/numpy — no external TA library required.
# ============================================================

import pandas as pd
import numpy as np

import config as _config


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

    # ---- Volume SMA ------------------------------------------------
    df["volume_sma"] = df["volume"].rolling(20).mean()

    # ---- Derived EMA signal columns --------------------------------
    df["ema_bullish"] = df["ema_fast"] > df["ema_slow"]

    prev_fast = df["ema_fast"].shift(1)
    prev_slow = df["ema_slow"].shift(1)
    df["ema_cross_up"]   = (prev_fast <= prev_slow) & (df["ema_fast"] > df["ema_slow"])
    df["ema_cross_down"] = (prev_fast >= prev_slow) & (df["ema_fast"] < df["ema_slow"])

    return df
