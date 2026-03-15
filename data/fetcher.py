# ============================================================
# data/fetcher.py — Kraken OHLCV fetcher via ccxt
#
# Kraken is available in the US, has no geo-restrictions, and
# provides years of hourly ETH/USD history via its public REST API.
# ============================================================

import time
import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone


# ccxt timeframe strings (Kraken native)
_SUPPORTED = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}


def _make_exchange() -> ccxt.kraken:
    return ccxt.kraken({
        "enableRateLimit": True,
    })


def fetch_ohlcv(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Kraken.

    Parameters
    ----------
    symbol    : e.g. "ETH/USD" (Kraken spot — price matches perp closely)
    timeframe : "1h", "4h", "1d", etc.
    days      : how many calendar days back to fetch

    Returns
    -------
    pd.DataFrame with DatetimeIndex (UTC, datetime64[us]) and columns:
        open, high, low, close, volume  (all float64)
    """
    if timeframe not in _SUPPORTED:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}. "
                         f"Choose from {_SUPPORTED}")

    exchange  = _make_exchange()
    since_ms  = int(
        (datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )

    all_rows: list = []
    max_retries = 5

    while True:
        for attempt in range(1, max_retries + 1):
            try:
                batch = exchange.fetch_ohlcv(
                    symbol, timeframe, since=since_ms, limit=720
                )
                break
            except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Network error after {max_retries} retries: {exc}"
                    ) from exc
                time.sleep(2 ** attempt)
            except ccxt.ExchangeError as exc:
                raise RuntimeError(f"Exchange error: {exc}") from exc

        if not batch:
            break

        all_rows.extend(batch)
        last_ts = batch[-1][0]

        # Stop if Kraken returned fewer rows than the limit (end of history)
        if len(batch) < 720:
            break

        since_ms = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_rows:
        raise ValueError(
            f"No data returned for {symbol} {timeframe} (last {days} days)"
        )

    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        .astype("datetime64[us, UTC]")
    )
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    return df
