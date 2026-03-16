# ============================================================
# data/fetcher.py — Kraken OHLCV fetcher via ccxt
#
# Kraken is available in the US, has no geo-restrictions, and
# provides years of hourly ETH/USD history via its public REST API.
# ============================================================

import sys
import time
import ccxt
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone


# ccxt timeframe strings (Kraken native)
_SUPPORTED = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}

MAX_PAGES = 50


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
    page = 0

    while True:
        if page >= MAX_PAGES:
            print(f"Warning: hit MAX_PAGES limit for {symbol} {timeframe}", file=sys.stderr)
            break

        for attempt in range(1, max_retries + 1):
            try:
                batch = exchange.fetch_ohlcv(
                    symbol, timeframe, since=since_ms, limit=720,
                    params={"timeout": 10000}
                )
                break
            except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.RateLimitExceeded) as exc:
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
        page += 1
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


def fetch_latest_prices() -> dict[str, float]:
    """
    Fetch current mid prices from Hyperliquid for all perps.

    Single lightweight API call — used by the fast stop-check loop between
    regime scans. Returns {hl_asset_name: price}, e.g. {"ETH": 2450.5}.
    Returns empty dict on any failure so the caller degrades gracefully.
    """
    try:
        resp = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={"type": "allMids"},
            timeout=5,
        )
        resp.raise_for_status()
        raw = resp.json()  # {"ETH": "2450.5", "BTC": "65000.1", ...}
        return {name: float(price) for name, price in raw.items() if price}
    except Exception as exc:
        print(f"Warning: price fetch failed: {exc}", file=sys.stderr)
        return {}


def fetch_funding_rates() -> dict[str, float]:
    """
    Fetch current predicted hourly funding rates from Hyperliquid for all perps.

    Hyperliquid settles funding every hour. The returned rate is the *predicted*
    rate for the current hour (can change until settlement).

    Returns
    -------
    dict mapping Hyperliquid asset name → hourly funding rate as a decimal.
    e.g. {"ETH": 0.0001, "BTC": -0.00005, ...}
    Returns an empty dict on any failure so the caller can degrade gracefully.
    """
    try:
        resp = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={"type": "metaAndAssetCtxs"},
            timeout=8,
        )
        resp.raise_for_status()
        meta, ctxs = resp.json()
        return {
            asset["name"]: float(ctx.get("funding", 0.0))
            for asset, ctx in zip(meta["universe"], ctxs)
        }
    except Exception as exc:
        print(f"Warning: funding rate fetch failed: {exc}", file=sys.stderr)
        return {}
