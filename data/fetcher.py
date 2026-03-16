# ============================================================
# data/fetcher.py — OHLCV fetcher + in-memory DataCache
#
# Data source: Hyperliquid candleSnapshot API (colocated — sub-ms latency).
# Kraken is kept as a fallback for assets with thin HL history.
#
# DataCache avoids re-fetching slow timeframes on every scan:
#   First call  → full history fetch
#   Later calls → incremental fetch only when a new candle has closed
# ============================================================

import sys
import time
import logging
import requests
import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# ── Shared HTTP session — keeps TCP connection alive to HL ───────────────────
# Colocated on HL infrastructure: round trips are <5 ms.
# A persistent session eliminates per-request TLS/TCP setup overhead.
_hl_session = requests.Session()
_hl_session.headers.update({"Content-Type": "application/json"})

HL_API = "https://api.hyperliquid.xyz/info"
HL_TIMEOUT = 5   # tight — colocated, should never need more

# ── Interval helpers ─────────────────────────────────────────────────────────
_SUPPORTED_INTERVALS = {"15m", "1h", "4h", "1d"}

_INTERVAL_SECONDS = {
    "15m":   15 * 60,
    "1h":    60 * 60,
    "4h":  4 * 60 * 60,
    "1d": 24 * 60 * 60,
}

_INTERVAL_MS = {k: v * 1000 for k, v in _INTERVAL_SECONDS.items()}


# ── Hyperliquid candle fetcher ────────────────────────────────────────────────

def fetch_hl_candles(coin: str, interval: str, since_ms: int) -> pd.DataFrame:
    """
    Fetch OHLCV candles from the Hyperliquid candleSnapshot endpoint.

    Since this server is colocated with HL, this is the fastest possible
    data source — used for all normal operation.

    Parameters
    ----------
    coin      : HL asset name, e.g. "ETH", "SOL"
    interval  : "15m", "1h", "4h", "1d"
    since_ms  : fetch candles with open time >= this epoch-ms timestamp
    """
    if interval not in _INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval!r}")

    end_ms       = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    interval_ms  = _INTERVAL_MS[interval]
    all_candles: list = []
    batch_start  = since_ms

    while batch_start < end_ms:
        batch_end = min(batch_start + interval_ms * 5_000, end_ms)
        try:
            resp = _hl_session.post(
                HL_API,
                json={"type": "candleSnapshot", "req": {
                    "coin":      coin,
                    "interval":  interval,
                    "startTime": batch_start,
                    "endTime":   batch_end,
                }},
                timeout=HL_TIMEOUT,
            )
            resp.raise_for_status()
            candles = resp.json()
        except Exception as exc:
            raise RuntimeError(
                f"HL candle fetch failed for {coin} {interval}: {exc}"
            ) from exc

        if not candles:
            break
        all_candles.extend(candles)
        if len(candles) < 2:
            break
        batch_start = int(candles[-1]["T"]) + 1

    if not all_candles:
        raise ValueError(f"No HL candles returned for {coin} {interval}")

    df = pd.DataFrame([{
        "timestamp": int(c["t"]),
        "open":      float(c["o"]),
        "high":      float(c["h"]),
        "low":       float(c["l"]),
        "close":     float(c["c"]),
        "volume":    float(c["v"]),
    } for c in all_candles])

    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        .astype("datetime64[us, UTC]")
    )
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index().astype(float)
    return df


# ── Kraken fallback (used only when HL history is too short) ─────────────────

_KRAKEN_SUPPORTED = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
_MAX_KRAKEN_PAGES = 50

def _fetch_kraken_candles(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch from Kraken via ccxt — fallback when HL history is insufficient."""
    exchange = ccxt.kraken({"enableRateLimit": True})
    since_ms = int(
        (datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )
    all_rows: list = []
    page = 0

    while True:
        if page >= _MAX_KRAKEN_PAGES:
            break
        for attempt in range(1, 6):
            try:
                batch = exchange.fetch_ohlcv(
                    symbol, interval, since=since_ms, limit=720,
                    params={"timeout": 10000},
                )
                break
            except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.RateLimitExceeded) as exc:
                if attempt == 5:
                    raise RuntimeError(f"Kraken network error: {exc}") from exc
                time.sleep(2 ** attempt)
            except ccxt.ExchangeError as exc:
                raise RuntimeError(f"Kraken exchange error: {exc}") from exc

        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < 720:
            break
        since_ms = batch[-1][0] + 1
        page += 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_rows:
        raise ValueError(f"No Kraken data for {symbol} {interval}")

    df = pd.DataFrame(
        all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        .astype("datetime64[us, UTC]")
    )
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# ── DataCache ─────────────────────────────────────────────────────────────────

class DataCache:
    """
    In-memory OHLCV cache keyed by (hl_coin, interval).

    First access  — full history fetch from Hyperliquid (falls back to Kraken
                    if HL history is too short for the requested lookback).
    Later accesses — incremental fetch only when a new candle has closed,
                    appended to the existing DataFrame.

    This makes a full 14-asset × 4-timeframe scan take ~10-15 s on the first
    tick, then a few seconds on subsequent ticks (only fetching new 15m bars).
    """

    # How many seconds after a new candle *should* have closed before we fetch.
    # A small buffer ensures the candle is finalised and visible via the API.
    _FETCH_BUFFER_S = 30

    def __init__(self):
        self._frames:     dict[tuple, pd.DataFrame] = {}
        self._last_fetch: dict[tuple, datetime]     = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, hl_coin: str, interval: str, lookback_days: int) -> pd.DataFrame:
        """
        Return an up-to-date OHLCV DataFrame.

        Fetches from Hyperliquid if:
          - no data is cached yet, or
          - a new candle has closed since the last bar in the cache.
        """
        key = (hl_coin, interval)

        if key not in self._frames:
            self._full_fetch(hl_coin, interval, lookback_days, key)
        elif self._new_candle_available(key, interval):
            self._incremental_fetch(hl_coin, interval, key)

        return self._frames[key]

    def warm(self, assets: list, intervals_days: dict):
        """
        Pre-fetch all assets × timeframes in parallel at startup.

        assets         : list of asset dicts from config.ASSETS
        intervals_days : {"15m": 3, "1h": 7, "4h": 120, "1d": 730}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        jobs = [
            (asset["hl"], interval, days)
            for asset in assets
            for interval, days in intervals_days.items()
        ]

        console_lines = []
        with ThreadPoolExecutor(max_workers=12) as ex:
            futures = {ex.submit(self.get, coin, iv, days): (coin, iv)
                       for coin, iv, days in jobs}
            for f in as_completed(futures):
                coin, iv = futures[f]
                try:
                    df = f.result()
                    console_lines.append(
                        f"  [green]✓[/green] {coin:6s} {iv:4s}  {len(df):5,} bars"
                    )
                except Exception as exc:
                    console_lines.append(
                        f"  [red]✗[/red] {coin:6s} {iv:4s}  {exc}"
                    )

        for line in sorted(console_lines):
            try:
                from rich.console import Console
                Console().print(line)
            except Exception:
                print(line)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _new_candle_available(self, key: tuple, interval: str) -> bool:
        """
        True when a new candle has closed since the last bar in the cache.

        last_bar_open + interval_s + buffer_s <= now
        """
        df = self._frames.get(key)
        if df is None or df.empty:
            return True
        last_open = df.index[-1].to_pydatetime()
        if last_open.tzinfo is None:
            last_open = last_open.replace(tzinfo=timezone.utc)
        next_open    = last_open + timedelta(seconds=_INTERVAL_SECONDS[interval])
        candle_ready = next_open + timedelta(seconds=self._FETCH_BUFFER_S)
        return datetime.now(tz=timezone.utc) >= candle_ready

    def _full_fetch(self, coin: str, interval: str, lookback_days: int, key: tuple):
        since_ms = int(
            (datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000
        )
        df = None
        try:
            df = fetch_hl_candles(coin, interval, since_ms)
            bars_available = len(df)
            bars_needed    = int(lookback_days * 24 * 3600 / _INTERVAL_SECONDS[interval])
            # Fall back to Kraken if HL history is materially shorter than requested
            if bars_available < bars_needed * 0.7:
                raise ValueError(
                    f"HL only returned {bars_available} bars, need ~{bars_needed}"
                )
        except Exception as hl_exc:
            logger.warning(f"{coin} {interval}: HL fetch failed ({hl_exc}), trying Kraken")
            # Map HL coin name back to a Kraken symbol via config if possible
            kraken_sym = _hl_to_kraken(coin)
            if kraken_sym:
                try:
                    df = _fetch_kraken_candles(kraken_sym, interval, lookback_days)
                except Exception as kr_exc:
                    raise RuntimeError(
                        f"Both HL and Kraken failed for {coin} {interval}: {kr_exc}"
                    ) from kr_exc
            else:
                raise RuntimeError(
                    f"HL fetch failed and no Kraken symbol for {coin}: {hl_exc}"
                ) from hl_exc

        self._frames[key]     = df
        self._last_fetch[key] = datetime.now(tz=timezone.utc)

    def _incremental_fetch(self, coin: str, interval: str, key: tuple):
        existing    = self._frames[key]
        last_ts_ms  = int(existing.index[-1].timestamp() * 1000) + 1
        try:
            new_df   = fetch_hl_candles(coin, interval, last_ts_ms)
            new_rows = new_df[new_df.index > existing.index[-1]]
            if not new_rows.empty:
                self._frames[key] = pd.concat([existing, new_rows])
                logger.debug(f"Cache updated {coin} {interval}: +{len(new_rows)} bars")
        except Exception as exc:
            # Keep stale data rather than crash — next tick will retry
            logger.warning(f"Incremental fetch failed for {coin} {interval}: {exc}")
        self._last_fetch[key] = datetime.now(tz=timezone.utc)


def _hl_to_kraken(hl_coin: str) -> str | None:
    """Return the Kraken symbol for a given HL coin name, or None."""
    try:
        import config
        for asset in config.ASSETS:
            if asset["hl"] == hl_coin:
                return asset["kraken"]
    except Exception:
        pass
    return None


# ── Live price / funding helpers (unchanged API, now use shared session) ─────

def fetch_ohlcv(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """
    Legacy entry point used by backtest and paper_trade paths.
    Fetches from Kraken via ccxt (not cached).
    """
    return _fetch_kraken_candles(symbol, timeframe, days)


def fetch_latest_prices() -> dict[str, float]:
    """
    Current mid prices from Hyperliquid for all perps.
    Used by the fast stop-check loop. Shares the persistent session.
    """
    try:
        resp = _hl_session.post(
            HL_API, json={"type": "allMids"}, timeout=HL_TIMEOUT
        )
        resp.raise_for_status()
        return {name: float(price) for name, price in resp.json().items() if price}
    except Exception as exc:
        logger.warning(f"Price fetch failed: {exc}")
        return {}


def fetch_funding_rates() -> dict[str, float]:
    """
    Predicted hourly funding rates from Hyperliquid for all perps.
    Shares the persistent session.
    """
    try:
        resp = _hl_session.post(
            HL_API, json={"type": "metaAndAssetCtxs"}, timeout=HL_TIMEOUT
        )
        resp.raise_for_status()
        meta, ctxs = resp.json()
        return {
            asset["name"]: float(ctx.get("funding", 0.0))
            for asset, ctx in zip(meta["universe"], ctxs)
        }
    except Exception as exc:
        logger.warning(f"Funding rate fetch failed: {exc}")
        return {}
