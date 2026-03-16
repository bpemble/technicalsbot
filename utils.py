# ============================================================
# utils.py — Shared display utilities
# ============================================================

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

TZ_CHICAGO = ZoneInfo("America/Chicago")


def now_chicago() -> datetime:
    """Current time in Chicago timezone."""
    return datetime.now(tz=TZ_CHICAGO)


def fmt_ts(ts, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """
    Format a timestamp for display in Chicago time.

    Accepts:
      - datetime objects (naive assumed UTC, aware converted)
      - ISO 8601 strings (from paper_wallet.json)
      - pandas Timestamps
      - None / empty string → returns "—"
    """
    if ts is None:
        return "—"
    try:
        s = str(ts).strip()
        if not s or s == "nan":
            return "—"
        # Parse to datetime
        if isinstance(ts, datetime):
            dt = ts
        else:
            # Handle pandas Timestamp or ISO string
            import pandas as pd
            dt = pd.Timestamp(s)
            # Convert pandas Timestamp to Python datetime
            dt = dt.to_pydatetime()

        # Attach UTC if naive
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        # Convert to Chicago
        dt_chicago = dt.astimezone(TZ_CHICAGO)
        return dt_chicago.strftime(fmt)
    except Exception:
        return str(ts)[:16] if ts else "—"


def fmt_now(fmt: str = "%Y-%m-%d %H:%M CT") -> str:
    """Current Chicago time as a formatted string."""
    return now_chicago().strftime(fmt)
