# ============================================================
# config.py — Central configuration for ETH/USDC Perps Bot
# ============================================================

import os

# --------------- Live trading credentials --------------------
# Never hardcode keys. Set these as environment variables on the server:
#
#   export HL_PRIVATE_KEY="0xabc123..."
#   export HL_WALLET_ADDRESS="0xdef456..."
#
# Both are None while paper trading — the live execution layer
# will refuse to submit real orders if either is missing.
HL_PRIVATE_KEY    = os.environ.get("HL_PRIVATE_KEY")
HL_WALLET_ADDRESS = os.environ.get("HL_WALLET_ADDRESS")

# --------------- Data ----------------------------------------
SYMBOL = "ETH/USD"            # Kraken spot symbol (price tracks perp closely)
BACKTEST_COIN    = "ETH"        # HL coin name used by the backtest (maps to fetch_ohlcv_hl)
PRIMARY_TF = "1d"             # Trend / bias timeframe (unlimited Kraken history)
ENTRY_TF = "4h"               # Entry timing timeframe (~4 months Kraken history)
LOOKBACK_DAYS = 730           # 2 years of history (daily); 4h gets ~120 days

# --------------- Capital & Risk ------------------------------
INITIAL_CAPITAL = 10_000.0   # USDC
LEVERAGE = 3                  # Used by swing/scalp runner only
RISK_PER_TRADE = 0.02         # 2 % of capital risked per trade (swing/scalp)

# --------------- Kelly sizing (regime trader) ----------------
# Full Kelly = 13 % derived from 2-yr backtest (win rate 46 %, avg W/L 1.62×).
# Half Kelly used for live deployment — conservative starting point.
# Each position is sized as an independent Kelly bet (6.5 % of equity at risk
# per position, scaled by conviction), capped so total portfolio notional
# never exceeds MAX_NOTIONAL_FACTOR × equity.
KELLY_FRACTION      = 0.0439  # optimised (was 0.0493) — slightly more conservative
MAX_NOTIONAL_FACTOR = 2.0     # hard ceiling: total notional ≤ 2× equity

# --------------- Stop / TP -----------------------------------
ATR_STOP_MULTIPLIER = 3.0833     # optimised (was 3.2184)
ATR_TP_MULTIPLIER = 2.5380       # optimised (was 1.5172) — wider TP, let winners run more
TRAIL_ACTIVATION_ATR = 2.2968   # optimised (was 1.7536) — trail activates even later
TRAIL_ATR_MULTIPLIER  = 0.8826  # optimised (was 0.8019)

# --------------- Indicators ----------------------------------
ATR_PERIOD = 14
ADX_PERIOD = 14               # Wilder default, same window as ATR
ADX_TREND_THRESHOLD  = 26.0     # optimised (was 24.0) — slightly more selective trend filter
BB_BW_LOOKBACK       = 50       # rolling window for BB bandwidth percentile calibration

# --------------- EMA -----------------------------------------
EMA_FAST = 21
EMA_SLOW = 55

# --------------- RSI -----------------------------------------
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# --------------- MACD ----------------------------------------
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# --------------- Costs ---------------------------------------
EXCHANGE_FEE = 0.00035        # Hyperliquid taker fee  (0.035 %)
FUNDING_RATE_DAILY = 0.0003   # Approximate daily funding cost (for P&L modelling)
SLIPPAGE = 0.0001             # 0.01 % additional cost on entries

# --------------- Scalp strategy parameters -------------------
SCALP_TREND_TF   = "1h"        # trend timeframe for scalp
SCALP_ENTRY_TF   = "15m"       # entry timeframe for scalp
SCALP_ATR_STOP   = 1.5         # tighter stop for scalps
SCALP_ATR_TP     = 2.5         # tighter TP for scalps
SCALP_RISK_PER_TRADE = 0.01    # 1% risk per scalp trade (smaller, more frequent)
SCALP_LOOKBACK_DAYS  = 3       # 15m data: Kraken gives ~3 days

# --------------- Strategy ------------------------------------
MIN_CONVICTION_SCORE  = 12.74  # optimised (was 13.79)

# --------------- Funding rate adjustment ---------------------
# Hyperliquid reports hourly funding rates (e.g. 0.0001 = 0.01 %/hr).
# When funding is crowded in the same direction as the regime score,
# the score is penalised to reflect the overcrowded-trade risk.
FUNDING_NORM_RATE   = 0.0003   # hourly rate that maps to full penalty (0.03 %/hr)
FUNDING_PENALTY_MAX = 20.0     # max score points deducted (out of 100)
FUNDING_CONTRARIAN_THRESHOLD = 0.5   # normalised rate at which opposite funding becomes a bonus
FUNDING_CONTRARIAN_BONUS     = 10.0  # max score points added for contrarian funding signal

# --------------- Paper trading --------------------------------
PAPER_CAPITAL    = 10_000.0    # starting paper capital (USDC)
REGIME_STATE_FILE = "wallet_regime.json"   # persisted wallet state for regime trader
SWING_STATE_FILE  = "wallet_swing.json"    # persisted wallet state for swing/scalp trader
POLL_INTERVAL_SEC = 300        # seconds between full regime scans (fastest signal = 15m candle)
FAST_POLL_SEC     = 15         # seconds between lightweight hard-stop checks

# --------------- Logging -------------------------------------
LOG_FILE             = "bot.log"
SCORES_FILE          = "scores.json"          # latest regime scores written each scan
EQUITY_HISTORY_FILE  = "equity_history.jsonl" # equity snapshots written each tick

# --------------- Open Interest signal ------------------------
# OI growing in same direction as score → amplify (conviction confirmed).
# OI shrinking against trend → dampen (possible unwinding).
OI_AMPLIFY_MAX   = 0.20   # max 20% score boost when OI grows with trend
OI_DAMPEN_MAX    = 0.20   # max 20% score cut when OI shrinks
OI_CHANGE_CLAMP  = 0.10   # clamp per-tick OI change to ±10% before scaling

# --------------- Fear & Greed Index --------------------------
# Fetched hourly from alternative.me; used as a portfolio-level throttle.
FNG_EXTREME_FEAR      = 25    # index ≤ 25 → no penalty (contrarian open)
FNG_EXTREME_GREED     = 75    # index ≥ 75 → reduce long bias
FNG_POSITIONS_REDUCE  = 2     # cut MAX_POSITIONS by this during extreme greed
FNG_GREED_LONG_PENALTY = 5.0  # score penalty on long signals during extreme greed
FNG_CACHE_TTL_SEC     = 3600  # seconds before re-fetching F&G (hourly)

# --------------- Volatility regime (normalised ATR) ----------
# Low-vol markets have thin follow-through; reduce size when ATR percentile is low.
NORM_ATR_LOOKBACK = 60    # rolling window (bars) for ATR percentile calibration
VOL_REGIME_LOW    = 0.81  # optimised (was 0.70) — engages vol brake earlier (more conservative)
VOL_REGIME_MIN    = 0.60  # minimum size multiplier in low-vol regime

# --------------- MA200 distance (extension brake) ------------
# Penalise entries when price is stretched far from the 200-bar MA.
MA200_PERIOD    = 200     # period for 200-bar moving average
MA200_NEAR_BAND = 0.15    # within 15% of MA200 → full size
MA200_FAR_BAND  = 0.35    # beyond 35% from MA200 → half size (max penalty)

# --------------- Multi-asset portfolio -----------------------
MAX_POSITIONS = 8           # max simultaneous open positions
TIER_CORR_FACTOR = 0.7986       # optimised (was 0.8694) — more aggressive correlation discount

ASSETS = [
    # Tier 1 — deepest liquidity
    {"name": "BTC",  "kraken": "BTC/USD",  "hl": "BTC",  "tier": 1},
    {"name": "ETH",  "kraken": "ETH/USD",  "hl": "ETH",  "tier": 1},
    {"name": "SOL",  "kraken": "SOL/USD",  "hl": "SOL",  "tier": 1},
    # Tier 2
    {"name": "AVAX", "kraken": "AVAX/USD", "hl": "AVAX", "tier": 2},
    {"name": "LINK", "kraken": "LINK/USD", "hl": "LINK", "tier": 2},
    {"name": "DOGE", "kraken": "DOGE/USD", "hl": "DOGE", "tier": 2},
    {"name": "ADA",  "kraken": "ADA/USD",  "hl": "ADA",  "tier": 2},
    {"name": "DOT",  "kraken": "DOT/USD",  "hl": "DOT",  "tier": 2},
    # Tier 3
    {"name": "ARB",  "kraken": "ARB/USD",  "hl": "ARB",  "tier": 3},
    {"name": "OP",   "kraken": "OP/USD",   "hl": "OP",   "tier": 3},
    {"name": "SUI",  "kraken": "SUI/USD",  "hl": "SUI",  "tier": 3},
    {"name": "APT",  "kraken": "APT/USD",  "hl": "APT",  "tier": 3},
    {"name": "INJ",  "kraken": "INJ/USD",  "hl": "INJ",  "tier": 3},
    {"name": "TIA",  "kraken": "TIA/USD",  "hl": "TIA",  "tier": 3},
]
