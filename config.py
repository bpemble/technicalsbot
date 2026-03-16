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
PRIMARY_TF = "1d"             # Trend / bias timeframe (unlimited Kraken history)
ENTRY_TF = "4h"               # Entry timing timeframe (~4 months Kraken history)
LOOKBACK_DAYS = 730           # 2 years of history (daily); 4h gets ~120 days

# --------------- Capital & Risk ------------------------------
INITIAL_CAPITAL = 10_000.0   # USDC
LEVERAGE = 3                  # Conservative leverage
RISK_PER_TRADE = 0.02         # 2 % of capital risked per trade

# --------------- Stop / TP -----------------------------------
ATR_STOP_MULTIPLIER = 2.0
ATR_TP_MULTIPLIER = 3.0

# --------------- Indicators ----------------------------------
ATR_PERIOD = 14
ADX_PERIOD = 14               # Wilder default, same window as ATR

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
MIN_CONVICTION_SCORE  = 15.0   # minimum absolute score to hold any position

# --------------- Funding rate adjustment ---------------------
# Hyperliquid reports hourly funding rates (e.g. 0.0001 = 0.01 %/hr).
# When funding is crowded in the same direction as the regime score,
# the score is penalised to reflect the overcrowded-trade risk.
FUNDING_NORM_RATE   = 0.0003   # hourly rate that maps to full penalty (0.03 %/hr)
FUNDING_PENALTY_MAX = 20.0     # max score points deducted (out of 100)

# --------------- Paper trading --------------------------------
PAPER_CAPITAL    = 10_000.0    # starting paper capital (USDC)
REGIME_STATE_FILE = "wallet_regime.json"   # persisted wallet state for regime trader
SWING_STATE_FILE  = "wallet_swing.json"    # persisted wallet state for swing/scalp trader
POLL_INTERVAL_SEC = 900        # seconds between full regime scans (15 min = 1 × 15m candle)
FAST_POLL_SEC     = 60         # seconds between lightweight hard-stop checks

# --------------- Logging -------------------------------------
LOG_FILE = "bot.log"

# --------------- Multi-asset portfolio -----------------------
MAX_POSITIONS = 10          # max simultaneous open positions

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
