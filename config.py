# ============================================================
# config.py — Central configuration for ETH/USDC Perps Bot
# ============================================================

# --------------- Data ----------------------------------------
SYMBOL = "ETH/USD"            # Kraken spot symbol (price tracks perp closely)
TIMEFRAMES = ["4h", "1d"]
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

# --------------- Paper trading --------------------------------
PAPER_CAPITAL    = 10_000.0    # starting paper capital (USDC)
PAPER_STATE_FILE = "paper_wallet.json"   # persisted wallet state
POLL_INTERVAL_SEC = 900        # 15 minutes between live ticks
