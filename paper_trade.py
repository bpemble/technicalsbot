"""
ETH/USD Paper Trader
Usage: python paper_trade.py

Runs both the swing strategy (daily + 4h) and scalp strategy (1h + 15m)
in paper-trading mode. State is saved to paper_wallet.json between runs.
Press Ctrl+C to stop.
"""
from live.runner import run

if __name__ == "__main__":
    run()
