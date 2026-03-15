"""
ETH/USD Regime Trader — Paper Mode
Usage: python regime_trade.py

Always-in conviction-weighted positioning. The bot continuously scores
the market from -100 to +100 and sizes its position proportionally.
State is saved to paper_wallet.json between restarts.
Press Ctrl+C to stop.
"""
from live.always_in_runner import run

if __name__ == "__main__":
    run()
