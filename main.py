"""
Regime Strategy Backtester — entry point.

Usage:
    python main.py              # backtest default asset (config.BACKTEST_COIN)
    python main.py --coin BTC   # backtest a different asset
"""
import argparse
from backtest.regime_backtest import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, default=None,
                        help="HL asset name to backtest (default: config.BACKTEST_COIN)")
    args = parser.parse_args()
    run(coin=args.coin)
