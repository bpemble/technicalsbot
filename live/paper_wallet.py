"""
Paper wallet — persists state to a JSON file across restarts.
Tracks capital, open positions, and closed trade history.
"""
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional


class PaperWallet:
    def __init__(self, state_file: str, initial_capital: float):
        self.state_file = state_file
        self.initial_capital = initial_capital
        self.state = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                # Fallback: persist initial_capital if missing from file
                if "initial_capital" not in data:
                    data["initial_capital"] = self.initial_capital
                return data
            except json.JSONDecodeError as e:
                print(f"Warning: corrupt wallet file ({e}), starting fresh", file=sys.stderr)
                return {
                    "capital": self.initial_capital,
                    "initial_capital": self.initial_capital,
                    "positions": {},
                    "trades": [],
                }
        return {
            "capital": self.initial_capital,
            "initial_capital": self.initial_capital,
            "positions": {},   # strategy_name -> position dict
            "trades": [],
        }

    def save(self):
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
        os.replace(tmp, self.state_file)

    # ------------------------------------------------------------------
    # Capital
    # ------------------------------------------------------------------

    @property
    def capital(self) -> float:
        return self.state["capital"]

    @capital.setter
    def capital(self, value: float):
        self.state["capital"] = value

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def has_position(self, strategy: str) -> bool:
        return strategy in self.state["positions"]

    def get_position(self, strategy: str) -> Optional[dict]:
        return self.state["positions"].get(strategy)

    def open_position(
        self,
        strategy: str,
        direction: str,
        entry_price: float,
        size: float,
        stop_loss: float,
        take_profit: float,
        fee: float,
    ):
        if strategy in self.state["positions"]:
            raise ValueError(f"Position already open for {strategy}")
        self.state["capital"] -= fee
        self.state["positions"][strategy] = {
            "strategy":    strategy,
            "direction":   direction,
            "entry_price": entry_price,
            "size":        size,
            "stop_loss":   stop_loss,
            "take_profit": take_profit,
            "entry_time":  datetime.now(tz=timezone.utc).isoformat(),
            "entry_fee":   fee,
        }
        self.save()

    def close_position(
        self,
        strategy: str,
        exit_price: float,
        exit_reason: str,
        fee: float,
    ) -> Optional[dict]:
        pos = self.state["positions"].pop(strategy, None)
        if pos is None:
            return None

        direction = pos["direction"]
        entry_price = pos["entry_price"]
        size = pos["size"]

        if direction == "long":
            gross_pnl = (exit_price - entry_price) * size
        else:
            gross_pnl = (entry_price - exit_price) * size

        # entry fee was already deducted from capital at open_position time,
        # so here we only credit gross_pnl and debit the exit fee.
        net_pnl = gross_pnl - fee
        self.state["capital"] += gross_pnl - fee

        trade = {
            **pos,
            "exit_price":  exit_price,
            "exit_time":   datetime.now(tz=timezone.utc).isoformat(),
            "exit_reason": exit_reason,
            "exit_fee":    fee,
            "gross_pnl":   gross_pnl,
            "net_pnl":     net_pnl,
        }
        self.state["trades"].append(trade)
        self.save()
        return trade

    def unrealized_pnl(self, strategy: str, current_price: float) -> float:
        pos = self.get_position(strategy)
        if pos is None:
            return 0.0
        if pos["direction"] == "long":
            return (current_price - pos["entry_price"]) * pos["size"]
        else:
            return (pos["entry_price"] - current_price) * pos["size"]

    def total_unrealized_pnl(self, prices: dict) -> float:
        total = 0.0
        for strategy, pos in self.state["positions"].items():
            price = prices.get(strategy, pos["entry_price"])
            total += self.unrealized_pnl(strategy, price)
        return total

    @property
    def trades(self) -> list:
        return self.state["trades"]

    @property
    def positions(self) -> dict:
        return self.state["positions"]

    def total_return_pct(self) -> float:
        ic = self.state.get("initial_capital", self.initial_capital)
        return ((self.capital - ic) / ic) * 100
