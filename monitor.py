"""
Paper wallet monitor — reads wallet state file and prints current status.
Usage:
  python monitor.py                          # shows regime wallet (default)
  python monitor.py --wallet wallet_swing.json

Displays the multi-asset portfolio: open positions, closed trade history,
and performance statistics. Run in a separate terminal at any time.
Note: unrealised PnL is not available here (no live prices). Run
always_in_runner.py or runner.py for live P&L tracking.
"""
import argparse
import json
import os
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

import config
from utils import fmt_ts, fmt_now

console = Console()


def load_wallet(path: str) -> dict:
    if not os.path.exists(path):
        console.print(
            f"[red]No wallet file found at '{path}'. "
            f"Has the paper trader been started yet?[/red]"
        )
        sys.exit(1)
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        console.print(
            f"[red]Wallet file '{path}' is corrupt and could not be read: {e}[/red]"
        )
        sys.exit(1)


def fmt_pnl(val: float) -> str:
    color = "green" if val >= 0 else "red"
    sign  = "+" if val >= 0 else ""
    return f"[{color}]{sign}{val:,.2f}[/{color}]"


def fmt_price(val: float) -> str:
    return f"${val:,.4f}" if val < 1.0 else f"${val:,.2f}"


def main():
    parser = argparse.ArgumentParser(description="Paper wallet monitor")
    parser.add_argument(
        "--wallet",
        default=config.REGIME_STATE_FILE,
        help=f"Path to wallet JSON file (default: {config.REGIME_STATE_FILE})",
    )
    args = parser.parse_args()

    wallet_path = args.wallet
    state = load_wallet(wallet_path)

    capital   = state["capital"]
    positions = state.get("positions", {})
    trades    = state.get("trades", [])
    initial   = state.get("initial_capital", config.PAPER_CAPITAL)

    wins   = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] <= 0]

    realised_pnl = sum(t["net_pnl"] for t in trades)
    win_rate     = (len(wins) / len(trades) * 100) if trades else 0.0

    # Equity = cash capital (unrealised requires live prices)
    equity       = capital
    total_return = (equity - initial) / initial * 100
    ret_color    = "green" if total_return >= 0 else "red"

    # ---- 1. Portfolio Summary panel ------------------------------
    summary = Table(box=box.SIMPLE, show_header=False, expand=False)
    summary.add_column("K", style="dim",  min_width=26)
    summary.add_column("V", style="bold", min_width=20)

    summary.add_row("Wallet file",        wallet_path)
    summary.add_row("Initial capital",    f"${initial:,.2f}")
    summary.add_row("Current cash",       f"${capital:,.2f}")
    summary.add_row("Equity (cash only)", f"${equity:,.2f}")
    summary.add_row(
        "Total return",
        f"[{ret_color}]{total_return:+.2f} %[/{ret_color}]",
    )
    summary.add_row("Realised PnL",       fmt_pnl(realised_pnl))
    summary.add_row("",                   "")
    summary.add_row("Closed trades",      str(len(trades)))
    summary.add_row(
        "Win rate",
        f"{win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)",
    )
    summary.add_row("Open positions",     str(len(positions)))
    summary.add_row(
        "",
        f"[dim](run always_in_runner.py or runner.py for live P&L)[/dim]",
    )

    console.print()
    console.print(Panel(
        summary,
        title=f"[bold cyan]Multi-Asset Paper Wallet — {fmt_now()}[/bold cyan]",
        border_style="cyan",
        expand=False,
    ))

    # ---- 2. Open Positions table ---------------------------------
    if positions:
        console.print()
        pt = Table(
            title="Open Positions",
            box=box.SIMPLE_HEAVY,
            title_style="bold cyan",
            show_header=True,
            header_style="bold magenta",
        )
        pt.add_column("Asset",       min_width=6)
        pt.add_column("Dir",         min_width=6,  justify="center")
        pt.add_column("Entry $",     min_width=12, justify="right")
        pt.add_column("Unreal PnL",  min_width=14, justify="right")
        pt.add_column("Stop",        min_width=12, justify="right")
        pt.add_column("Entry Time",  min_width=17)

        for asset_name, pos in positions.items():
            dir_color = "cyan" if pos["direction"] == "long" else "magenta"
            entry_time_str = fmt_ts(pos.get("entry_time"))
            pt.add_row(
                f"[bold]{asset_name}[/bold]",
                f"[{dir_color}]{pos['direction'].upper()}[/{dir_color}]",
                fmt_price(pos["entry_price"]),
                "[dim]—[/dim]",
                fmt_price(pos["stop_loss"]),
                entry_time_str,
            )

        console.print(pt)

    # ---- 3. Closed Trades table (last 20) -----------------------
    if trades:
        console.print()
        n = min(20, len(trades))
        trade_table = Table(
            title=f"Last {n} Closed Trades",
            box=box.SIMPLE_HEAVY,
            title_style="bold cyan",
            show_header=True,
            header_style="bold magenta",
        )
        trade_table.add_column("#",        justify="right", style="dim", min_width=4)
        trade_table.add_column("Asset",    min_width=6)
        trade_table.add_column("Dir",      min_width=6,  justify="center")
        trade_table.add_column("Entry $",  min_width=12, justify="right")
        trade_table.add_column("Exit $",   min_width=12, justify="right")
        trade_table.add_column("Net PnL",  min_width=12, justify="right")
        trade_table.add_column("Reason",   min_width=14)
        trade_table.add_column("Opened",   min_width=11)
        trade_table.add_column("Closed",   min_width=11)

        reason_colors = {
            "hard_stop":      "red",
            "stop_loss":      "red",
            "take_profit":    "green",
            "regime_flip":    "yellow",
            "regime_flat":    "yellow",
            "regime_exit":    "yellow",
            "signal_reverse": "yellow",
            "resize":         "dim",
        }

        for i, t in enumerate(trades[-n:], start=len(trades) - n + 1):
            dir_color = "cyan" if t["direction"] == "long" else "magenta"
            reason    = t.get("exit_reason", "—")
            rc        = reason_colors.get(reason, "white")
            # Asset name is stored in the "strategy" field
            asset_name = t.get("strategy", "—")

            opened_str = fmt_ts(t.get("entry_time"), fmt="%Y-%m-%d")
            closed_str = fmt_ts(t.get("exit_time"),  fmt="%Y-%m-%d")

            trade_table.add_row(
                str(i),
                f"[bold]{asset_name}[/bold]",
                f"[{dir_color}]{t['direction'].upper()}[/{dir_color}]",
                fmt_price(t["entry_price"]),
                fmt_price(t["exit_price"]),
                fmt_pnl(t["net_pnl"]),
                f"[{rc}]{reason}[/{rc}]",
                opened_str,
                closed_str,
            )

        console.print(trade_table)

    # ---- 4. Performance stats -----------------------------------
    if trades:
        console.print()

        avg_win  = sum(t["net_pnl"] for t in wins)   / len(wins)   if wins   else 0.0
        avg_loss = sum(t["net_pnl"] for t in losses) / len(losses) if losses else 0.0
        expectancy = realised_pnl / len(trades)

        gross_wins   = sum(t["net_pnl"] for t in wins)
        gross_losses = abs(sum(t["net_pnl"] for t in losses))
        if gross_losses == 0:
            profit_factor = float("inf")
        else:
            profit_factor = gross_wins / gross_losses
        pf_color = "green" if profit_factor == float("inf") or profit_factor >= 1.0 else "red"
        pf_str = "∞" if profit_factor == float("inf") else f"{profit_factor:.2f}"

        best_trade  = max((t["net_pnl"] for t in trades), default=0.0)
        worst_trade = min((t["net_pnl"] for t in trades), default=0.0)

        perf = Table(box=box.SIMPLE, show_header=False, expand=False)
        perf.add_column("K", style="dim",  min_width=22)
        perf.add_column("V", style="bold", min_width=16)

        perf.add_row("Avg win",       fmt_pnl(avg_win))
        perf.add_row("Avg loss",      fmt_pnl(avg_loss))
        perf.add_row("Expectancy",    fmt_pnl(expectancy))
        perf.add_row(
            "Profit factor",
            f"[{pf_color}]{pf_str}[/{pf_color}]",
        )
        perf.add_row("Best trade",    fmt_pnl(best_trade))
        perf.add_row("Worst trade",   fmt_pnl(worst_trade))

        console.print(Panel(
            perf,
            title="[bold]Performance[/bold]",
            border_style="dim",
            expand=False,
        ))

    console.print()


if __name__ == "__main__":
    main()
