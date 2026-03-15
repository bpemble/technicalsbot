"""
Paper wallet monitor — reads paper_wallet.json and prints current status.
Usage: python monitor.py

Run this in a separate terminal at any time to check on the paper trader.
"""
import json
import os
import sys
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

import config

console = Console()


def load_wallet(path: str) -> dict:
    if not os.path.exists(path):
        console.print(f"[red]No wallet file found at '{path}'. Has the paper trader been started yet?[/red]")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def fmt_pnl(val: float) -> str:
    color = "green" if val >= 0 else "red"
    return f"[{color}]{'+' if val >= 0 else ''}{val:,.2f}[/{color}]"


def main():
    state = load_wallet(config.PAPER_STATE_FILE)

    capital   = state["capital"]
    positions = state.get("positions", {})
    trades    = state.get("trades", [])
    initial   = config.PAPER_CAPITAL

    # ---- Summary panel ------------------------------------------
    total_return = ((capital - initial) / initial) * 100
    ret_color    = "green" if total_return >= 0 else "red"

    wins   = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] <= 0]
    win_rate = (len(wins) / len(trades) * 100) if trades else 0
    gross_pnl = sum(t["net_pnl"] for t in trades)

    summary = Table(box=box.SIMPLE, show_header=False, expand=False)
    summary.add_column("K", style="dim",  min_width=24)
    summary.add_column("V", style="bold", min_width=18)

    summary.add_row("Initial capital",   f"${initial:,.2f}")
    summary.add_row("Current cash",      f"${capital:,.2f}")
    summary.add_row("Total return",      f"[{ret_color}]{total_return:+.2f} %[/{ret_color}]")
    summary.add_row("Realised PnL",      fmt_pnl(gross_pnl))
    summary.add_row("",                  "")
    summary.add_row("Closed trades",     str(len(trades)))
    summary.add_row("Win rate",          f"{win_rate:.1f} %  ({len(wins)}W / {len(losses)}L)")
    summary.add_row("Open positions",    str(len(positions)))

    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    console.print()
    console.print(Panel(summary, title=f"[bold cyan]Paper Wallet — {ts}[/bold cyan]",
                        border_style="cyan", expand=False))

    # ---- Open positions -----------------------------------------
    if positions:
        console.print()
        pos_table = Table(title="Open Positions", box=box.SIMPLE_HEAVY,
                          title_style="bold cyan", show_header=True,
                          header_style="bold magenta")
        pos_table.add_column("Strategy",    min_width=8)
        pos_table.add_column("Direction",   min_width=7)
        pos_table.add_column("Entry $",     justify="right", min_width=10)
        pos_table.add_column("Size (ETH)",  justify="right", min_width=10)
        pos_table.add_column("Stop",        justify="right", min_width=10)
        pos_table.add_column("Target",      justify="right", min_width=10)
        pos_table.add_column("Opened",      min_width=17)

        for name, pos in positions.items():
            dir_color = "cyan" if pos["direction"] == "long" else "magenta"
            pos_table.add_row(
                name,
                f"[{dir_color}]{pos['direction'].upper()}[/{dir_color}]",
                f"${pos['entry_price']:,.2f}",
                f"{pos['size']:.4f}",
                f"${pos['stop_loss']:,.2f}",
                f"${pos['take_profit']:,.2f}",
                pos["entry_time"][:16].replace("T", " "),
            )
        console.print(pos_table)

    # ---- Closed trade log ---------------------------------------
    if trades:
        console.print()
        n = min(20, len(trades))
        trade_table = Table(title=f"Last {n} Closed Trades", box=box.SIMPLE_HEAVY,
                            title_style="bold cyan", show_header=True,
                            header_style="bold magenta")
        trade_table.add_column("#",          justify="right", style="dim", min_width=4)
        trade_table.add_column("Strategy",   min_width=8)
        trade_table.add_column("Dir",        min_width=6)
        trade_table.add_column("Entry $",    justify="right", min_width=10)
        trade_table.add_column("Exit $",     justify="right", min_width=10)
        trade_table.add_column("Net PnL",    justify="right", min_width=11)
        trade_table.add_column("Reason",     min_width=14)
        trade_table.add_column("Opened",     min_width=13)
        trade_table.add_column("Closed",     min_width=13)

        reason_colors = {
            "stop_loss":      "red",
            "take_profit":    "green",
            "signal_reverse": "yellow",
        }

        for i, t in enumerate(trades[-n:], start=len(trades) - n + 1):
            dir_color = "cyan" if t["direction"] == "long" else "magenta"
            rc = reason_colors.get(t.get("exit_reason", ""), "white")
            trade_table.add_row(
                str(i),
                t.get("strategy", "—"),
                f"[{dir_color}]{t['direction'].upper()}[/{dir_color}]",
                f"${t['entry_price']:,.2f}",
                f"${t['exit_price']:,.2f}",
                fmt_pnl(t["net_pnl"]),
                f"[{rc}]{t.get('exit_reason', '—')}[/{rc}]",
                t["entry_time"][:10],
                t["exit_time"][:10],
            )
        console.print(trade_table)

    # ---- Performance summary ------------------------------------
    if trades:
        console.print()
        avg_win  = sum(t["net_pnl"] for t in wins)  / len(wins)  if wins   else 0
        avg_loss = sum(t["net_pnl"] for t in losses) / len(losses) if losses else 0
        expectancy = gross_pnl / len(trades)

        perf = Table(box=box.SIMPLE, show_header=False, expand=False)
        perf.add_column("K", style="dim",  min_width=22)
        perf.add_column("V", style="bold", min_width=14)
        perf.add_row("Avg win",       fmt_pnl(avg_win))
        perf.add_row("Avg loss",      fmt_pnl(avg_loss))
        perf.add_row("Expectancy",    fmt_pnl(expectancy))
        perf.add_row("Profit factor",
            f"[{'green' if losses else 'dim'}]"
            f"{abs(sum(t['net_pnl'] for t in wins)) / max(abs(sum(t['net_pnl'] for t in losses)), 0.01):.2f}"
            f"[/{'green' if losses else 'dim'}]"
        )
        console.print(Panel(perf, title="[bold]Performance[/bold]", border_style="dim", expand=False))

    console.print()


if __name__ == "__main__":
    main()
