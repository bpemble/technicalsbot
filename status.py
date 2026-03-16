#!/usr/bin/env python3
"""
Usage: python status.py

Prints a clean snapshot of the paper wallet: open positions, conviction,
sizing, live P&L, and account summary. Fetches live prices from Hyperliquid.
"""

import json
import os
import requests
from datetime import datetime, timezone
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

import config

console = Console()

STATE_FILES = {
    "Regime Trader":  config.REGIME_STATE_FILE,
    "Swing Trader":   config.SWING_STATE_FILE,
}

SCORES = {}  # populated from last scan in stdout.log


def load_last_scores() -> dict[str, float]:
    """Parse the most recent per-asset scores from stdout.log."""
    log_path = os.path.join(os.path.dirname(__file__), config.LOG_FILE.replace("bot.log", "stdout.log"))
    if not os.path.exists(log_path):
        return {}
    scores = {}
    import re
    pattern = re.compile(r"^\s{2}(\w+)\.\.\.\s+([+-]?\d+\.?\d*)")
    try:
        # Walk lines in reverse to find the last complete scan block
        with open(log_path, "rb") as f:
            # Read last 8KB — enough to cover one full scan
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read().decode("utf-8", errors="ignore")
        seen = set()
        for line in reversed(tail.splitlines()):
            m = pattern.match(line)
            if m:
                name, score = m.group(1), float(m.group(2))
                if name not in seen:
                    scores[name] = score
                    seen.add(name)
    except Exception:
        pass
    return scores


def fetch_live_prices() -> dict[str, float]:
    try:
        resp = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={"type": "allMids"},
            timeout=5,
        )
        resp.raise_for_status()
        return {k: float(v) for k, v in resp.json().items() if v}
    except Exception as e:
        console.print(f"[yellow]Warning: could not fetch live prices ({e})[/yellow]")
        return {}


def load_wallet(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt_price(p: float) -> str:
    if p == 0:
        return "—"
    return f"${p:,.4f}" if p < 10 else f"${p:,.2f}"


def fmt_pnl(v: float) -> str:
    color = "green" if v >= 0 else "red"
    sign  = "+" if v >= 0 else ""
    return f"[{color}]{sign}${v:,.2f}[/{color}]"


def fmt_ret(v: float) -> str:
    color = "green" if v >= 0 else "red"
    sign  = "+" if v >= 0 else ""
    return f"[{color}]{sign}{v:.2f}%[/{color}]"


def age(entry_time_str: str) -> str:
    try:
        t = datetime.fromisoformat(entry_time_str)
        delta = datetime.now(tz=timezone.utc) - t
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m = rem // 60
        return f"{h}h {m:02d}m"
    except Exception:
        return "—"


def print_wallet(label: str, wallet: dict, prices: dict):
    positions  = wallet.get("positions", {})
    capital    = wallet.get("capital", 0.0)
    initial    = wallet.get("initial_capital", config.PAPER_CAPITAL)
    trades     = wallet.get("trades", [])

    # ── Positions table ──────────────────────────────────────────────
    t = Table(
        title=f"[bold cyan]{label} — Open Positions[/bold cyan]",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        title_style="bold cyan",
        show_footer=False,
        pad_edge=True,
    )
    t.add_column("Asset",      min_width=6,  style="bold")
    t.add_column("Dir",        min_width=6,  justify="center")
    t.add_column("Score",      min_width=7,  justify="right")
    t.add_column("Conv",       min_width=6,  justify="right")
    t.add_column("Entry",      min_width=11, justify="right")
    t.add_column("Now",        min_width=11, justify="right")
    t.add_column("Size",       min_width=12, justify="right")
    t.add_column("Notional",   min_width=10, justify="right")
    t.add_column("Unreal PnL", min_width=11, justify="right")
    t.add_column("Stop",       min_width=11, justify="right")
    t.add_column("Age",        min_width=8,  justify="right")

    total_upnl     = 0.0
    total_notional = 0.0

    if not positions:
        console.print(Panel(f"[dim]No open positions in {label}[/dim]", border_style="dim"))
        return

    for name, pos in sorted(positions.items()):
        hl_name       = name  # asset names match HL names in config
        current_price = prices.get(hl_name, pos["entry_price"])
        entry_price   = pos["entry_price"]
        size          = pos["size"]
        direction     = pos["direction"]

        upnl = (current_price - entry_price) * size if direction == "long" \
               else (entry_price - current_price) * size
        notional = size * current_price
        total_upnl     += upnl
        total_notional += notional

        score = SCORES.get(name)
        score_str = f"[green]+{score:.1f}[/green]" if score and score > 0 \
                    else f"[red]{score:.1f}[/red]" if score and score < 0 \
                    else "[dim]—[/dim]"
        conv_str  = f"{abs(score)/100*100:.0f}%" if score is not None else "—"

        dir_color = "cyan" if direction == "long" else "magenta"
        upnl_color = "green" if upnl >= 0 else "red"
        sign = "+" if upnl >= 0 else ""

        t.add_row(
            name,
            f"[{dir_color}]{direction.upper()}[/{dir_color}]",
            score_str,
            conv_str,
            fmt_price(entry_price),
            fmt_price(current_price),
            f"{size:,.2f}",
            f"${notional:,.0f}",
            f"[{upnl_color}]{sign}${upnl:,.2f}[/{upnl_color}]",
            fmt_price(pos["stop_loss"]),
            age(pos.get("entry_time", "")),
        )

    console.print()
    console.print(t)

    # ── Summary panel ────────────────────────────────────────────────
    equity = capital + total_upnl
    ret    = (equity - initial) / initial * 100

    wins   = sum(1 for tr in trades if tr.get("net_pnl", 0) >= 0)
    losses = sum(1 for tr in trades if tr.get("net_pnl", 0) <  0)
    win_rate = wins / len(trades) * 100 if trades else 0.0

    console.print(
        f"  Cash: [bold]${capital:,.2f}[/bold]  "
        f"Unrealised: {fmt_pnl(total_upnl)}  "
        f"Equity: [bold]${equity:,.2f}[/bold]  "
        f"Return: {fmt_ret(ret)}  "
        f"Notional: [bold]${total_notional:,.0f}[/bold]  "
        f"Trades: [bold]{len(trades)}[/bold]  "
        f"Win rate: [bold]{win_rate:.0f}%[/bold]"
    )
    console.print()


def main():
    console.print()
    console.print(Panel(
        "[bold cyan]Bot Status Snapshot[/bold cyan]\n"
        f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  —  fetching live prices from Hyperliquid...[/dim]",
        border_style="cyan", expand=False,
    ))

    prices = fetch_live_prices()
    SCORES.update(load_last_scores())

    any_printed = False
    for label, path in STATE_FILES.items():
        wallet = load_wallet(path)
        if wallet is None:
            console.print(f"[dim]{label}: no state file found ({path})[/dim]\n")
            continue
        print_wallet(label, wallet, prices)
        any_printed = True

    if not any_printed:
        console.print("[yellow]No wallet state files found. Is the bot running?[/yellow]")

    if prices:
        console.print(
            f"  [dim]Live prices from Hyperliquid  |  "
            f"Scores shown are from last bot scan (15-min cycle)[/dim]\n"
        )


if __name__ == "__main__":
    main()
