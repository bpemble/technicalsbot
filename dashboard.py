"""
Ethereal Capital — Hyperliquid Bot Dashboard
Run: uvicorn dashboard:app --host 0.0.0.0 --port 8080
"""

import json
import math
import os
from datetime import datetime, timezone

import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

import config

app = FastAPI(title="Ethereal Capital — Hyperliquid Bot")

HL_API      = "https://api.hyperliquid.xyz/info"
HL_TIMEOUT  = 5
_hl_session = requests.Session()
_hl_session.headers.update({"Content-Type": "application/json"})


# ── Data helpers ─────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _load_equity_history() -> list[dict]:
    """Load equity snapshots from the JSONL history file."""
    path = config.EQUITY_HISTORY_FILE
    if not os.path.exists(path):
        return []
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pts.append(json.loads(line))
                except Exception:
                    pass
    return pts


def _append_equity_snapshot(point: dict):
    """Append an equity snapshot to the history file (used by dashboard for live prices)."""
    try:
        with open(config.EQUITY_HISTORY_FILE, "a") as f:
            f.write(json.dumps(point) + "\n")
    except Exception:
        pass


def _live_prices() -> dict[str, float]:
    try:
        r = _hl_session.post(HL_API, json={"type": "allMids"}, timeout=HL_TIMEOUT)
        r.raise_for_status()
        return {k: float(v) for k, v in r.json().items() if v}
    except Exception:
        return {}


# ── API endpoint ─────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    wallet          = _load_json(config.REGIME_STATE_FILE)
    scores          = _load_json(config.SCORES_FILE)
    prices          = _live_prices()
    equity_history  = _load_equity_history()

    positions = wallet.get("positions", {})
    trades    = wallet.get("trades",    [])
    capital   = wallet.get("capital",   0.0)
    initial   = wallet.get("initial_capital", config.PAPER_CAPITAL)

    score_map = {a["name"]: a for a in scores.get("assets", [])}

    # ── Enrich open positions ─────────────────────────────────────────────────
    enriched_positions = []
    total_upnl = 0.0
    for name, pos in positions.items():
        live  = prices.get(name, pos["entry_price"])
        entry = pos["entry_price"]
        size  = pos["size"]
        upnl  = (live - entry) * size if pos["direction"] == "long" \
                else (entry - live) * size
        total_upnl += upnl
        sc = score_map.get(name, {})
        enriched_positions.append({
            "name":       name,
            "direction":  pos["direction"],
            "entry":      entry,
            "live":       live,
            "size":       size,
            "notional":   round(live * size, 2),
            "upnl":       round(upnl, 2),
            "stop":       pos.get("stop_loss", 0),
            "entry_time": pos.get("entry_time", ""),
            "score":      sc.get("score"),
            "conviction": sc.get("conviction"),
        })
    enriched_positions.sort(key=lambda p: abs(p.get("score") or 0), reverse=True)

    # ── Equity curve from closed trades ───────────────────────────────────────
    sorted_trades = sorted(trades, key=lambda t: t.get("exit_time", ""))
    running = initial
    equity_curve = [{"t": None, "v": initial}]  # start anchor
    for t in sorted_trades:
        running += t.get("net_pnl", 0)
        equity_curve.append({"t": t.get("exit_time"), "v": round(running, 2)})
    # Current equity (with unrealised)
    equity = capital + total_upnl
    equity_curve.append({"t": datetime.now(tz=timezone.utc).isoformat(), "v": round(equity, 2)})

    # ── Performance ──────────────────────────────────────────────────────────
    ret_pct    = (equity - initial) / initial * 100
    wins       = [t for t in trades if t.get("net_pnl", 0) >= 0]
    losses     = [t for t in trades if t.get("net_pnl", 0) <  0]
    win_rate   = len(wins) / len(trades) * 100 if trades else 0
    avg_win    = sum(t["net_pnl"] for t in wins)   / len(wins)   if wins   else 0
    avg_loss   = sum(t["net_pnl"] for t in losses) / len(losses) if losses else 0
    total_fees = sum(t.get("exit_fee", 0) + t.get("entry_fee", 0) for t in trades)
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    # ── Annualised return & Sharpe ────────────────────────────────────────────
    # Prefer per-minute equity snapshots (more data); fall back to closed-trade curve.
    stat_pts = equity_history if len(equity_history) >= 3 else [p for p in equity_curve if p.get("t")]
    ann_return_pct = 0.0
    sharpe         = 0.0
    days_live      = 0.0

    if stat_pts:
        try:
            first_dt  = datetime.fromisoformat(stat_pts[0]["t"])
            now_dt    = datetime.now(tz=timezone.utc)
            days_live = (now_dt - first_dt).total_seconds() / 86400
            if days_live > 0 and initial > 0:
                ann_return_pct = ((equity / initial) ** (365.25 / days_live) - 1) * 100
        except Exception:
            pass

    if len(stat_pts) >= 3 and days_live > 0:
        try:
            vals = [initial] + [p["v"] for p in stat_pts]
            rets = [(vals[i] - vals[i-1]) / vals[i-1]
                    for i in range(1, len(vals)) if vals[i-1] > 0]
            if len(rets) >= 2:
                n        = len(rets)
                mean_r   = sum(rets) / n
                variance = sum((r - mean_r) ** 2 for r in rets) / (n - 1)
                std_r    = math.sqrt(variance)
                if std_r > 0:
                    pts_per_year = n / days_live * 365.25
                    sharpe = mean_r / std_r * math.sqrt(pts_per_year)
        except Exception:
            pass

    # ── Live equity snapshot (written each dashboard poll for live-price candles) ─
    now_iso    = datetime.now(tz=timezone.utc).isoformat()
    live_point = {"t": now_iso, "v": round(equity, 2)}
    _append_equity_snapshot(live_point)

    # Build snapshot list for the client: history + this live point
    # Cap at last 20 000 points (~1 week at 30 s poll cadence) to limit payload
    equity_snapshots = (equity_history + [live_point])[-20_000:]

    # ── Recent trades (last 30, newest first) ────────────────────────────────
    recent_trades = []
    for t in reversed(sorted_trades[-30:]):
        recent_trades.append({
            "asset":      t.get("strategy", "—"),
            "direction":  t.get("direction", "—"),
            "entry":      t.get("entry_price"),
            "exit":       t.get("exit_price"),
            "size":       t.get("size"),
            "pnl":        round(t.get("net_pnl", 0), 2),
            "reason":     t.get("exit_reason", "—"),
            "exit_time":  t.get("exit_time", ""),
        })

    return JSONResponse({
        "ts":            datetime.now(tz=timezone.utc).isoformat(),
        "last_scan":     scores.get("timestamp", "—"),
        "last_scan_iso": scores.get("timestamp_iso"),
        "tick":          scores.get("tick", 0),
        "scan_interval": config.POLL_INTERVAL_SEC,
        "equity":        round(equity, 2),
        "cash":          round(capital, 2),
        "initial":       round(initial, 2),
        "return_pct":    round(ret_pct, 4),
        "total_upnl":    round(total_upnl, 2),
        "positions":     enriched_positions,
        "regime":        scores.get("assets", []),
        "equity_curve":     equity_curve,
        "equity_snapshots": equity_snapshots,
        "trades":           recent_trades,
        "ann_return_pct":   round(ann_return_pct, 2),
        "sharpe":           round(sharpe, 3),
        "days_live":        round(days_live, 1),
        "performance": {
            "total_trades": len(trades),
            "win_rate":     round(win_rate, 1),
            "avg_win":      round(avg_win, 2),
            "avg_loss":     round(avg_loss, 2),
            "expectancy":   round(expectancy, 2),
            "total_fees":   round(total_fees, 2),
        },
    })


# ── Dashboard HTML ────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ethereal Capital — Hyperliquid Bot</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #07080c;
    --surface:   #0d0f18;
    --surface2:  #12141f;
    --border:    rgba(255,255,255,0.06);
    --gold:      #c8a96e;
    --gold-dim:  rgba(200,169,110,0.12);
    --text:      #e8eaf0;
    --muted:     #4a5068;
    --long:      #38bdf8;
    --short:     #c084fc;
    --green:     #22c55e;
    --red:       #f87171;
    --green-dim: rgba(34,197,94,0.10);
    --red-dim:   rgba(248,113,113,0.10);
    --radius:    8px;
    --font:      'Inter', -apple-system, 'Segoe UI', system-ui, sans-serif;
    --mono:      'SF Mono', 'Fira Code', 'Consolas', monospace;
    --serif:     'Cormorant Garamond', Georgia, 'Times New Roman', serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 13px;
    line-height: 1.5;
    min-height: 100vh;
  }

  /* ── Header ── */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 28px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    position: sticky; top: 0; z-index: 100;
  }
  .logo { display: flex; align-items: center; gap: 16px; }
  .logo-mark { display:flex; align-items:center; }
  .logo-mark svg { animation: ouro-glow 4s ease-in-out infinite; }
  @keyframes ouro-glow {
    0%,100% { filter: drop-shadow(0 0 4px rgba(200,169,110,0.20)); }
    50%     { filter: drop-shadow(0 0 11px rgba(200,169,110,0.55)); }
  }
  .logo-divider { width: 1px; height: 32px; background: linear-gradient(to bottom, transparent, var(--gold), transparent); opacity: 0.4; flex-shrink: 0; }
  .logo-text { display: flex; flex-direction: column; gap: 3px; }
  .logo-title { font-family: var(--serif); font-size: 19px; font-style: italic; font-weight: 400; letter-spacing: 0.13em; color: var(--text); }
  .logo-sub { font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted); }
  .header-right { display: flex; align-items: center; gap: 18px; }
  .live-badge { display: flex; align-items: center; gap: 6px; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
  .live-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); box-shadow: 0 0 6px var(--green); animation: pulse 2s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
  #clock { font-family: var(--mono); font-size: 12px; color: var(--muted); min-width: 78px; text-align: right; }

  .btn-refresh {
    display: flex; align-items: center; gap: 6px;
    padding: 5px 12px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: transparent;
    color: var(--muted);
    font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
    cursor: pointer; transition: all 0.15s;
    font-family: var(--font);
  }
  .btn-refresh:hover { border-color: var(--gold); color: var(--gold); }
  .btn-refresh.spinning svg { animation: spin 0.7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Refresh bar ── */
  .refresh-bar { height: 2px; background: var(--gold-dim); }
  .refresh-progress { height: 100%; background: var(--gold); width: 0%; transition: width 1s linear; }

  .stale-warn { display: none; padding: 8px 28px; background: rgba(248,113,113,0.07); border-bottom: 1px solid rgba(248,113,113,0.15); color: var(--red); font-size: 11px; }

  /* ── Layout ── */
  main { padding: 22px 28px; display: flex; flex-direction: column; gap: 18px; max-width: 1600px; margin: 0 auto; }

  /* ── Stats bar ── */
  .stats-bar { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
  .stat { background: var(--surface); padding: 14px 18px; display: flex; flex-direction: column; gap: 3px; }
  .stat-label { font-size: 9px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
  .stat-value { font-size: 20px; font-weight: 300; font-family: var(--font); letter-spacing: -0.01em; }
  .stat-value.pos { color: var(--green); }
  .stat-value.neg { color: var(--red); }
  .stat-value.gold { color: var(--gold); }
  .stat-sub { font-size: 10px; color: var(--muted); font-family: var(--mono); }
  .stat-sub.pos { color: var(--green); }
  .stat-sub.neg { color: var(--red); }

  /* ── Countdown ring ── */
  .countdown-ring-wrap { display: flex; align-items: center; opacity: 0.85; }
  .ring-bg   { stroke: rgba(255,255,255,0.06); }
  .ring-fill { stroke: var(--gold); transition: stroke-dashoffset 0.5s linear; }
  .ring-text { fill: var(--muted); font-family: var(--mono); font-size: 8px; }

  /* ── Grid ── */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  @media(max-width:1100px){ .grid-2{ grid-template-columns:1fr; } }

  /* ── Card ── */
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
  .card-header { display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; border-bottom: 1px solid var(--border); }
  .card-title { font-size: 9px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--gold); font-weight: 600; }
  .card-meta { font-size: 10px; color: var(--muted); font-family: var(--mono); }

  /* ── Tables ── */
  .tbl-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; }
  th { padding: 7px 13px; text-align: right; font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); font-weight: 500; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th:first-child { text-align: left; }
  td { padding: 8px 13px; text-align: right; font-family: var(--mono); font-size: 12px; border-bottom: 1px solid rgba(255,255,255,0.025); white-space: nowrap; }
  td:first-child { text-align: left; font-weight: 600; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--surface2); }

  .dir-long  { color: var(--long);  font-weight: 600; }
  .dir-short { color: var(--short); font-weight: 600; }
  .dir-flat  { color: var(--muted); }
  .pos-pnl { color: var(--green); }
  .neg-pnl { color: var(--red); }

  .pos-badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 9px; letter-spacing: 0.06em; font-weight: 600; }
  .pos-badge.open { background: var(--green-dim); color: var(--green); }

  .tier-badge { display: inline-block; width: 14px; height: 14px; border-radius: 2px; font-size: 8px; font-weight: 700; text-align: center; line-height: 14px; }
  .tier-1 { background: var(--gold); color: #07080c; }
  .tier-2 { background: #7a8799; color: #07080c; }
  .tier-3 { background: #7a5c3e; color: #e8c9a8; }

  .score-bar-wrap { display: flex; align-items: center; width: 90px; }
  .score-bar-track { flex: 1; height: 3px; background: var(--surface2); border-radius: 2px; overflow: hidden; }
  .score-bar-fill { height: 100%; border-radius: 2px; }
  .fill-long  { background: var(--long);  }
  .fill-short { background: var(--short); }

  .reason-pill { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 9px; letter-spacing: 0.05em; }
  .reason-stop_loss      { background: rgba(248,113,113,0.12); color: var(--red); }
  .reason-take_profit    { background: rgba(34,197,94,0.10); color: var(--green); }
  .reason-regime_exit    { background: rgba(200,169,110,0.12); color: var(--gold); }
  .reason-regime_flip    { background: rgba(192,132,252,0.12); color: var(--short); }
  .reason-hard_stop      { background: rgba(248,113,113,0.18); color: var(--red); }
  .reason-resize         { background: rgba(255,255,255,0.05); color: var(--muted); }
  .reason-regime_flat    { background: rgba(255,255,255,0.05); color: var(--muted); }

  /* ── Timeframe toggle ── */
  .tf-btn { padding: 2px 9px; border: 1px solid var(--border); border-radius: 3px; background: transparent; color: var(--muted); font-size: 9px; letter-spacing: 0.09em; text-transform: uppercase; cursor: pointer; font-family: var(--font); transition: all 0.15s; }
  .tf-btn:hover { border-color: var(--gold); color: var(--gold); }
  .tf-btn.active { border-color: var(--gold); color: var(--gold); background: var(--gold-dim); }

  /* ── Equity curve ── */
  .chart-wrap { padding: 4px 0 0; position: relative; }
  #equity-chart { width: 100%; height: 260px; }
  #equity-tooltip {
    position: absolute;
    top: 14px; left: 14px;
    background: rgba(13,15,24,0.92);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
    pointer-events: none;
    display: none;
    z-index: 10;
    min-width: 140px;
  }
  .tt-label { font-size: 9px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin-bottom: 3px; }
  .tt-value { font-size: 16px; font-weight: 300; font-family: var(--font); color: var(--text); }
  .tt-change { font-size: 11px; font-family: var(--mono); margin-top: 2px; }
  .tt-time   { font-size: 10px; color: var(--muted); font-family: var(--mono); margin-top: 4px; }

  /* ── Footer ── */
  footer { text-align: center; padding: 24px; border-top: 1px solid var(--border); margin-top: 4px; }
  footer .footer-mark { font-family: var(--serif); font-size: 18px; color: var(--gold); font-weight: 300; }
  footer .footer-tagline { font-family: var(--serif); font-size: 13px; font-style: italic; font-weight: 300; color: var(--muted); letter-spacing: 0.04em; margin-top: 4px; }

  .empty-row td { text-align: center !important; color: var(--muted); padding: 28px !important; font-weight: 400 !important; }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-mark">
      <!-- Ouroboros infinity — snake eating its own tail in a figure-8 -->
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 104 52" width="56" height="28" aria-label="Ouroboros" role="img">
        <defs>
          <!-- Top-to-bottom gradient gives the body a rounded 3-D feel -->
          <linearGradient id="sg" x1="0" y1="0" x2="0" y2="52" gradientUnits="userSpaceOnUse">
            <stop offset="0%"   stop-color="#e8cc88"/>
            <stop offset="45%"  stop-color="#c8a96e"/>
            <stop offset="100%" stop-color="#6e460e"/>
          </linearGradient>
        </defs>

        <!-- ── LEFT LOBE (tail / under strand — drawn first) ─────────── -->
        <!-- Dark border -->
        <path d="M 52,26 C 52,46 34,51 21,51 C 5,51 1,39 1,26 C 1,13 5,1 21,1 C 34,1 52,6 52,26"
              fill="none" stroke="#1e1004" stroke-width="11" stroke-linecap="round"/>
        <!-- Gold body -->
        <path d="M 52,26 C 52,46 34,51 21,51 C 6,51 1,39 1,26 C 1,13 6,1 21,1 C 34,1 52,6 52,26"
              fill="none" stroke="url(#sg)" stroke-width="7.5" stroke-linecap="round"/>
        <!-- Scale shimmer — subtle dashed lighter overlay -->
        <path d="M 52,26 C 52,46 34,51 21,51 C 6,51 1,39 1,26 C 1,13 6,1 21,1 C 34,1 52,6 52,26"
              fill="none" stroke="rgba(255,248,210,0.20)" stroke-width="7" stroke-linecap="round"
              stroke-dasharray="2.5 5.5"/>
        <!-- Spine highlight -->
        <path d="M 52,26 C 52,46 34,51 21,51 C 6,51 1,39 1,26 C 1,13 6,1 21,1 C 34,1 52,6 52,26"
              fill="none" stroke="rgba(255,248,210,0.16)" stroke-width="1.8" stroke-linecap="round"/>

        <!-- ── RIGHT LOBE (neck / over strand — drawn second) ────────── -->
        <!-- Dark border -->
        <path d="M 52,26 C 52,6 70,1 83,1 C 99,1 103,13 103,26 C 103,39 99,51 83,51 C 70,51 52,46 52,26"
              fill="none" stroke="#1e1004" stroke-width="11" stroke-linecap="round"/>
        <!-- Gold body -->
        <path d="M 52,26 C 52,6 70,1 83,1 C 99,1 103,13 103,26 C 103,39 99,51 83,51 C 70,51 52,46 52,26"
              fill="none" stroke="url(#sg)" stroke-width="7.5" stroke-linecap="round"/>
        <!-- Scale shimmer -->
        <path d="M 52,26 C 52,6 70,1 83,1 C 99,1 103,13 103,26 C 103,39 99,51 83,51 C 70,51 52,46 52,26"
              fill="none" stroke="rgba(255,248,210,0.20)" stroke-width="7" stroke-linecap="round"
              stroke-dasharray="2.5 5.5"/>
        <!-- Spine highlight -->
        <path d="M 52,26 C 52,6 70,1 83,1 C 99,1 103,13 103,26 C 103,39 99,51 83,51 C 70,51 52,46 52,26"
              fill="none" stroke="rgba(255,248,210,0.16)" stroke-width="1.8" stroke-linecap="round"/>

        <!-- ── SNAKE HEAD — at centre, mouth open left (biting tail) ─── -->
        <!-- Head shadow/border -->
        <ellipse cx="53" cy="26" rx="11" ry="7.5" fill="#1e1004"/>
        <!-- Head base -->
        <ellipse cx="53" cy="26" rx="10"  ry="6.5" fill="#9a7232"/>
        <!-- Dorsal highlight (lighter top of skull) -->
        <ellipse cx="52.5" cy="23.5" rx="7" ry="3.8" fill="#c8a96e" opacity="0.55"/>
        <!-- Snout (slightly darker, projects left toward tail) -->
        <ellipse cx="45" cy="26" rx="5.5" ry="4.2" fill="#7a5020"/>
        <!-- Jaw gap line — mouth open to consume tail -->
        <path d="M 47,23.2 C 44.5,24.2 41,25.5 41,26 C 41,26.5 44.5,27.8 47,28.8"
              fill="none" stroke="#0f0602" stroke-width="1.4" stroke-linecap="round"/>

        <!-- Eye — vertical slit pupil, snake-style -->
        <circle cx="57" cy="22" r="3.2" fill="#0c0600"/>
        <ellipse cx="57" cy="22" rx="1.4" ry="2.8" fill="#2a6018"/>
        <ellipse cx="57" cy="22" rx="0.45" ry="2.5" fill="#040200"/>
        <circle  cx="57.8" cy="20.7" r="0.75" fill="rgba(255,255,255,0.40)"/>

        <!-- Tongue — forked, deep red -->
        <line x1="42"  y1="26"   x2="37.5" y2="26"   stroke="#8a1010" stroke-width="1.3" stroke-linecap="round"/>
        <line x1="37.5" y1="26"  x2="34.5" y2="23"   stroke="#8a1010" stroke-width="1.1" stroke-linecap="round"/>
        <line x1="37.5" y1="26"  x2="34.5" y2="29"   stroke="#8a1010" stroke-width="1.1" stroke-linecap="round"/>
      </svg>
    </div>
    <div class="logo-divider"></div>
    <div class="logo-text">
      <div class="logo-title">Ethereal Capital</div>
      <div class="logo-sub">Hyperliquid Bot</div>
    </div>
  </div>
  <div class="header-right">
    <div id="last-scan-label" style="font-size:10px;color:var(--muted)">—</div>
    <div class="countdown-ring-wrap" title="Next regime scan">
      <svg id="countdown-ring" viewBox="0 0 36 36" width="36" height="36">
        <circle class="ring-bg"   cx="18" cy="18" r="14" fill="none" stroke-width="2.5"/>
        <circle class="ring-fill" cx="18" cy="18" r="14" fill="none" stroke-width="2.5"
                stroke-dasharray="87.96" stroke-dashoffset="87.96"
                transform="rotate(-90 18 18)" id="ring-arc"/>
        <text class="ring-text" x="18" y="21" text-anchor="middle" id="ring-text">—</text>
      </svg>
    </div>
    <button class="btn-refresh" id="btn-refresh" onclick="manualRefresh()">
      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
      Refresh
    </button>
    <div class="live-badge">
      <div class="live-dot" id="live-dot"></div>
      <span>Paper</span>
    </div>
    <div id="clock">—</div>
  </div>
</header>

<div class="refresh-bar"><div class="refresh-progress" id="refresh-bar"></div></div>
<div class="stale-warn" id="stale-warn">⚠ Data may be stale — last update failed</div>

<main>

  <!-- Stats bar -->
  <div class="stats-bar">
    <div class="stat">
      <div class="stat-label">Equity</div>
      <div class="stat-value gold" id="s-equity">—</div>
      <div class="stat-sub" id="s-initial">—</div>
    </div>
    <div class="stat">
      <div class="stat-label">Total Return</div>
      <div class="stat-value" id="s-return">—</div>
      <div class="stat-sub" id="s-upnl">—</div>
    </div>
    <div class="stat">
      <div class="stat-label">Positions</div>
      <div class="stat-value" id="s-positions">—</div>
      <div class="stat-sub" id="s-notional">—</div>
    </div>
    <div class="stat">
      <div class="stat-label">Win Rate</div>
      <div class="stat-value" id="s-winrate">—</div>
      <div class="stat-sub" id="s-trades">—</div>
    </div>
    <div class="stat">
      <div class="stat-label">Avg Trade PnL</div>
      <div class="stat-value" id="s-expectancy">—</div>
      <div class="stat-sub" id="s-avgwinloss">—</div>
    </div>
    <div class="stat">
      <div class="stat-label">Ann. Return</div>
      <div class="stat-value" id="s-annret">—</div>
      <div class="stat-sub" id="s-dayslive">—</div>
    </div>
    <div class="stat">
      <div class="stat-label">Sharpe Ratio</div>
      <div class="stat-value" id="s-sharpe">—</div>
      <div class="stat-sub" id="s-sharpe-sub">annualised</div>
    </div>
    <div class="stat">
      <div class="stat-label">Next Scan</div>
      <div class="stat-value gold" id="s-countdown">—</div>
      <div class="stat-sub" id="s-tick">—</div>
    </div>
  </div>

  <!-- Equity curve -->
  <div class="card">
    <div class="card-header">
      <div class="card-title">Equity Curve</div>
      <div style="display:flex;align-items:center;gap:10px">
        <div style="display:flex;gap:4px">
          <button class="tf-btn active" data-mins="5"    onclick="setCandle(5)">5M</button>
          <button class="tf-btn"        data-mins="60"   onclick="setCandle(60)">1H</button>
          <button class="tf-btn"        data-mins="1440" onclick="setCandle(1440)">1D</button>
        </div>
        <div class="card-meta" id="curve-meta">—</div>
      </div>
    </div>
    <div class="chart-wrap">
      <div id="equity-chart"></div>
      <div id="equity-tooltip">
        <div class="tt-label">Equity</div>
        <div class="tt-value" id="tt-value">—</div>
        <div class="tt-change" id="tt-change">—</div>
        <div class="tt-time" id="tt-time">—</div>
      </div>
    </div>
  </div>

  <!-- Positions + Regime -->
  <div class="grid-2">

    <div class="card">
      <div class="card-header">
        <div class="card-title">Open Positions</div>
        <div class="card-meta" id="pos-count">—</div>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead><tr>
            <th>Asset</th><th>Dir</th><th>Score</th>
            <th>Entry</th><th>Now</th><th>Notional</th>
            <th>Unreal PnL</th><th>Stop</th><th>Age</th>
          </tr></thead>
          <tbody id="positions-body"><tr class="empty-row"><td colspan="9">Loading…</td></tr></tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-title">Regime Scores</div>
        <div class="card-meta" id="regime-meta">—</div>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead><tr>
            <th>Asset</th><th>T</th><th>Price</th><th>Score</th>
            <th>Conv</th><th>Dir</th><th style="text-align:left;padding-left:8px">Bar</th><th>Fund/hr</th>
          </tr></thead>
          <tbody id="regime-body"><tr class="empty-row"><td colspan="8">Loading…</td></tr></tbody>
        </table>
      </div>
    </div>

  </div>

  <!-- Closed trades -->
  <div class="card">
    <div class="card-header">
      <div class="card-title">Closed Trades</div>
      <div class="card-meta" id="trades-meta">—</div>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Asset</th><th>Dir</th><th>Entry</th><th>Exit</th>
          <th>Size</th><th>Net PnL</th><th>Reason</th><th>Closed</th>
        </tr></thead>
        <tbody id="trades-body"><tr class="empty-row"><td colspan="8">Loading…</td></tr></tbody>
      </table>
    </div>
  </div>

</main>

<footer>
  <div class="footer-mark">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 104 52" width="28" height="14" aria-hidden="true" style="opacity:0.35">
      <defs>
        <linearGradient id="sfg" x1="0" y1="0" x2="0" y2="52" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stop-color="#e8cc88"/><stop offset="100%" stop-color="#6e460e"/>
        </linearGradient>
      </defs>
      <path d="M 52,26 C 52,46 34,51 21,51 C 5,51 1,39 1,26 C 1,13 5,1 21,1 C 34,1 52,6 52,26"
            fill="none" stroke="#1e1004" stroke-width="11" stroke-linecap="round"/>
      <path d="M 52,26 C 52,46 34,51 21,51 C 6,51 1,39 1,26 C 1,13 6,1 21,1 C 34,1 52,6 52,26"
            fill="none" stroke="url(#sfg)" stroke-width="7.5" stroke-linecap="round"/>
      <path d="M 52,26 C 52,6 70,1 83,1 C 99,1 103,13 103,26 C 103,39 99,51 83,51 C 70,51 52,46 52,26"
            fill="none" stroke="#1e1004" stroke-width="11" stroke-linecap="round"/>
      <path d="M 52,26 C 52,6 70,1 83,1 C 99,1 103,13 103,26 C 103,39 99,51 83,51 C 70,51 52,46 52,26"
            fill="none" stroke="url(#sfg)" stroke-width="7.5" stroke-linecap="round"/>
      <ellipse cx="53" cy="26" rx="10" ry="6.5" fill="#7a5020"/>
      <ellipse cx="57" cy="22" rx="1.4" ry="2.8" fill="#2a6018"/>
      <ellipse cx="57" cy="22" rx="0.45" ry="2.5" fill="#040200"/>
    </svg>
  </div>
</footer>

<script>
const REFRESH_MS    = 30000;
let   _scanInterval = 60;
let   _lastScanIso  = null;
let   _progressIv   = null;
let   _countdownIv  = null;

// ── Formatters ────────────────────────────────────────────────────────────────
const fmt = {
  price: v => v == null ? '—' : v < 10 ? `$${v.toFixed(4)}` : `$${v.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}`,
  pct:   v => v == null ? '—' : `${v>=0?'+':''}${v.toFixed(2)}%`,
  pnl:   v => v == null ? '—' : `${v>=0?'+':'-'}$${Math.abs(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}`,
  size:  v => v == null ? '—' : v >= 100 ? v.toLocaleString('en-US',{maximumFractionDigits:0}) : v.toFixed(4),
  score: v => v != null ? `${v>=0?'+':''}${v.toFixed(1)}` : '—',
  age:   s => {
    if (!s) return '—';
    const secs = Math.floor((Date.now() - new Date(s)) / 1000);
    const h = Math.floor(secs/3600), m = Math.floor((secs%3600)/60);
    return `${h}h ${String(m).padStart(2,'0')}m`;
  },
  ts: s => {
    if (!s) return '—';
    const d = new Date(s);
    return d.toLocaleString('en-US',{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'America/Chicago'});
  },
};

const pnlClass = v => v == null ? '' : v >= 0 ? 'pos-pnl' : 'neg-pnl';
const dirClass = d => d==='long'?'dir-long':d==='short'?'dir-short':'dir-flat';
const set = (id, v) => { const e = document.getElementById(id); if(e) e.textContent = v; };
const setClass = (id, cls) => { const e = document.getElementById(id); if(e) e.className = cls; };

function scoreBar(score, dir) {
  if (score == null) return '—';
  const pct = Math.min(Math.abs(score), 100);
  const cls = dir === 'long' ? 'fill-long' : dir === 'short' ? 'fill-short' : 'fill-long';
  return `<div class="score-bar-wrap"><div class="score-bar-track"><div class="score-bar-fill ${cls}" style="width:${pct}%"></div></div></div>`;
}

// ── Equity curve (TradingView Lightweight Charts) ─────────────────────────────
let _equityChart   = null;
let _candleInterval = 5;   // minutes; user-selectable
let _lastD          = null; // cached API response for re-renders on toggle

function buildOHLC(points, intervalMinutes) {
  const bucketSec = intervalMinutes * 60;
  const candles   = {};
  for (const pt of points) {
    try {
      const ts     = new Date(pt.t).getTime() / 1000;
      const bucket = Math.floor(ts / bucketSec) * bucketSec;
      const v      = pt.v;
      if (!candles[bucket]) {
        candles[bucket] = { time: bucket, open: v, high: v, low: v, close: v };
      } else {
        const c = candles[bucket];
        if (v > c.high) c.high = v;
        if (v < c.low)  c.low  = v;
        c.close = v;
      }
    } catch(e) {}
  }
  const sorted = Object.values(candles).sort((a, b) => a.time - b.time);
  // Pin each candle's open to the previous candle's close — no gaps
  for (let i = 1; i < sorted.length; i++) {
    sorted[i].open = sorted[i - 1].close;
    sorted[i].high = Math.max(sorted[i].high, sorted[i].open);
    sorted[i].low  = Math.min(sorted[i].low,  sorted[i].open);
  }
  return sorted;
}

function setCandle(mins) {
  _candleInterval = mins;
  document.querySelectorAll('.tf-btn').forEach(b =>
    b.classList.toggle('active', parseInt(b.dataset.mins) === mins)
  );
  if (_lastD) renderEquityCurve(_lastD);
}

function renderEquityCurve(d) {
  _lastD = d;
  const container = document.getElementById('equity-chart');
  const tooltip   = document.getElementById('equity-tooltip');
  const initial   = d.initial;
  const snapshots = d.equity_snapshots || [];

  const ohlc = buildOHLC(snapshots, _candleInterval);

  if (ohlc.length < 2) {
    if (_equityChart) { _equityChart.remove(); _equityChart = null; }
    container.innerHTML = '<div style="height:260px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px">Waiting for data… (need 2 candles)</div>';
    document.getElementById('curve-meta').textContent = '—';
    return;
  }

  if (_equityChart) { _equityChart.remove(); _equityChart = null; }
  container.innerHTML = '';

  const chart = LightweightCharts.createChart(container, {
    width:  container.clientWidth,
    height: 260,
    layout: {
      background: { color: '#0d0f18' },
      textColor:  '#4a5068',
      fontFamily: "'SF Mono','Fira Code','Consolas',monospace",
      fontSize:   11,
    },
    grid: {
      vertLines: { color: 'rgba(255,255,255,0.03)' },
      horzLines: { color: 'rgba(255,255,255,0.03)' },
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
      vertLine: { color: 'rgba(200,169,110,0.5)', width: 1, style: LightweightCharts.LineStyle.Solid, labelBackgroundColor: '#1a1c28' },
      horzLine: { color: 'rgba(200,169,110,0.5)', width: 1, style: LightweightCharts.LineStyle.Solid, labelBackgroundColor: '#1a1c28' },
    },
    rightPriceScale: { borderColor: 'rgba(255,255,255,0.06)' },
    timeScale: {
      borderColor:    'rgba(255,255,255,0.06)',
      timeVisible:    true,
      secondsVisible: false,
    },
    localization: {
      priceFormatter: v => `$${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
    },
    handleScroll: true,
    handleScale:  true,
  });

  const series = chart.addCandlestickSeries({
    upColor:          '#22c55e',
    downColor:        '#f87171',
    borderUpColor:    '#22c55e',
    borderDownColor:  '#f87171',
    wickUpColor:      'rgba(34,197,94,0.65)',
    wickDownColor:    'rgba(248,113,113,0.65)',
    priceLineVisible: false,
    lastValueVisible: true,
  });
  series.setData(ohlc);

  series.createPriceLine({
    price: initial, color: 'rgba(200,169,110,0.5)',
    lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed,
    axisLabelVisible: true, title: 'start',
  });

  const lastClose   = ohlc[ohlc.length - 1].close;
  const total       = lastClose - initial;
  const lbl         = _candleInterval < 60 ? `${_candleInterval}m` : _candleInterval < 1440 ? `${_candleInterval/60}h` : '1d';
  document.getElementById('curve-meta').textContent =
    `${ohlc.length} ${lbl} candles · ${fmt.pnl(total)} total`;

  // Show ~35 candles; user can still scroll/zoom freely
  const TARGET_BARS = 70;
  const barSpacing  = Math.max(3, Math.floor((container.clientWidth - 60) / TARGET_BARS));
  chart.timeScale().applyOptions({ barSpacing, minBarSpacing: 4 });
  chart.timeScale().scrollToRealTime();
  _equityChart = chart;

  new ResizeObserver(() => {
    const bs = Math.max(3, Math.floor((container.clientWidth - 60) / TARGET_BARS));
    chart.applyOptions({ width: container.clientWidth });
    chart.timeScale().applyOptions({ barSpacing: bs });
  }).observe(container);

  // Crosshair tooltip
  chart.subscribeCrosshairMove(param => {
    if (!param.time || !param.seriesData || !param.seriesData.has(series)) {
      tooltip.style.display = 'none'; return;
    }
    const c         = param.seriesData.get(series);
    const value     = c.close;
    const change    = value - initial;
    const changePct = (change / initial) * 100;
    const candleChg = c.close - c.open;
    const ts        = new Date(param.time * 1000);
    const timeStr   = ts.toLocaleString('en-US', { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit', hour12:false, timeZone:'America/Chicago' });
    const cc        = change >= 0 ? '#22c55e' : '#f87171';
    const sign      = change >= 0 ? '+' : '';
    const ccc       = candleChg >= 0 ? '#22c55e' : '#f87171';
    const csign     = candleChg >= 0 ? '+' : '';
    document.getElementById('tt-value').textContent = `$${value.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}`;
    document.getElementById('tt-change').innerHTML  =
      `<span style="color:${cc}">${sign}$${Math.abs(change).toFixed(2)} (${sign}${changePct.toFixed(2)}%)</span>` +
      `<br><span style="color:${ccc};font-size:10px">candle ${csign}$${Math.abs(candleChg).toFixed(2)}</span>`;
    document.getElementById('tt-time').textContent = timeStr;
    const x = param.point.x;
    tooltip.style.left    = x > container.clientWidth / 2 ? 'auto' : '14px';
    tooltip.style.right   = x > container.clientWidth / 2 ? '14px' : 'auto';
    tooltip.style.display = 'block';
  });
  container.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
}

// ── Render helpers ────────────────────────────────────────────────────────────
function renderStats(d) {
  document.title = `$${d.equity.toLocaleString('en-US',{minimumFractionDigits:0,maximumFractionDigits:0})}  ·  ∞`;

  set('s-equity', `$${d.equity.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}`);
  set('s-initial', `started $${d.initial.toLocaleString('en-US',{minimumFractionDigits:0})}`);

  const retEl = document.getElementById('s-return');
  retEl.textContent = fmt.pct(d.return_pct);
  retEl.className = 'stat-value ' + (d.return_pct >= 0 ? 'pos' : 'neg');

  const upEl = document.getElementById('s-upnl');
  upEl.textContent = `${fmt.pnl(d.total_upnl)} unreal`;
  upEl.className = 'stat-sub ' + (d.total_upnl >= 0 ? 'pos' : 'neg');

  set('s-positions', d.positions.length);
  const notional = d.positions.reduce((s,p)=>s+(p.notional||0),0);
  set('s-notional', `$${notional.toLocaleString('en-US',{maximumFractionDigits:0})} notional`);

  const wrEl = document.getElementById('s-winrate');
  wrEl.textContent = `${d.performance.win_rate.toFixed(0)}%`;
  wrEl.className = 'stat-value ' + (d.performance.win_rate >= 50 ? 'pos' : 'neg');
  set('s-trades', `${d.performance.total_trades} closed trades`);

  const expEl = document.getElementById('s-expectancy');
  expEl.textContent = fmt.pnl(d.performance.expectancy);
  expEl.className = 'stat-value ' + (d.performance.expectancy >= 0 ? 'pos' : 'neg');
  const wSign = d.performance.avg_win  >= 0 ? '+' : '';
  const lSign = d.performance.avg_loss >= 0 ? '+' : '';
  const avgW  = `$${Math.abs(d.performance.avg_win).toFixed(2)}`;
  const avgL  = `$${Math.abs(d.performance.avg_loss).toFixed(2)}`;
  set('s-avgwinloss', d.performance.total_trades ? `avg W: +${avgW}  ·  avg L: -${avgL}` : 'no trades yet');

  // Annualised return
  const annEl = document.getElementById('s-annret');
  if (d.days_live > 0 && d.ann_return_pct !== 0) {
    annEl.textContent = fmt.pct(d.ann_return_pct);
    annEl.className = 'stat-value ' + (d.ann_return_pct >= 0 ? 'pos' : 'neg');
  } else {
    annEl.textContent = '—';
    annEl.className = 'stat-value';
  }
  set('s-dayslive', d.days_live > 0 ? `${d.days_live.toFixed(1)}d live` : 'no data yet');

  // Sharpe ratio
  const shEl = document.getElementById('s-sharpe');
  if (d.sharpe !== 0) {
    shEl.textContent = d.sharpe.toFixed(2);
    shEl.className = 'stat-value ' + (d.sharpe >= 1 ? 'pos' : d.sharpe >= 0 ? 'gold' : 'neg');
  } else {
    shEl.textContent = '—';
    shEl.className = 'stat-value';
  }
  set('s-sharpe-sub', d.sharpe !== 0 ? (d.sharpe >= 2 ? 'excellent' : d.sharpe >= 1 ? 'good' : d.sharpe >= 0 ? 'building' : 'negative') : 'need more trades');

  set('s-tick', `tick #${d.tick}`);

  _scanInterval = d.scan_interval || 60;
  _lastScanIso  = d.last_scan_iso;
}

function renderPositions(positions) {
  set('pos-count', `${positions.length} open`);
  const tbody = document.getElementById('positions-body');
  if (!positions.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="9">No open positions</td></tr>';
    return;
  }
  tbody.innerHTML = positions.map(p => {
    const dc = dirClass(p.direction), uc = pnlClass(p.upnl), sc = pnlClass(p.score);
    return `<tr>
      <td>${p.name}</td>
      <td class="${dc}">${p.direction.toUpperCase()}</td>
      <td class="${sc}">${fmt.score(p.score)}</td>
      <td>${fmt.price(p.entry)}</td>
      <td>${fmt.price(p.live)}</td>
      <td>$${(p.notional||0).toLocaleString('en-US',{maximumFractionDigits:0})}</td>
      <td class="${uc}">${fmt.pnl(p.upnl)}</td>
      <td>${fmt.price(p.stop)}</td>
      <td style="color:var(--muted)">${fmt.age(p.entry_time)}</td>
    </tr>`;
  }).join('');
}

function renderRegime(assets, openNames) {
  set('regime-meta', `${assets.length} assets · ${assets.filter(a=>a.direction!=='flat').length} active`);
  const tbody = document.getElementById('regime-body');
  if (!assets.length) { tbody.innerHTML = '<tr class="empty-row"><td colspan="8">Waiting for scan…</td></tr>'; return; }
  let html = '', divider = false;
  assets.forEach((a, i) => {
    if (i === 10 && !divider) { html += `<tr><td colspan="8" style="padding:0;background:var(--border);height:1px"></td></tr>`; divider = true; }
    const dc = dirClass(a.direction), sc = pnlClass(a.score);
    const isOpen = openNames.has(a.name);
    const fr = a.funding_rate;
    const frStr = Math.abs(fr) < 0.00001 ? '<span style="color:var(--muted)">—</span>'
      : fr > 0 ? `<span style="color:#facc15">+${(fr*100).toFixed(3)}%</span>`
               : `<span style="color:var(--long)">${(fr*100).toFixed(3)}%</span>`;
    html += `<tr>
      <td>${a.name}${isOpen?` <span class="pos-badge open">OPEN</span>`:''}</td>
      <td style="text-align:center"><span class="tier-badge tier-${a.tier}">${a.tier}</span></td>
      <td>${fmt.price(a.price)}</td>
      <td class="${sc}">${fmt.score(a.score)}</td>
      <td style="color:var(--muted)">${(a.conviction*100).toFixed(0)}%</td>
      <td class="${dc}">${a.direction.toUpperCase()}</td>
      <td style="text-align:left;padding-left:8px">${scoreBar(a.score, a.direction)}</td>
      <td>${frStr}</td>
    </tr>`;
  });
  tbody.innerHTML = html;
}

function renderTrades(trades) {
  set('trades-meta', `last ${trades.length} closed`);
  const tbody = document.getElementById('trades-body');
  if (!trades.length) { tbody.innerHTML = '<tr class="empty-row"><td colspan="8">No closed trades yet</td></tr>'; return; }
  tbody.innerHTML = trades.map(t => {
    const dc = dirClass(t.direction), pc = pnlClass(t.pnl);
    const reasonCls = `reason-pill reason-${t.reason}`;
    return `<tr>
      <td>${t.asset}</td>
      <td class="${dc}">${t.direction.toUpperCase()}</td>
      <td>${fmt.price(t.entry)}</td>
      <td>${fmt.price(t.exit)}</td>
      <td style="color:var(--muted)">${fmt.size(t.size)}</td>
      <td class="${pc}">${fmt.pnl(t.pnl)}</td>
      <td><span class="${reasonCls}">${t.reason}</span></td>
      <td style="color:var(--muted)">${fmt.ts(t.exit_time)}</td>
    </tr>`;
  }).join('');
}

// ── Countdown ─────────────────────────────────────────────────────────────────
const RING_CIRC = 87.96; // 2π × r=14
function startCountdown() {
  clearInterval(_countdownIv);
  _countdownIv = setInterval(() => {
    const arc = document.getElementById('ring-arc');
    const ringTxt = document.getElementById('ring-text');
    if (!_lastScanIso || !_scanInterval) {
      set('s-countdown', '—');
      if (arc) arc.style.strokeDashoffset = RING_CIRC;
      if (ringTxt) ringTxt.textContent = '—';
      return;
    }
    const elapsed   = (Date.now() - new Date(_lastScanIso)) / 1000;
    const remaining = Math.max(0, _scanInterval - elapsed);
    const pct       = remaining / _scanInterval;

    // Stats bar — M:SS
    const m = Math.floor(remaining / 60);
    const s = Math.floor(remaining % 60);
    set('s-countdown', `${m}:${String(s).padStart(2, '0')}`);

    // SVG ring — depletes as scan approaches; fill when fresh
    if (arc) arc.style.strokeDashoffset = RING_CIRC * (1 - pct);
    if (ringTxt) ringTxt.textContent = remaining > 59
      ? `${Math.ceil(remaining / 60)}m`
      : `${Math.ceil(remaining)}`;
  }, 500);
}

// ── Progress bar ──────────────────────────────────────────────────────────────
function startProgressBar() {
  clearInterval(_progressIv);
  const bar = document.getElementById('refresh-bar');
  let pct = 0; bar.style.width = '0%';
  _progressIv = setInterval(() => {
    pct += 100 / (REFRESH_MS / 1000);
    bar.style.width = Math.min(pct, 100) + '%';
    if (pct >= 100) clearInterval(_progressIv);
  }, 1000);
}

// ── Fetch & render ────────────────────────────────────────────────────────────
async function refresh(manual = false) {
  const btn = document.getElementById('btn-refresh');
  btn.classList.add('spinning');
  try {
    const r = await fetch('/api/status');
    if (!r.ok) throw new Error(r.statusText);
    const d = await r.json();

    renderStats(d);
    renderPositions(d.positions);
    renderRegime(d.regime, new Set(d.positions.map(p => p.name)));
    renderTrades(d.trades);
    renderEquityCurve(d);

    set('last-scan-label', `updated ${new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'America/Chicago'})} CT`);
    document.getElementById('stale-warn').style.display = 'none';
    document.getElementById('live-dot').style.cssText = 'background:var(--green);box-shadow:0 0 6px var(--green)';
    startCountdown();
    startProgressBar();
  } catch(e) {
    document.getElementById('stale-warn').style.display = 'block';
    document.getElementById('live-dot').style.cssText = 'background:var(--red)';
    console.error('Refresh failed:', e);
  }
  btn.classList.remove('spinning');
}

function manualRefresh() { refresh(true); }

// ── Clock ─────────────────────────────────────────────────────────────────────
setInterval(() => {
  document.getElementById('clock').textContent =
    new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit',timeZone:'America/Chicago'});
}, 1000);

// ── Boot ──────────────────────────────────────────────────────────────────────
refresh();
setInterval(refresh, REFRESH_MS);
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTML
