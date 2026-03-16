"""
Ethereal Capital — Hyperliquid Bot Dashboard
Run: uvicorn dashboard:app --host 0.0.0.0 --port 8080
"""

import json
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
    wallet = _load_json(config.REGIME_STATE_FILE)
    scores = _load_json(config.SCORES_FILE)
    prices = _live_prices()

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
        "equity_curve":  equity_curve,
        "trades":        recent_trades,
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
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&display=swap" rel="stylesheet">
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
    --font:      -apple-system, 'Segoe UI', system-ui, sans-serif;
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
  .logo { display: flex; align-items: center; gap: 14px; }
  .logo-mark { font-family: var(--serif); font-size: 28px; color: var(--gold); font-weight: 300; letter-spacing: -1px; line-height: 1; }
  .logo-text { display: flex; flex-direction: column; gap: 2px; }
  .logo-title { font-family: var(--serif); font-size: 18px; font-weight: 500; letter-spacing: 0.08em; color: var(--text); }
  .logo-sub { font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted); }
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
  .stat-value { font-size: 22px; font-weight: 400; font-family: var(--serif); letter-spacing: 0.01em; }
  .stat-value.pos { color: var(--green); }
  .stat-value.neg { color: var(--red); }
  .stat-value.gold { color: var(--gold); }
  .stat-sub { font-size: 10px; color: var(--muted); font-family: var(--mono); }
  .stat-sub.pos { color: var(--green); }
  .stat-sub.neg { color: var(--red); }

  /* ── Countdown ── */
  #countdown-ring {
    width: 36px; height: 36px; flex-shrink: 0;
  }
  .countdown-wrap { display: flex; align-items: center; gap: 10px; }
  .countdown-label { display: flex; flex-direction: column; gap: 2px; }

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

  .tier-badge { display: inline-block; width: 14px; height: 14px; border-radius: 2px; font-size: 8px; font-weight: 700; text-align: center; line-height: 14px; color: var(--bg); background: var(--muted); opacity: 0.6; }
  .tier-1 { background: var(--gold); opacity: 1; }

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

  /* ── Equity curve ── */
  .chart-wrap { padding: 4px 0 0; position: relative; }
  #equity-chart { width: 100%; height: 220px; }
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
  .tt-value { font-size: 20px; font-weight: 400; font-family: var(--serif); color: var(--text); letter-spacing: 0.01em; }
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
    <div class="logo-mark">∞</div>
    <div class="logo-text">
      <div class="logo-title">Ethereal Capital</div>
      <div class="logo-sub">Hyperliquid Bot</div>
    </div>
  </div>
  <div class="header-right">
    <div id="last-scan-label" style="font-size:10px;color:var(--muted)">—</div>
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
      <div class="stat-label">Expectancy</div>
      <div class="stat-value" id="s-expectancy">—</div>
      <div class="stat-sub" id="s-avgwinloss">—</div>
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
      <div class="card-meta" id="curve-meta">—</div>
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
  <div class="footer-mark">∞</div>
  <div class="footer-tagline">Enduring Wealth. Enduring Legacy.</div>
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
    return d.toLocaleString('en-US',{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',hour12:false});
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
let _equityChart = null;

function renderEquityCurve(points, initial) {
  const container = document.getElementById('equity-chart');
  const tooltip   = document.getElementById('equity-tooltip');

  // Build series data — skip the null-timestamp anchor, keep the rest
  const data = (points || [])
    .filter(p => p.t)
    .map(p => ({ time: Math.floor(new Date(p.t).getTime() / 1000), value: p.v }))
    .filter(p => !isNaN(p.time))
    // dedupe — lightweight-charts requires strictly ascending time
    .filter((p, i, arr) => i === 0 || p.time > arr[i-1].time);

  if (data.length < 2) {
    container.innerHTML = '<div style="height:220px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px">No closed trades yet</div>';
    return;
  }

  const isPos     = data[data.length - 1].value >= initial;
  const lineColor = isPos ? '#22c55e' : '#f87171';
  const topColor  = isPos ? 'rgba(34,197,94,0.25)'  : 'rgba(248,113,113,0.22)';
  const botColor  = isPos ? 'rgba(34,197,94,0.02)'  : 'rgba(248,113,113,0.02)';

  // Destroy previous chart instance on re-render
  if (_equityChart) { _equityChart.remove(); _equityChart = null; }
  container.innerHTML = '';

  const chart = LightweightCharts.createChart(container, {
    width:  container.clientWidth,
    height: 220,
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
      vertLine: { color: 'rgba(200,169,110,0.6)', width: 1, style: LightweightCharts.LineStyle.Solid, labelBackgroundColor: '#1a1c28' },
      horzLine: { color: 'rgba(200,169,110,0.6)', width: 1, style: LightweightCharts.LineStyle.Solid, labelBackgroundColor: '#1a1c28' },
    },
    rightPriceScale: {
      borderColor: 'rgba(255,255,255,0.06)',
    },
    timeScale: {
      borderColor:     'rgba(255,255,255,0.06)',
      timeVisible:     true,
      secondsVisible:  false,
      tickMarkFormatter: ts => {
        const d = new Date(ts * 1000);
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'America/Chicago' });
      },
    },
    localization: {
      priceFormatter: v => `$${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
    },
    handleScroll: true,
    handleScale:  true,
  });

  const series = chart.addAreaSeries({
    lineColor, topColor, bottomColor: botColor,
    lineWidth: 2,
    priceLineVisible: false,
    lastValueVisible: true,
    crosshairMarkerVisible: true,
    crosshairMarkerRadius: 5,
    crosshairMarkerBorderColor: lineColor,
    crosshairMarkerBackgroundColor: '#0d0f18',
  });

  series.setData(data);

  // Baseline — initial capital
  series.createPriceLine({
    price:              initial,
    color:              'rgba(200,169,110,0.45)',
    lineWidth:          1,
    lineStyle:          LightweightCharts.LineStyle.Dashed,
    axisLabelVisible:   true,
    title:              'start',
  });

  chart.timeScale().fitContent();
  _equityChart = chart;

  // Resize observer
  new ResizeObserver(() => {
    chart.applyOptions({ width: container.clientWidth });
  }).observe(container);

  // Hover tooltip
  chart.subscribeCrosshairMove(param => {
    if (!param.time || !param.seriesData || !param.seriesData.has(series)) {
      tooltip.style.display = 'none';
      return;
    }
    const pt      = param.seriesData.get(series);
    const value   = pt.value;
    const change  = value - initial;
    const changePct = (change / initial) * 100;
    const ts      = new Date(param.time * 1000);
    const timeStr = ts.toLocaleString('en-US', { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit', hour12:false, timeZone:'America/Chicago' });
    const cc      = change >= 0 ? '#22c55e' : '#f87171';
    const sign    = change >= 0 ? '+' : '';

    document.getElementById('tt-value').textContent  = `$${value.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}`;
    document.getElementById('tt-change').innerHTML   = `<span style="color:${cc}">${sign}$${Math.abs(change).toFixed(2)} (${sign}${changePct.toFixed(2)}%)</span>`;
    document.getElementById('tt-time').textContent   = timeStr;

    // Keep tooltip inside the card
    const rect = container.getBoundingClientRect();
    const x    = param.point.x;
    tooltip.style.left    = x > container.clientWidth / 2 ? 'auto' : '14px';
    tooltip.style.right   = x > container.clientWidth / 2 ? '14px' : 'auto';
    tooltip.style.display = 'block';
  });

  // Hide tooltip on mouse leave
  container.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

  // Curve summary
  const total = data[data.length - 1].value - initial;
  document.getElementById('curve-meta').textContent =
    `${data.length} trades · ${fmt.pnl(total)} total`;
}

// ── Render helpers ────────────────────────────────────────────────────────────
function renderStats(d) {
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
  set('s-avgwinloss', `${fmt.pnl(d.performance.avg_win)} / ${fmt.pnl(d.performance.avg_loss)}`);

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
      <td style="text-align:center"><span class="tier-badge ${a.tier===1?'tier-1':''}">${a.tier}</span></td>
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
function startCountdown() {
  clearInterval(_countdownIv);
  _countdownIv = setInterval(() => {
    if (!_lastScanIso) { set('s-countdown', '—'); return; }
    const elapsed = (Date.now() - new Date(_lastScanIso)) / 1000;
    const remaining = Math.max(0, Math.ceil(_scanInterval - elapsed));
    set('s-countdown', `${remaining}s`);
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
    renderEquityCurve(d.equity_curve, d.initial);

    set('last-scan-label', `updated ${new Date().toLocaleTimeString()}`);
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
    new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
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
