#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto IV Smile (Dual Lists):
- Earnings in next N days -> top-K by market cap
- Plus a fixed "theme" watchlist you provided
- For each symbol: pick ~30DTE expiry, fetch option chain via yfinance,
  standardize IV smile by moneyness (K/S), overlay last N snapshots,
  and save both PNG and CSV.

Notes:
- yfinance options are delayed ~15–20 minutes.
- This script *builds* your intraday history by taking snapshots regularly.
"""

import os
import time
import math
import argparse
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser
from zoneinfo import ZoneInfo
from typing import List, Optional, Tuple

# ---------------- Configuration ----------------
CHI_TZ = ZoneInfo("America/Chicago")
FINNHUB_BASE = "https://finnhub.io/api/v1"

THEME_LIST = [
    # 核能（傳統 / 新型 / 鈾礦）
    "CEG", "VST", "EXC", "D", "DUK", "OKLO", "SMR", "BWXT",
    "CCJ", "UUUU", "DNN", "UEC",
    # 數據 / 晶片 / AI / 雲端
    "VRT", "NVDA", "AMD", "INTC", "MSFT", "EQIX", "GOOGL",
    # 量子
    "IONQ", "RGTI", "QBTS", "IBM", "HON",
    # 稀土
    "MP", "CRML", "USAR",
    # 你指定的下週重點
    "NXPI", "UPS", "UNH", "BA", "GOOGL", "META", "MSFT", "MA", "AMZN"
]

# ---------------- Utilities ----------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def market_hours_chicago(now: datetime) -> bool:
    # Regular Trading Hours: 08:30–15:00 CT
    t = now.astimezone(CHI_TZ).time()
    return (t >= datetime.strptime("08:30", "%H:%M").time() and
            t <= datetime.strptime("15:00", "%H:%M").time())

# ---------------- Earnings (Finnhub) ----------------
def fetch_earnings_symbols_finnhub(api_key: str, start_date: str, end_date: str) -> List[str]:
    """
    Finnhub earnings calendar:
    GET /calendar/earnings?from=YYYY-MM-DD&to=YYYY-MM-DD&token=API_KEY
    """
    url = f"{FINNHUB_BASE}/calendar/earnings"
    params = {"from": start_date, "to": end_date, "token": api_key}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json() or {}
    items = data.get("earningsCalendar", []) or data.get("earnings", []) or []
    syms = []
    for it in items:
        sym = (it.get("symbol") or "").strip().upper()
        if sym:
            syms.append(sym)
    return sorted(set(syms))

# ---------------- Market Cap (yfinance) ----------------
def get_market_caps(tickers: List[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["symbol", "market_cap"])
    tkr = yf.Tickers(" ".join(tickers))
    out = []
    for sym, obj in tkr.tickers.items():
        mc = None
        try:
            fi = getattr(obj, "fast_info", {})
            mc = fi.get("market_cap", None)
        except Exception:
            pass
        if mc is None:
            try:
                info = obj.info or {}
                mc = info.get("marketCap", None)
            except Exception:
                pass
        out.append({"symbol": sym.upper(), "market_cap": mc})
    return pd.DataFrame(out)

def pick_top_by_mcap(symbols: List[str], limit: int) -> List[str]:
    df = get_market_caps(symbols)
    if df.empty or df["market_cap"].isna().all():
        return sorted(symbols)[:limit]
    ranked = df.dropna(subset=["market_cap"]).sort_values("market_cap", ascending=False)["symbol"].tolist()
    missing = [s for s in symbols if s not in set(ranked)]
    ordered = ranked + missing
    return ordered[:limit] if limit else ordered

# ---------------- Options (yfinance) ----------------
def choose_target_expiration(exp_list: List[str], target_dte: int = 30, min_dte: int = 7) -> Optional[str]:
    today = datetime.now(timezone.utc).date()
    best, bestdiff = None, 10**9
    for e in exp_list or []:
        try:
            d = dtparser.parse(e).date()
        except Exception:
            continue
        dte = (d - today).days
        if dte < min_dte:
            continue
        diff = abs(dte - target_dte)
        if diff < bestdiff:
            best, bestdiff = e, diff
    if not best and exp_list:
        best = exp_list[-1]
    return best

def fetch_chain(symbol: str, expiration: str) -> pd.DataFrame:
    tk = yf.Ticker(symbol)
    oc = tk.option_chain(expiration)
    calls = oc.calls.copy()
    puts  = oc.puts.copy()
    calls["type"] = "call"
    puts["type"]  = "put"
    df = pd.concat([calls, puts], ignore_index=True)

    # spot price
    spot = None
    try:
        spot = tk.fast_info.get("last_price") or tk.info.get("currentPrice")
    except Exception:
        pass
    if spot is None:
        try:
            h = tk.history(period="1d", interval="1m")
            if not h.empty:
                spot = float(h["Close"].iloc[-1])
        except Exception:
            pass

    df["spot"] = spot
    df = df.dropna(subset=["impliedVolatility"])
    df["impliedVolatility"] = df["impliedVolatility"] * 100.0  # to %
    return df

# ---------------- Standardize (moneyness) ----------------
def to_moneyness(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df["spot"].isna().all():
        return pd.DataFrame()
    S = df["spot"].iloc[0]
    if not (isinstance(S, (float, int)) and S == S and S > 0):
        return pd.DataFrame()
    out = df.copy()
    out["moneyness"] = out["strike"] / float(S)
    keep = ["contractSymbol","type","strike","moneyness","impliedVolatility","spot",
            "lastPrice","bid","ask","volume","openInterest"]
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan
    return out[keep].dropna(subset=["moneyness","impliedVolatility"])

def build_mgrid(kmin=0.85, kmax=1.15, step=0.01) -> np.ndarray:
    return np.arange(kmin, kmax + 1e-9, step)

def interp_on_grid(df_mny: pd.DataFrame, mgrid: np.ndarray, opt_type: str) -> np.ndarray:
    sub = df_mny[df_mny["type"] == opt_type].sort_values("moneyness")
    if sub.empty:
        return np.full_like(mgrid, np.nan, dtype=float)
    x = sub["moneyness"].values
    y = sub["impliedVolatility"].values
    return np.interp(mgrid, x, y, left=np.nan, right=np.nan)

# ---------------- I/O & Plot ----------------
def save_snapshot_csv(df_mny: pd.DataFrame, out_root: str, bucket: str,
                      symbol: str, expiration: str, ts_local: datetime) -> str:
    csv_dir = os.path.join(out_root, bucket, symbol, "csv")
    ensure_dir(csv_dir)
    ts_str = ts_local.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(csv_dir, f"{symbol}_{expiration}_{ts_str}.csv")
    df_mny.to_csv(path, index=False)
    return path

def list_history_csv(csv_dir: str, n: int) -> List[str]:
    if not os.path.isdir(csv_dir):
        return []
    files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.lower().endswith(".csv")]
    files.sort()
    return files[-n:] if n > 0 else []

def plot_overlay(symbol: str, expiration: str, ts_local: datetime,
                 mgrid: np.ndarray, call_y: np.ndarray, put_y: np.ndarray,
                 out_root: str, bucket: str, history_n: int, ypad: float = 2.0) -> str:
    sym_dir = os.path.join(out_root, bucket, symbol)
    ensure_dir(sym_dir)
    png_path = os.path.join(sym_dir, f"{symbol}_{expiration}_{ts_local.strftime('%Y%m%d_%H%M%S')}.png")

    # compute y-range based on current + history
    yvals = np.concatenate([call_y[~np.isnan(call_y)], put_y[~np.isnan(put_y)]])
    if yvals.size == 0:
        ymin, ymax = 0.0, 100.0
    else:
        ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)

    # read recent history
    csv_dir = os.path.join(sym_dir, "csv")
    ensure_dir(csv_dir)
    hist_csvs = list_history_csv(csv_dir, history_n - 1)
    hist_curves = []
    for path in hist_csvs:
        try:
            dfh = pd.read_csv(path)
            if {"moneyness","type","impliedVolatility"}.issubset(dfh.columns):
                ch = dfh[dfh["type"]=="call"].sort_values("moneyness")
                ph = dfh[dfh["type"]=="put" ].sort_values("moneyness")
                cy = np.interp(mgrid, ch["moneyness"].values, ch["impliedVolatility"].values,
                               left=np.nan, right=np.nan) if not ch.empty else np.full_like(mgrid, np.nan)
                py = np.interp(mgrid, ph["moneyness"].values, ph["impliedVolatility"].values,
                               left=np.nan, right=np.nan) if not ph.empty else np.full_like(mgrid, np.nan)
                hist_curves.append((cy, py))
                vals = np.concatenate([cy[~np.isnan(cy)], py[~np.isnan(py)]])
                if vals.size:
                    ymin = min(ymin, np.nanmin(vals))
                    ymax = max(ymax, np.nanmax(vals))
        except Exception:
            continue

    ymin = max(0.0, ymin - ypad)
    ymax = ymax + ypad

    # plot
    plt.figure(figsize=(9, 5.5))
    for cy, py in hist_curves:
        plt.plot(mgrid, cy, alpha=0.25, linewidth=1.0)
        plt.plot(mgrid, py, alpha=0.25, linewidth=1.0)
    plt.plot(mgrid, call_y, linewidth=2.0, label="Calls (K/S)")
    plt.plot(mgrid, put_y,  linewidth=2.0, label="Puts (K/S)")
    plt.title(f"{symbol}  IV Smile (Std. K/S)  |  Exp {expiration}\n{ts_local.strftime('%Y-%m-%d %H:%M %Z')}")
    plt.xlabel("Moneyness  K/S")
    plt.ylabel("Implied Volatility  (%)")
    plt.xlim(mgrid.min(), mgrid.max())
    plt.ylim(ymin, ymax)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    return png_path

# ---------------- Pipeline ----------------
def process_symbols(symbols: List[str], out_root: str, bucket: str,
                    target_dte: int, history_n: int, mgrid: np.ndarray):
    symbols = list(dict.fromkeys([s.upper() for s in symbols if isinstance(s, str) and s.strip()]))
    for sym in symbols:
        ts_local = datetime.now(CHI_TZ)
        try:
            tk = yf.Ticker(sym)
            exps = tk.options or []
            exp = choose_target_expiration(exps, target_dte=target_dte, min_dte=7)
            if not exp:
                print(f"[{bucket}:{sym}] 無合適到期日，略過。")
                continue
            df = fetch_chain(sym, exp)
            if df.empty:
                print(f"[{bucket}:{sym}] 無期權資料，略過。")
                continue
            df_mny = to_moneyness(df)
            if df_mny.empty:
                print(f"[{bucket}:{sym}] 無法計算 moneyness（可能沒有現價），略過。")
                continue

            # interpolate
            call_y = interp_on_grid(df_mny, mgrid, "call")
            put_y  = interp_on_grid(df_mny, mgrid, "put")

            # save csv + png
            save_snapshot_csv(df_mny, out_root, bucket, sym, exp, ts_local)
            png_path = plot_overlay(sym, exp, ts_local, mgrid, call_y, put_y,
                                    out_root, bucket, history_n)
            print(f"[{ts_local.isoformat(timespec='seconds')}] {bucket}:{sym} exp {exp} -> {png_path}")
        except Exception as e:
            print(f"[{bucket}:{sym}] Error: {e}")

def run_once(days: int, limit: int, target_dte: int, out_root: str, history_n: int):
    ensure_dir(out_root)
    mgrid = build_mgrid(0.85, 1.15, 0.01)

    # Earnings bucket
    finnhub_key = os.getenv("FINNHUB_TOKEN") or os.getenv("FINNHUB_KEY")
    if not finnhub_key:
        raise SystemExit("Missing Finnhub API key. Set FINNHUB_TOKEN env var.")
    today = datetime.now(CHI_TZ).date()
    start_str = today.isoformat()
    end_str   = (today + timedelta(days=days)).isoformat()

    earnings_syms = fetch_earnings_symbols_finnhub(finnhub_key, start_str, end_str)
    if not earnings_syms:
        print(f"[{datetime.now(CHI_TZ).isoformat(timespec='seconds')}] 未找到未來 {days} 天的財報股票。")
    else:
        top_syms = pick_top_by_mcap(earnings_syms, limit)
        print(f"Earnings Top {len(top_syms)}: {top_syms}")
        process_symbols(top_syms, out_root, "iv_smiles_earnings", target_dte, history_n, mgrid)

    # Theme bucket
    theme_syms = sorted(set(THEME_LIST))
    print(f"Theme Watchlist ({len(theme_syms)}): {theme_syms}")
    process_symbols(theme_syms, out_root, "iv_smiles_theme", target_dte, history_n, mgrid)

def main():
    ap = argparse.ArgumentParser(description="Auto IV smiles for earnings Top-N + fixed theme list (clean & comparable).")
    ap.add_argument("--days", type=int, default=5, help="Lookahead days for earnings (default 5).")
    ap.add_argument("--limit", type=int, default=20, help="Max earnings symbols (default 20).")
    ap.add_argument("--target-dte", type=int, default=30, help="Target DTE for expiry selection (default 30).")
    ap.add_argument("--interval", type=int, default=3600, help="Seconds between rounds (default 3600).")
    ap.add_argument("--history", type=int, default=6, help="Overlay last N snapshots per symbol (default 6).")
    ap.add_argument("--out", default="tool/code/simple_smile_hourly/iv_smiles_dual", help="Output root directory.")
    ap.add_argument("--once", action="store_true", help="Run one round and exit.")
    ap.add_argument("--no-market-hours-guard", action="store_true", help="Run regardless of market hours.")
    args = ap.parse_args()

    ensure_dir(args.out)

    if args.once:
        run_once(args.days, args.limit, args.target_dte, args.out, args.history)
        return

    while True:
        now = datetime.now(CHI_TZ)
        if args.no_market_hours_guard or market_hours_chicago(now):
            print(f"\n=== Round @ {now.isoformat(timespec='seconds')} {CHI_TZ.key} ===")
            run_once(args.days, args.limit, args.target_dte, args.out, args.history)
        else:
            print(f"[{now.isoformat(timespec='seconds')}] 非交易時段，略過本輪。")
        time.sleep(max(30, args.interval))

if __name__ == "__main__":
    main()