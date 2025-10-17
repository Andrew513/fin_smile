
import argparse
import datetime as dt
import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ------------------------- Config & Helpers -------------------------

DEF_RISK_FREE = float(os.getenv("RISK_FREE_RATE", "0.04"))  # approx annual risk-free
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()

@dataclass
class ParityAnomaly:
    ticker: str
    asof: dt.date
    expiry: dt.date
    dte: int
    strike: float
    spot: float
    call_mid: float
    put_mid: float
    call_iv: float
    put_iv: float
    dividend_yield: float
    r: float
    parity_residual: float  # (C - P) - (S*e^{-qT} - K*e^{-rT})
    iv_gap: float           # call_iv - put_iv
    score: float            # composite anomaly score

def mid_price(row) -> Optional[float]:
    # Prefer (bid+ask)/2; else lastPrice
    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    last = row.get("lastPrice", np.nan)
    price = np.nan
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0:
        price = (bid + ask) / 2.0
    elif np.isfinite(last) and last > 0:
        price = float(last)
    return None if not np.isfinite(price) else float(price)

def choose_expiry(expiries: list, target_dte: int, min_dte: int, max_dte: int) -> Optional[str]:
    today = dt.date.today()
    valid = []
    for e in expiries:
        ed = dt.date.fromisoformat(e)
        dte = (ed - today).days
        if min_dte <= dte <= max_dte:
            valid.append((abs(dte - target_dte), e))
    if not valid:
        return None
    return sorted(valid)[0][1]

def fetch_chain(ticker: str, min_dte: int, max_dte: int, target_dte: int):
    tkr = yf.Ticker(ticker)
    expiries = tkr.options
    if not expiries:
        return None, None, None
    picked = choose_expiry(expiries, target_dte, min_dte, max_dte)
    if picked is None:
        return None, None, None
    opt = tkr.option_chain(picked)
    calls, puts = opt.calls, opt.puts
    info = tkr.info or {}
    # Spot price
    spot = np.nan
    try:
        spot = float(tkr.fast_info.last_price)
    except Exception:
        spot = float(info.get("regularMarketPrice") or info.get("currentPrice") or np.nan)
    return calls, puts, (picked, spot, info)

def parity_residual(call_mid: float, put_mid: float, S: float, K: float, r: float, q: float, T_years: float) -> float:
    # Residual = (C - P) - (S*e^{-qT} - K*e^{-rT})
    lhs = call_mid - put_mid
    rhs = S * math.exp(-q * T_years) - K * math.exp(-r * T_years)
    return lhs - rhs

def make_anomalies_for_ticker(ticker: str, min_dte: int, max_dte: int, target_dte: int) -> List[ParityAnomaly]:
    calls, puts, meta = fetch_chain(ticker, min_dte, max_dte, target_dte)
    if calls is None or puts is None or meta is None:
        return []
    expiry_str, spot, info = meta
    if not np.isfinite(spot):
        return []
    expiry = dt.date.fromisoformat(expiry_str)
    today = dt.date.today()
    dte = (expiry - today).days
    T_years = max(dte, 0) / 365.0

    q = float(info.get("dividendYield") or 0.0)  # fractional
    r = DEF_RISK_FREE

    calls = calls.copy()
    puts = puts.copy()
    calls["call_mid"] = calls.apply(mid_price, axis=1)
    puts["put_mid"] = puts.apply(mid_price, axis=1)

    c = calls[["strike","call_mid","impliedVolatility"]].rename(columns={"impliedVolatility":"call_iv"})
    p = puts [["strike","put_mid","impliedVolatility"]].rename(columns={"impliedVolatility":"put_iv"})
    ch = pd.merge(c, p, on="strike", how="inner")

    anomalies: List[ParityAnomaly] = []
    for _, row in ch.iterrows():
        K = float(row["strike"])
        call_mid = row["call_mid"]
        put_mid  = row["put_mid"]
        call_iv  = row["call_iv"]
        put_iv   = row["put_iv"]
        if any([x is None for x in [call_mid, put_mid]]):
            continue
        if not all(np.isfinite([call_mid, put_mid, call_iv, put_iv])):
            continue
        res = parity_residual(call_mid, put_mid, spot, K, r, q, T_years)
        iv_gap = float(call_iv - put_iv)
        score = abs(res) + 0.5 * abs(iv_gap) * spot * 0.01  # crude scaling to combine $ and vol points
        anomalies.append(ParityAnomaly(
            ticker=ticker, asof=today, expiry=expiry, dte=dte, strike=K, spot=spot,
            call_mid=float(call_mid), put_mid=float(put_mid),
            call_iv=float(call_iv), put_iv=float(put_iv),
            dividend_yield=q, r=r,
            parity_residual=float(res), iv_gap=float(iv_gap), score=float(score)
        ))
    anomalies.sort(key=lambda x: x.score, reverse=True)
    return anomalies

def format_report(anoms: List[ParityAnomaly], top_n: int = 5) -> str:
    if not anoms:
        return "No anomalies found."
    rows = []
    hdr = "ticker asof expiry dte strike spot call_mid put_mid call_iv put_iv parity_res iv_gap score".split()
    rows.append("\t".join(hdr))
    for a in anoms[:top_n]:
        rows.append("\t".join([
            a.ticker, str(a.asof), str(a.expiry), str(a.dte),
            f"{a.strike:.2f}", f"{a.spot:.2f}", f"{a.call_mid:.2f}", f"{a.put_mid:.2f}",
            f"{a.call_iv*100:.2f}%", f"{a.put_iv*100:.2f}%", f"{a.parity_residual:.3f}", f"{a.iv_gap*100:.2f}%", f"{a.score:.3f}"
        ]))
    return "\n".join(rows)

def post_slack(text: str):
    if not SLACK_WEBHOOK_URL:
        return
    try:
        requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=10)
    except Exception as e:
        print(f"[WARN] Slack post failed: {e}")

def main():
    ap = argparse.ArgumentParser(description="Put–Call Parity anomaly scanner")
    ap.add_argument("--tickers", type=str, required=True, help="Comma-separated tickers, e.g., AAPL,MSFT,TSLA")
    ap.add_argument("--min_dte", type=int, default=20)
    ap.add_argument("--max_dte", type=int, default=60)
    ap.add_argument("--target_dte", type=int, default=30)
    ap.add_argument("--top_n", type=int, default=5)
    ap.add_argument("--min_abs_residual", type=float, default=0.10, help="Min $ residual to flag")
    ap.add_argument("--min_iv_gap_pct", type=float, default=0.03, help="Min abs(call_iv - put_iv) in decimal, e.g. 0.03=3pp")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    all_rows = []
    for t in tickers:
        anoms = make_anomalies_for_ticker(t, args.min_dte, args.max_dte, args.target_dte)
        # thresholds
        anoms = [a for a in anoms if (abs(a.parity_residual) >= args.min_abs_residual and abs(a.iv_gap) >= args.min_iv_gap_pct)]
        report = format_report(anoms, top_n=args.top_n)
        header = f"== {t} | {dt.date.today()} =="
        print(header)
        print(report)
        print()
        post_slack(f"{header}\n```{report}```")
        for a in anoms[:args.top_n]:
            all_rows.append({
                "ticker": a.ticker, "asof": a.asof, "expiry": a.expiry, "dte": a.dte,
                "strike": a.strike, "spot": a.spot, "call_mid": a.call_mid, "put_mid": a.put_mid,
                "call_iv": a.call_iv, "put_iv": a.put_iv, "parity_residual": a.parity_residual, "iv_gap": a.iv_gap, "score": a.score
            })
    if all_rows:
        df = pd.DataFrame(all_rows)
        out = f"parity_anomalies_{dt.date.today().isoformat()}.csv"
        df.to_csv(out, index=False)
        print(f"Saved: {out}")

if __name__ == "__main__":
    main()
