
import argparse
import datetime as dt
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import requests
import yfinance as yf

DEF_RISK_FREE = float(os.getenv("RISK_FREE_RATE", "0.04"))
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
HISTORY_DIR = os.getenv("HISTORY_DIR", "./history")
os.makedirs(HISTORY_DIR, exist_ok=True)

@dataclass
class Row:
    ticker: str
    asof: dt.date
    expiry: dt.date
    dte: int
    strike: float
    spot: float
    side: str
    moneyness: float
    call_bid: float
    call_ask: float
    put_bid: float
    put_ask: float
    call_iv: float
    put_iv: float
    call_vol: Optional[float]
    put_vol: Optional[float]
    call_oi: Optional[float]
    put_oi: Optional[float]
    parity_residual: float
    iv_gap: float
    iv_gap_z: Optional[float]
    parity_res_z: Optional[float]
    k_consistency: int
    xexpiry_consistency: int
    event_flag: int
    borrow_penalty: float
    score: float
    label: str

def mid(bid: float, ask: float) -> Optional[float]:
    if not (np.isfinite(bid) and np.isfinite(ask)): return None
    if bid <= 0 or ask <= 0: return None
    return (bid + ask) / 2.0

def spread_ok(bid: float, ask: float, max_frac: float = 0.2) -> bool:
    m = mid(bid, ask)
    if m is None: return False
    return (ask - bid) / m <= max_frac

def choose_expiry(expiries: List[str], target_dte: int, min_dte: int, max_dte: int) -> List[str]:
    today = dt.date.today()
    candidates = []
    for e in expiries:
        ed = dt.date.fromisoformat(e)
        dte = (ed - today).days
        if min_dte <= dte <= max_dte:
            candidates.append((abs(dte - target_dte), e))
    candidates.sort(key=lambda x: x[0])
    return [e for _, e in candidates[:2]]

def fetch_chain(ticker: str, min_dte: int, max_dte: int, target_dte: int):
    tkr = yf.Ticker(ticker)
    expiries = tkr.options or []
    if not expiries: return []
    picked = choose_expiry(expiries, target_dte, min_dte, max_dte)
    info = tkr.info or {}
    try:
        spot = float(tkr.fast_info.last_price)
    except Exception:
        spot = float(info.get("regularMarketPrice") or info.get("currentPrice") or np.nan)
    out = []
    for e in picked:
        opt = tkr.option_chain(e)
        out.append((e, opt.calls.copy(), opt.puts.copy(), spot, info))
    return out

def pv_dividends(div_df: Optional[pd.DataFrame], asof: dt.date, expiry: dt.date, r: float) -> float:
    if div_df is None or div_df.empty: return 0.0
    pv = 0.0
    for _, row in div_df.iterrows():
        d = pd.to_datetime(row["ex_date"]).date()
        if asof < d <= expiry:
            days = (d - asof).days
            pv += float(row["amount"]) * math.exp(-r * days/365.0)
    return pv

def parity_residual(call_mid: float, put_mid: float, S: float, K: float, r: float, T: float, pv_div: float) -> float:
    lhs = call_mid - put_mid
    rhs = (S - pv_div) - K * math.exp(-r * T)
    return lhs - rhs

def dte_bucket(dte: int) -> str:
    if dte <= 25: return "20-25"
    if dte <= 35: return "26-35"
    if dte <= 45: return "36-45"
    return "46-60"

def load_events(csv_path: Optional[str]) -> Dict[str, set]:
    if not csv_path or not os.path.exists(csv_path): return {}
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    ev = {}
    for t, g in df.groupby("ticker"):
        ev[t] = set(g["date"].tolist())
    return ev

def rolling_baseline_path(ticker: str) -> str:
    return os.path.join(HISTORY_DIR, f"{ticker}_baseline.parquet")

def update_and_get_baseline(ticker: str, today_rows: pd.DataFrame) -> pd.DataFrame:
    path = rolling_baseline_path(ticker)
    cols = ["date","side","dte_bucket","iv_gap","parity_residual"]
    today = dt.date.today()
    add = today_rows.assign(date=today)[cols]
    if os.path.exists(path):
        old = pd.read_parquet(path)
        base = pd.concat([old, add], ignore_index=True)
        cutoff = (pd.to_datetime(today) - pd.Timedelta(days=90)).date()
        base = base[base["date"] >= cutoff].copy()
    else:
        base = add
    base.to_parquet(path, index=False)
    return base

def robust_z(x: float, median: float, mad: float) -> Optional[float]:
    if mad is None or not np.isfinite(mad) or mad == 0: return None
    return (x - median) / (1.4826 * mad)

def compute_k_consistency(rows: List['Row']) -> List[int]:
    ks = sorted([(i, r.moneyness) for i, r in enumerate(rows)], key=lambda x: x[1])
    idx_to_cons = {i:1 for i,_ in ks}
    for pos, (i, mny) in enumerate(ks):
        for jpos in [pos-1, pos+1]:
            if 0 <= jpos < len(ks):
                j = ks[jpos][0]
                if np.sign(rows[i].iv_gap) == np.sign(rows[j].iv_gap):
                    idx_to_cons[i] += 1
    return [idx_to_cons[i] for i,_ in enumerate(rows)]

def post_slack(text: str):
    if not SLACK_WEBHOOK_URL: return
    try:
        requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=10)
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser(description="Parity/IV anomaly scanner v3")
    ap.add_argument("--tickers", type=str, required=True)
    ap.add_argument("--min_dte", type=int, default=20)
    ap.add_argument("--max_dte", type=int, default=60)
    ap.add_argument("--target_dte", type=int, default=30)
    ap.add_argument("--iv_min", type=float, default=0.03)
    ap.add_argument("--iv_max", type=float, default=1.50)
    ap.add_argument("--max_spread_frac", type=float, default=0.20)
    ap.add_argument("--mny_min", type=float, default=0.90)
    ap.add_argument("--mny_max", type=float, default=1.20)
    ap.add_argument("--top_n", type=int, default=5)
    ap.add_argument("--events_csv", type=str, default="")
    ap.add_argument("--divs_csv", type=str, default="")
    ap.add_argument("--save_signals", action="store_true")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    event_map = load_events(args.events_csv)
    divs = None
    if args.divs_csv and os.path.exists(args.divs_csv):
        divs = pd.read_csv(args.divs_csv)
        divs["ex_date"] = pd.to_datetime(divs["ex_date"]).dt.date

    today = dt.date.today()
    all_rows: List[Row] = []

    for t in tickers:
        batches = fetch_chain(t, args.min_dte, args.max_dte, args.target_dte)
        per_expiry_rows: Dict[str, List[Row]] = {}
        for expiry_str, calls, puts, spot, info in batches:
            if not np.isfinite(spot): continue
            expiry = dt.date.fromisoformat(expiry_str)
            dte = (expiry - today).days
            T = max(dte, 0)/365.0

            div_df = None
            if divs is not None:
                div_df = divs[divs["ticker"]==t][["ex_date","amount"]].copy()

            calls = calls.rename(columns={"impliedVolatility":"call_iv"})
            puts  = puts.rename(columns={"impliedVolatility":"put_iv"})
            merged = pd.merge(
                calls[["strike","bid","ask","call_iv","volume","openInterest"]].rename(
                    columns={"bid":"call_bid","ask":"call_ask","volume":"call_vol","openInterest":"call_oi"}
                ),
                puts[["strike","bid","ask","put_iv","volume","openInterest"]].rename(
                    columns={"bid":"put_bid","ask":"put_ask","volume":"put_vol","openInterest":"put_oi"}
                ),
                on="strike", how="inner"
            )

            merged["call_mid"] = (merged["call_bid"] + merged["call_ask"]) / 2.0
            merged["put_mid"]  = (merged["put_bid"]  + merged["put_ask"])  / 2.0
            mfilter = (
                (merged["call_bid"]>0) & (merged["call_ask"]>0) &
                (merged["put_bid"]>0)  & (merged["put_ask"]>0)  &
                ((merged["call_ask"]-merged["call_bid"])/merged["call_mid"] <= args.max_spread_frac) &
                ((merged["put_ask"] -merged["put_bid"] )/merged["put_mid"]  <= args.max_spread_frac) &
                (merged["call_iv"].between(args.iv_min, args.iv_max)) &
                (merged["put_iv"].between(args.iv_min, args.iv_max))
            )
            merged = merged[mfilter].copy()
            if merged.empty: continue

            merged["moneyness"] = merged["strike"] / spot
            merged = merged[(merged["moneyness"].between(args.mny_min, args.mny_max))].copy()
            if merged.empty: continue

            pv_div = pv_dividends(div_df, today, expiry, DEF_RISK_FREE)
            merged["parity_residual"] = merged.apply(
                lambda r: parity_residual(r["call_mid"], r["put_mid"], spot, float(r["strike"]), DEF_RISK_FREE, T, pv_div),
                axis=1
            )
            merged["iv_gap"] = merged["call_iv"] - merged["put_iv"]

            side = "K>S"  # bucket label for baseline
            bucket = "20-25" if dte<=25 else "26-35" if dte<=35 else "36-45" if dte<=45 else "46-60"
            base_input = merged.assign(side=side, dte_bucket=bucket)[["side","dte_bucket","iv_gap","parity_residual"]]
            base = update_and_get_baseline(t, base_input)
            stats = (
                base.groupby(["side","dte_bucket"])
                    .agg(iv_gap_median=("iv_gap","median"),
                         iv_gap_mad=("iv_gap", lambda s: np.median(np.abs(s - np.median(s)))),
                         pr_median=("parity_residual","median"),
                         pr_mad=("parity_residual", lambda s: np.median(np.abs(s - np.median(s)))))
                    .reset_index()
            )
            st = stats[(stats["side"]==side) & (stats["dte_bucket"]==bucket)]
            if st.empty:
                iv_med = pr_med = 0.0
                iv_mad = pr_mad = 1.0
            else:
                iv_med = float(st["iv_gap_median"].iloc[0])
                iv_mad = float(st["iv_gap_mad"].iloc[0] or 1.0)
                pr_med = float(st["pr_median"].iloc[0])
                pr_mad = float(st["pr_mad"].iloc[0] or 1.0)

            merged["iv_gap_z"] = merged["iv_gap"].apply(lambda x: robust_z(x, iv_med, iv_mad))
            merged["parity_res_z"] = merged["parity_residual"].apply(lambda x: robust_z(x, pr_med, pr_mad))

            rows = []
            for _, r in merged.iterrows():
                side = "K>S" if r["strike"] > spot else "K<S"
                ev_flag = 1 if t in event_map and today in event_map[t] else 0
                borrow_penalty = 0.0
                score = (abs(r["parity_res_z"]) if pd.notnull(r["parity_res_z"]) else 0) \
                        + 0.6*(abs(r["iv_gap_z"]) if pd.notnull(r["iv_gap_z"]) else 0) \
                        - 0.5*ev_flag - 0.5*borrow_penalty
                label = "None"
                if (pd.notnull(r["iv_gap_z"]) and pd.notnull(r["parity_res_z"])):
                    if (r["iv_gap_z"] >= 1.0 and abs(r["parity_res_z"]) >= 1.0):
                        label = "S1_bull" if side=="K>S" else "S1_bear"
                    elif (r["iv_gap_z"] >= 1.5 and abs(r["parity_res_z"]) < 1.0):
                        label = "S2_sentiment_bull" if side=="K>S" else "S2_sentiment_bear"
                    elif (side=="K>S" and r["iv_gap_z"] <= -1.0):
                        label = "S3_fear_on_call_side"
                rows.append(Row(
                    ticker=t, asof=today, expiry=expiry, dte=dte, strike=float(r["strike"]), spot=float(spot),
                    side=side, moneyness=float(r["moneyness"]),
                    call_bid=float(r["call_bid"]), call_ask=float(r["call_ask"]),
                    put_bid=float(r["put_bid"]), put_ask=float(r["put_ask"]),
                    call_iv=float(r["call_iv"]), put_iv=float(r["put_iv"]),
                    call_vol=float(r.get("call_vol", np.nan)) if pd.notnull(r.get("call_vol", np.nan)) else None,
                    put_vol=float(r.get("put_vol", np.nan)) if pd.notnull(r.get("put_vol", np.nan)) else None,
                    call_oi=float(r.get("call_oi", np.nan)) if pd.notnull(r.get("call_oi", np.nan)) else None,
                    put_oi=float(r.get("put_oi", np.nan)) if pd.notnull(r.get("put_oi", np.nan)) else None,
                    parity_residual=float(r["parity_residual"]), iv_gap=float(r["iv_gap"]),
                    iv_gap_z=float(r["iv_gap_z"]) if pd.notnull(r["iv_gap_z"]) else None,
                    parity_res_z=float(r["parity_res_z"]) if pd.notnull(r["parity_res_z"]) else None,
                    k_consistency=1, xexpiry_consistency=0,
                    event_flag=ev_flag, borrow_penalty=borrow_penalty, score=float(score), label=label
                ))
            per_expiry_rows[expiry_str] = rows

        # K-neighborhood consistency
        for expiry_str, rows in per_expiry_rows.items():
            if rows:
                cons_vals = compute_k_consistency(rows)
                for i, cval in enumerate(cons_vals):
                    rows[i].k_consistency = cval

        # cross-expiry consistency
        if len(per_expiry_rows) >= 2:
            exps = list(per_expiry_rows.keys())
            r0 = per_expiry_rows[exps[0]]
            r1 = per_expiry_rows[exps[1]]
            for i, a in enumerate(r0):
                j = min(range(len(r1)), key=lambda k: abs(r1[k].moneyness - a.moneyness))
                if np.sign(a.iv_gap) == np.sign(r1[j].iv_gap):
                    r0[i].xexpiry_consistency = 1
            for i, a in enumerate(r1):
                j = min(range(len(r0)), key=lambda k: abs(r0[k].moneyness - a.moneyness))
                if np.sign(a.iv_gap) == np.sign(r0[j].iv_gap):
                    r1[i].xexpiry_consistency = 1

        for rows in per_expiry_rows.values():
            all_rows.extend(rows)

    if not all_rows:
        print("No signals after filters.")
        return

    df = pd.DataFrame([r.__dict__ for r in all_rows])
    df["score"] = df["score"] + 0.3*df["k_consistency"] + 0.3*df["xexpiry_consistency"]
    df = df.sort_values("score", ascending=False)

    for t in df["ticker"].unique():
        sub = df[df["ticker"]==t].head(args.top_n)
        print(f"== {t} | {today} ==")
        print(sub[["asof","expiry","dte","strike","spot","side","moneyness","call_iv","put_iv","iv_gap","iv_gap_z","parity_residual","parity_res_z","k_consistency","xexpiry_consistency","event_flag","score","label"]].to_string(index=False))
        print()

    if args.save_signals:
        out = os.path.join(".", f"signals_{today.isoformat()}.csv")
        df.to_csv(out, index=False)
        print(f"Saved signals to {out}")

    if SLACK_WEBHOOK_URL:
        brief = df.groupby("ticker")["score"].max().sort_values(ascending=False).head(5)
        post_slack(f"[v3] {today} Top signals:\n" + brief.to_string())

if __name__ == "__main__":
    main()
