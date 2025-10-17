#!/usr/bin/env python3
# iv_earnings_scanner.py
# Scan a ticker universe for: (1) earnings next week, (2) relatively small stock volume,
# and (3) options with IV & volume above thresholds. Ranks by IV * volume.
#
# Usage:
#   pip install yfinance pandas matplotlib
#   python iv_earnings_scanner.py
#
# Notes:
# - Provide your own TICKERS list (universe) below or load from a file.
# - Internet access is required for yfinance to fetch data.
# - Outputs a CSV summary and (optionally) plots smiles per matched ticker.

import datetime as dt
import math
from typing import Dict, Optional, Tuple, List

import pandas as pd
import yfinance as yf

# ---------------------- USER SETTINGS ----------------------

# 1) Define your universe (edit this list)
TICKERS = [
    "MP", "CRML", "USAR",  # examples given
    # add more tickers here ...
]

# 2) Define what "較小交易量" means (average daily trading volume threshold)
MAX_STOCK_ADTV = 2_000_000  # shares/day (10-day average). Adjust as needed.

# 3) Option filters
NEAREST_N_EXPIRIES = 2       # how many nearest expiries to check
MIN_OPTION_VOLUME = 300      # minimum per-contract option volume to consider
MIN_IV = 0.45                # minimum implied vol (e.g., 0.45 = 45%)
MIN_IV_X_VOL = 120.0         # minimum (IV * volume) for highlighting liquidity+IV

# 4) Plot settings
MAKE_PLOTS = False           # set True to save PNG smiles per matched ticker

# 5) Output
OUT_CSV = "iv_earnings_candidates.csv"


# ---------------------- HELPERS ----------------------

def next_week_window(today: Optional[dt.date] = None) -> Tuple[dt.datetime, dt.datetime]:
    """Return [start, end) window for 'next week' in UTC-naive datetimes."""
    if today is None:
        today = dt.date.today()
    # Define next week as the next 7 calendar days starting tomorrow
    start = today + dt.timedelta(days=1)
    end = start + dt.timedelta(days=7)
    start_dt = dt.datetime.combine(start, dt.time.min)
    end_dt = dt.datetime.combine(end, dt.time.min)
    return start_dt, end_dt


def get_next_earnings_date(ticker: str) -> Optional[pd.Timestamp]:
    """Return the nearest upcoming earnings date for a ticker, if available."""
    try:
        t = yf.Ticker(ticker)
        # get_earnings_dates returns a DataFrame with index as dates
        df = t.get_earnings_dates(limit=8)  # a few upcoming/past dates
        if df is None or df.empty:
            return None
        # pick the earliest future date
        now = pd.Timestamp.utcnow().tz_localize(None)
        future = df[df.index.tz_localize(None) >= now]
        if future.empty:
            return None
        return future.index.min().tz_localize(None)
    except Exception as e:
        print(f"[WARN] {ticker}: earnings fetch failed: {e}")
        return None


def avg_volume_10d(ticker: str) -> Optional[float]:
    """Return the 10-day average volume for the equity (shares/day)."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        vol = info.get("averageVolume10days", None)
        if vol is None:
            # fallback: compute from recent history
            hist = t.history(period="1mo", interval="1d")
            if hist is None or hist.empty:
                return None
            return float(hist["Volume"].tail(10).mean())
        return float(vol)
    except Exception as e:
        print(f"[WARN] {ticker}: avg volume fetch failed: {e}")
        return None


def nearest_expiries(ticker: str, n: int = 2) -> List[str]:
    try:
        t = yf.Ticker(ticker)
        exps = list(t.options or [])
        return exps[:n]
    except Exception as e:
        print(f"[WARN] {ticker}: options expiries fetch failed: {e}")
        return []


def load_option_chain(ticker: str, expiry: str) -> Optional[pd.DataFrame]:
    """Return merged calls+puts with columns: strike, type, impliedVolatility, volume, lastPrice, bid, ask."""
    try:
        t = yf.Ticker(ticker)
        oc = t.option_chain(expiry)
        calls = oc.calls.copy()
        puts = oc.puts.copy()
        calls["type"] = "call"
        puts["type"] = "put"
        df = pd.concat([calls, puts], ignore_index=True, sort=False)
        # keep only relevant columns
        keep = ["strike", "type", "impliedVolatility", "volume", "lastPrice", "bid", "ask", "openInterest"]
        return df[keep].dropna(subset=["impliedVolatility"])
    except Exception as e:
        print(f"[WARN] {ticker}: option chain fetch failed for {expiry}: {e}")
        return None


def summarize_options(df: pd.DataFrame) -> pd.DataFrame:
    """Add iv_x_vol column and filter to highlight contracts meeting thresholds."""
    df = df.copy()
    df["iv_x_vol"] = df["impliedVolatility"] * df["volume"].astype(float)
    # mark whether contract passes individual thresholds
    df["pass_filters"] = (
        (df["volume"] >= MIN_OPTION_VOLUME) &
        (df["impliedVolatility"] >= MIN_IV) &
        (df["iv_x_vol"] >= MIN_IV_X_VOL)
    )
    return df.sort_values("iv_x_vol", ascending=False)


# ---------------------- MAIN SCAN ----------------------

def main():
    start, end = next_week_window()
    print(f"Scanning earnings in next-week window: {start.date()} -> {end.date()}")

    rows = []

    for tk in TICKERS:
        print(f"\n=== {tk} ===")

        ed = get_next_earnings_date(tk)
        if ed is None:
            print("  No upcoming earnings date found.")
            continue

        if not (start <= ed.to_pydatetime() < end):
            print(f"  Earnings not in next-week window: {ed.date()}")
            continue

        adtv = avg_volume_10d(tk)
        if adtv is None:
            print("  Could not determine 10D average volume; skipping.")
            continue

        if adtv > MAX_STOCK_ADTV:
            print(f"  Skipping due to larger ADTV ({adtv:,.0f} > {MAX_STOCK_ADTV:,.0f})")
            continue

        print(f"  Earnings: {ed.date()} | 10D ADTV: {adtv:,.0f} shares/day")

        expiries = nearest_expiries(tk, NEAREST_N_EXPIRIES)
        if not expiries:
            print("  No option expiries found; skipping.")
            continue

        best_records = []
        total_pass_vol = 0
        for ex in expiries:
            oc = load_option_chain(tk, ex)
            if oc is None or oc.empty:
                continue
            oc2 = summarize_options(oc)

            # aggregate stats
            total_pass_vol += oc2.loc[oc2["pass_filters"], "volume"].sum()

            top = oc2.head(3)  # top 3 by IV * volume
            for _, r in top.iterrows():
                best_records.append({
                    "ticker": tk, "earnings_date": ed.date(), "expiry": ex,
                    "type": r["type"], "strike": r["strike"],
                    "iv": float(r["impliedVolatility"]),
                    "volume": int(r["volume"]),
                    "iv_x_vol": float(r["iv_x_vol"]),
                    "bid": float(r.get("bid", math.nan)),
                    "ask": float(r.get("ask", math.nan)),
                    "openInterest": int(r.get("openInterest", 0)),
                    "adtv_10d": int(adtv),
                })

        if best_records:
            # keep the top 5 across expiries for this ticker
            best_records = sorted(best_records, key=lambda x: x["iv_x_vol"], reverse=True)[:5]
            rows.extend(best_records)
            print(f"  Found {len(best_records)} interesting contracts; total pass volume (sum): {int(total_pass_vol)}")
        else:
            print("  No contracts met the filters.")

    if not rows:
        print("\nNo candidates found under current thresholds.")
        return

    out = pd.DataFrame(rows)
    out["iv_pct"] = (out["iv"] * 100.0).round(2)
    out["iv_x_vol"] = out["iv_x_vol"].round(2)
    out = out[["ticker", "earnings_date", "expiry", "type", "strike", "iv_pct", "volume", "iv_x_vol", "bid", "ask", "openInterest", "adtv_10d"]]

    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    try:
        from tabulate import tabulate
        print(tabulate(out, headers="keys", tablefmt="github", showindex=False))
    except Exception:
        print(out)

if __name__ == "__main__":
    main()
