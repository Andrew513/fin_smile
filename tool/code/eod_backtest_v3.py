
import argparse
import os
import glob
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

def load_signals(signals_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(signals_dir, "signals_*.csv")))
    if not files:
        raise SystemExit("No signals_*.csv files found")
    dfs = []
    for f in files:
        x = pd.read_csv(f, parse_dates=["asof","expiry"])
        x["asof"] = pd.to_datetime(x["asof"]).dt.date
        dfs.append(x)
    df = pd.concat(dfs, ignore_index=True)
    return df

def get_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False)["Adj Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers if isinstance(tickers,str) else "PX")
    return data

def forward_return(px: pd.Series, date: dt.date, holding_days: int) -> float:
    dates = px.index.date
    if date not in dates: return np.nan
    idx = list(dates).index(date)
    j = idx + holding_days
    if j >= len(px): return np.nan
    p0 = float(px.iloc[idx]); p1 = float(px.iloc[j])
    return (p1 - p0) / p0

def main():
    ap = argparse.ArgumentParser(description="End-of-day backtest for v3 signals")
    ap.add_argument("--signals_dir", type=str, default=".")
    ap.add_argument("--tickers", type=str, required=True)
    ap.add_argument("--holding", type=int, default=1, help="holding days (1 or 2)")
    args = ap.parse_args()

    df = load_signals(args.signals_dir)
    tickers = sorted(list({t for t in args.tickers.split(',') if t}))
    start = df["asof"].min() - pd.Timedelta(days=5)
    end   = df["asof"].max() + pd.Timedelta(days=args.holding+5)

    px = get_prices(tickers, start=start, end=end)

    rets = []
    for _, r in df.iterrows():
        t = r["ticker"]; d = r["asof"]
        series = px[t] if t in px.columns else px.iloc[:,0]
        rets.append(forward_return(series, d, args.holding))
    df["fwd_ret"] = rets

    summ = (
        df.dropna(subset=["fwd_ret"])
          .groupby(["ticker","label","side"])
          .agg(n=("fwd_ret","count"),
               hit=("fwd_ret", lambda s: float((s>0).mean())),
               mean_ret=("fwd_ret","mean"),
               median_ret=("fwd_ret","median"),
               p75=("fwd_ret", lambda s: float(np.percentile(s, 75))),
               p25=("fwd_ret", lambda s: float(np.percentile(s, 25))))
          .reset_index()
          .sort_values(["mean_ret"], ascending=False)
    )
    out = os.path.join(args.signals_dir, f"bt_summary_h{args.holding}.csv")
    summ.to_csv(out, index=False)
    print(summ.to_string(index=False))
    print(f"Saved backtest summary to {out}")

if __name__ == "__main__":
    main()
