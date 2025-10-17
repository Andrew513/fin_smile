import pandas as pd
import numpy as np
import math, os, argparse, datetime as dt

# === 參數 ===
r = 0.04    # 無風險利率
q = 0.01    # 年化股利率近似（AAPL 約 1%）
min_iv, max_iv = 0.03, 1.5
max_spread_frac = 0.2

def mid(bid, ask):
    try:
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
    except Exception:
        pass
    return np.nan

def parity_res(call_mid, put_mid, S, K, r, q, T):
    if pd.isna(call_mid) or pd.isna(put_mid) or pd.isna(S) or pd.isna(K) or pd.isna(T):
        return np.nan
    return (call_mid - put_mid) - (S * math.exp(-q * T) - K * math.exp(-r * T))

def try_load(path):
    seps = [',', '\t', ';', '|']
    for s in seps:
        try:
            df_try = pd.read_csv(path, sep=s, low_memory=False)
            if len(df_try.columns) > 5:
                print(f"[load] Loaded with sep='{s}'")
                return df_try
        except Exception:
            pass
    df_try = pd.read_csv(path, delim_whitespace=True, low_memory=False)
    print("[load] Loaded with whitespace separator")
    return df_try

def process(df, ticker, outdir):
    df.columns = [c.strip().replace("[","").replace("]","") for c in df.columns]
    for c in ["DTE","C_BID","C_ASK","P_BID","P_ASK","C_IV","P_IV","UNDERLYING_LAST","STRIKE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # IV 單位檢查
    if df["C_IV"].median() > 2: 
        df["C_IV"] = df["C_IV"] / 100.0
        df["P_IV"] = df["P_IV"] / 100.0

    df["call_mid"] = df.apply(lambda x: mid(x.get("C_BID"), x.get("C_ASK")), axis=1)
    df["put_mid"] = df.apply(lambda x: mid(x.get("P_BID"), x.get("P_ASK")), axis=1)
    df["T"] = df["DTE"] / 365.0
    df["parity_residual"] = df.apply(
        lambda x: parity_res(x.get("call_mid"), x.get("put_mid"), x.get("UNDERLYING_LAST"), x.get("STRIKE"), r, q, x.get("T")),
        axis=1
    )
    df["iv_gap"] = df["C_IV"] - df["P_IV"]

    # 品質濾鏡
    call_spread_frac = np.where(df["call_mid"]>0, (df["C_ASK"]-df["C_BID"])/df["call_mid"], np.inf)
    put_spread_frac  = np.where(df["put_mid"]>0,  (df["P_ASK"]-df["P_BID"])/df["put_mid"], np.inf)
    iv_ok = df["C_IV"].between(min_iv, max_iv) & df["P_IV"].between(min_iv, max_iv)
    df = df[
        (df["C_BID"]>0) & (df["C_ASK"]>0) &
        (df["P_BID"]>0) & (df["P_ASK"]>0) &
        (df["call_mid"]>0) & (df["put_mid"]>0) &
        (call_spread_frac<=max_spread_frac) & (put_spread_frac<=max_spread_frac) &
        iv_ok
    ]

    # moneyness 與短期過濾
    df["moneyness"] = df.apply(
        lambda x: (x["STRIKE"]/x["UNDERLYING_LAST"]) if pd.notna(x["STRIKE"]) and pd.notna(x["UNDERLYING_LAST"]) and x["UNDERLYING_LAST"]!=0 else np.nan,
        axis=1
    )
    df = df[df["DTE"].between(20,60) & df["moneyness"].between(0.9,1.2)]

    if df.empty:
        print("[warn] No data left after filters.")
        return

    df["side"] = np.where(df["STRIKE"]>df["UNDERLYING_LAST"],"K>S","K<S")
    df["score"] = df["parity_residual"].abs() + 0.5*df["iv_gap"].abs()*df["UNDERLYING_LAST"]*0.01
    df["label"] = np.where(
        (df["iv_gap"]>0)&(df["parity_residual"]>0),"S1_bull",
        np.where((df["iv_gap"]<0)&(df["parity_residual"]<0),"S1_bear","None")
    )

    df = df.sort_values("score",ascending=False).drop_duplicates(subset=["QUOTE_DATE","EXPIRE_DATE","STRIKE"],keep="first")

    out_date = str(df["QUOTE_DATE"].iloc[0]).strip().replace(" ","").replace("/","-")
    os.makedirs(outdir,exist_ok=True)
    out = os.path.join(outdir,f"signals_{out_date}.csv")
    df.to_csv(out,index=False)
    print(f"[save] {out} ({len(df)} rows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--ticker", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="signals")
    args = ap.parse_args()

    df_all = try_load(args.input)

    # 清理欄位名稱（移除方括號與前後空白），再確認 QUOTE_DATE 是否存在
    df_all.columns = [c.strip().replace("[", "").replace("]", "") for c in df_all.columns]
    if "QUOTE_DATE" not in df_all.columns:
        raise ValueError("Missing QUOTE_DATE column in input CSV")

    # clean possible spaces / stray types
    df_all["QUOTE_DATE"] = df_all["QUOTE_DATE"].astype(str).str.strip()

    # loop over each unique trading date
    for qd in sorted(df_all["QUOTE_DATE"].unique()):
        df_day = df_all[df_all["QUOTE_DATE"] == qd].copy()
        if len(df_day) < 10:
            continue
        print(f"\n=== processing {qd} ({len(df_day)} rows) ===")
        process(df_day, args.ticker, args.outdir)