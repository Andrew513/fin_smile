import pandas as pd
import numpy as np
import math, datetime as dt

# === 參數設定 ===
r = 0.04   # 無風險利率
q = 0.00   # 股利率，可先假設 0
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
    # 若任何價格為 NaN，回傳 NaN
    if pd.isna(call_mid) or pd.isna(put_mid) or pd.isna(S) or pd.isna(K) or pd.isna(T):
        return np.nan
    return (call_mid - put_mid) - (S * math.exp(-q * T) - K * math.exp(-r * T))

# === 讀取資料 ===
file_path = "/Users/andrew/Downloads/SideProjects/finance/tool/data/aapl_2016_2020.csv"

def try_load(path):
    # 嘗試幾種常見分隔符，若都失敗回傳最後一次嘗試的 DataFrame
    seps = ['\t', ',', ';', '|']
    for s in seps:
        try:
            df_try = pd.read_csv(path, sep=s)
            # 若看起來有多個欄位就接受
            if len(df_try.columns) > 1:
                print(f"Loaded with sep='{s}'")
                return df_try
        except Exception:
            pass
    # 最後嘗試用空白分隔
    try:
        df_try = pd.read_csv(path, delim_whitespace=True)
        print("Loaded with delim_whitespace=True")
        return df_try
    except Exception:
        # 單一欄位回傳原始讀取（raise later）
        return pd.read_csv(path, sep='\t', engine='python')

df = try_load(file_path)

# 清理欄位名稱（移除方括號與前後空白）
df.columns = [c.strip().replace("[", "").replace("]", "") for c in df.columns]
print("Columns after load (cleaned):", list(df.columns))

# 將關鍵欄位轉為數值型，避免字串造成的運算錯誤
num_cols = ["DTE", "C_BID", "C_ASK", "P_BID", "P_ASK", "C_IV", "P_IV", "UNDERLYING_LAST", "STRIKE"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# 計算 mid 價（使用清理後的欄位名稱，例如 C_BID, C_ASK, P_BID, P_ASK）
df["call_mid"] = df.apply(lambda x: mid(x.get("C_BID"), x.get("C_ASK")), axis=1)
df["put_mid"] = df.apply(lambda x: mid(x.get("P_BID"), x.get("P_ASK")), axis=1)

# 年化時間
df["T"] = df["DTE"] / 365.0

# parity 殘差
df["parity_residual"] = df.apply(
    lambda x: parity_res(x.get("call_mid"), x.get("put_mid"), x.get("UNDERLYING_LAST"), x.get("STRIKE"), r, q, x.get("T")),
    axis=1
)

# IV 差
df["iv_gap"] = df.get("C_IV") - df.get("P_IV")

# 清理不合理值：
# - 要求原始 bid/ask 都存在且 > 0
# - 要求 mid > 0
# - 要求 bid/ask spread 除以 mid 小於等於 max_spread_frac
# - 要求隱含波動率在合理區間
mask = (
    (df.get("C_BID").fillna(0) > 0) &
    (df.get("C_ASK").fillna(0) > 0) &
    (df.get("P_BID").fillna(0) > 0) &
    (df.get("P_ASK").fillna(0) > 0) &
    (df["call_mid"].fillna(0) > 0) &
    (df["put_mid"].fillna(0) > 0)
)

# spread fraction guards（避免除以 0）
call_spread_frac = np.where(df["call_mid"] > 0, (df.get("C_ASK") - df.get("C_BID")) / df["call_mid"], np.inf)
put_spread_frac = np.where(df["put_mid"] > 0, (df.get("P_ASK") - df.get("P_BID")) / df["put_mid"], np.inf)

iv_ok = df.get("C_IV").between(min_iv, max_iv) & df.get("P_IV").between(min_iv, max_iv)

df = df[mask & (call_spread_frac <= max_spread_frac) & (put_spread_frac <= max_spread_frac) & iv_ok]

# moneyness (K/S) 若底層價格為 0 或 NaN，則產生 NaN
df["moneyness"] = df.apply(lambda x: (x.get("STRIKE") / x.get("UNDERLYING_LAST")) if (pd.notna(x.get("STRIKE")) and pd.notna(x.get("UNDERLYING_LAST")) and x.get("UNDERLYING_LAST") != 0) else np.nan, axis=1)

# === 打分與訊號 ===
df["score"] = df["parity_residual"].abs() + 0.5 * df["iv_gap"].abs() * df["UNDERLYING_LAST"] * 0.01
df["label"] = np.where(
    (df["iv_gap"] > 0) & (df["parity_residual"] > 0), "S1_bull",
    np.where((df["iv_gap"] < 0) & (df["parity_residual"] < 0), "S1_bear", "None")
)

# === 顯示 Top 10 異常 ===
df = df.sort_values("score", ascending=False)
cols = ["QUOTE_DATE", "EXPIRE_DATE", "DTE", "STRIKE", "UNDERLYING_LAST", "call_mid", "put_mid",
        "C_IV", "P_IV", "iv_gap", "parity_residual", "score", "label"]

print(df[cols].head(10))

# 儲存結果供回測（若資料為空則提示）
if df.empty:
    print("No rows passed the filters — no signals saved.")
else:
    raw_date = str(df['QUOTE_DATE'].iloc[0])
    date_str = raw_date.strip().replace(' ', '').replace(':', '').replace('/', '-')
    out = f"signals_{date_str}.csv"
    df.to_csv(out, index=False)
    print(f"Saved signals to {out}")