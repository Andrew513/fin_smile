import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date

try:
    import yfinance as yf
except Exception:
    yf = None

# -----------------------------
# Defaults / thresholds (edit here)
# -----------------------------

# 0.35 -> 0.5 -> 0.80
LOW_ADTV_QUANTILE = 0.80  # "small" stock liquidity = bottom 50% ADTV

# Strict
# new strict
STRICT_MIN_OPT_VOL = 2500
STRICT_MIN_OI = 3000
STRICT_MIN_UNIQUE_STRIKES = 6
STRICT_MAX_MED_REL_SPREAD = 0.60
# STRICT_MIN_OPT_VOL = 5000
# STRICT_MIN_OI = 10000
# STRICT_MIN_UNIQUE_STRIKES = 10
# STRICT_MAX_MED_REL_SPREAD = 0.35

# Relaxed
#new relaxed
RELAX_MIN_OPT_VOL = 300
RELAX_MIN_OI = 200
RELAX_MIN_UNIQUE_STRIKES = 3
RELAX_MAX_MED_REL_SPREAD = 0.80
# RELAX_MIN_OPT_VOL = 1000
# RELAX_MIN_OI = 1000
# RELAX_MIN_UNIQUE_STRIKES = 4
# RELAX_MAX_MED_REL_SPREAD = 1.00

OUTPUT_DIR = Path("/mnt/data/smile_pipeline_outputs")

# Plotting/source controls
USE_FULL_CHAIN = True   # fetch full option chain (all strikes) from yfinance for plotting & metrics

# 6->4
MIN_POINTS_PER_SIDE = 4 # minimum unique strikes required per side to draw a reliable smile
MAX_IV_FOR_PLOTTING = 5.0  # ignore IV > 500% (chain-level safeguard; not used for plotting)

# Cleaning thresholds for plotting/smile
MIN_IV_PCT = 1.0       # drop IV < 1%

# 300 -> 350
MAX_IV_PCT = 350.0     # drop IV > 300%

# 0.60 -> 0.80
REL_SPREAD_MAX = 0.80  # drop quotes with relative spread > 0.60

# Unusual-activity detection thresholds (cross-sectional)
# 2.0 -> 1.5
ATM_IV_Z_MIN = 1.5       # z-score of ATM IV vs group

# 3.0 -> 2.0
VOL_RATIO_MIN = 2.0      # near-expiry option volume vs group mean
SPREAD_MAX_FOR_ALERT = 0.60

# -----------------------------
# Helpers
# -----------------------------
def safe_rel_spread(bid: float, ask: float) -> float:
    """Relative spread = (ask - bid) / mid; returns NaN when undefined."""
    try:
        if bid is None or ask is None:
            return np.nan
        bid = float(bid)
        ask = float(ask)
        if bid <= 0 and ask <= 0:
            return np.nan
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return np.nan
        return max(0.0, (ask - bid) / mid)
    except Exception:
        return np.nan

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Parse dates
    for col in ["earnings_date", "expiry"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    # Compute relative spread
    if not {"bid","ask"}.issubset(df.columns):
        raise ValueError("CSV缺少 bid/ask 欄位，無法計算相對價差")
    df["rel_spread"] = [safe_rel_spread(b,a) for b,a in zip(df["bid"], df["ask"])]
    return df

def nearest_expiry_for_ticker(df_ticker: pd.DataFrame) -> date:
    """Pick the nearest future expiry (>= today). Fall back to minimum expiry if none future."""
    expiries = pd.to_datetime(df_ticker["expiry"], errors="coerce").dropna().dt.date.unique()
    if len(expiries) == 0:
        return None
    today = date.today()
    future = [e for e in expiries if e >= today]
    if future:
        return min(future)
    return min(expiries)

def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Per-ticker base metrics from full set
    agg_all = (
        df.groupby("ticker")
          .agg(
              adtv_10d=("adtv_10d", "max"),  # ADTV is repeated per row; use max/same
              total_opt_vol=("volume", "sum"),
              sum_oi=("openInterest", "sum"),
              unique_strikes=("strike", "nunique"),
              med_rel_spread=("rel_spread", "median"),
              ivxvol_sum=("iv_x_vol", "sum"),
          )
          .reset_index()
    )

    # Nearest-expiry metrics
    rows = []
    for tkr, g in df.groupby("ticker"):
        ne = nearest_expiry_for_ticker(g)
        if ne is None:
            continue
        near = g[g["expiry"] == ne]
        rows.append({
            "ticker": tkr,
            "nearest_expiry": ne,
            "unique_strikes_near": near["strike"].nunique(),
            "opt_vol_near": near["volume"].sum(),
            "oi_near": near["openInterest"].sum(),
            "med_rel_spread_near": near["rel_spread"].median(),
        })
    agg_near = pd.DataFrame(rows)

    # Merge
    metrics = pd.merge(agg_all, agg_near, on="ticker", how="left")

    # Compute ADTV cutoffs (quantile-based)
    if metrics["adtv_10d"].notna().any():
        low_adtv_cut = metrics["adtv_10d"].quantile(LOW_ADTV_QUANTILE)
    else:
        low_adtv_cut = np.nan
    metrics["low_adtv_flag"] = metrics["adtv_10d"] <= low_adtv_cut
    metrics["low_adtv_cut"] = low_adtv_cut

    return metrics

def screen_candidates(metrics: pd.DataFrame):
    strict_mask = (
        metrics["low_adtv_flag"].fillna(False)
        & (metrics["opt_vol_near"].fillna(0) >= STRICT_MIN_OPT_VOL)
        & (metrics["oi_near"].fillna(0) >= STRICT_MIN_OI)
        & (metrics["unique_strikes_near"].fillna(0) >= STRICT_MIN_UNIQUE_STRIKES)
        & (metrics["med_rel_spread_near"].fillna(1e9) <= STRICT_MAX_MED_REL_SPREAD)
    )
    relaxed_mask = (
        metrics["low_adtv_flag"].fillna(False)
        & (metrics["opt_vol_near"].fillna(0) >= RELAX_MIN_OPT_VOL)
        & (metrics["oi_near"].fillna(0) >= RELAX_MIN_OI)
        & (metrics["unique_strikes_near"].fillna(0) >= RELAX_MIN_UNIQUE_STRIKES)
        & (metrics["med_rel_spread_near"].fillna(1e9) <= RELAX_MAX_MED_REL_SPREAD)
    )
    strict = metrics.loc[strict_mask].copy()
    relaxed = metrics.loc[relaxed_mask].copy()
    return strict, relaxed

def fetch_full_chain_df(tkr: str, expiry: date) -> pd.DataFrame:
    """Fetch full option chain for ticker/expiry via yfinance; return columns: type, strike, iv_pct, volume, openInterest, rel_spread.
    Returns empty DataFrame on error or if yfinance missing.
    """
    if yf is None:
        return pd.DataFrame(columns=["type","strike","iv_pct","volume","openInterest","rel_spread"])
    try:
        tk = yf.Ticker(tkr)
        expiry_str = pd.Timestamp(expiry).strftime("%Y-%m-%d")
        chain = tk.option_chain(expiry_str)
        calls = chain.calls.copy(); puts = chain.puts.copy()
        calls["type"] = "call"; puts["type"] = "put"
        # unify columns
        use_cols = [
            "type","strike","impliedVolatility","volume","openInterest","bid","ask"
        ]
        calls = calls[[c for c in use_cols if c in calls.columns]]
        puts  = puts[[c for c in use_cols if c in puts.columns]]
        df_full = pd.concat([calls, puts], ignore_index=True)
        # basic validity
        df_full = df_full.dropna(subset=["strike","impliedVolatility"]).copy()
        # compute rel spread when possible
        if {"bid","ask"}.issubset(df_full.columns):
            mid = (df_full["bid"].astype(float) + df_full["ask"].astype(float)) / 2.0
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = (df_full["ask"].astype(float) - df_full["bid"].astype(float)) / mid.replace(0, np.nan)
            df_full["rel_spread"] = rel.clip(lower=0)
        else:
            df_full["rel_spread"] = np.nan
        # compute iv_pct
        df_full["iv_pct"] = df_full["impliedVolatility"].astype(float) * 100.0
        # ensure liquidity fields exist
        for c in ["volume","openInterest"]:
            if c not in df_full.columns:
                df_full[c] = 0
        # apply cleaning
        df_full = df_full[
            (df_full["iv_pct"] >= MIN_IV_PCT) & (df_full["iv_pct"] <= MAX_IV_PCT)
        ]
        df_full = df_full[(df_full["volume"].fillna(0) > 0) | (df_full["openInterest"].fillna(0) > 0)]
        df_full = df_full[(df_full["rel_spread"].isna()) | (df_full["rel_spread"] <= REL_SPREAD_MAX)]
        return df_full[["type","strike","iv_pct","volume","openInterest","rel_spread"]]
    except Exception:
        return pd.DataFrame(columns=["type","strike","iv_pct","volume","openInterest","rel_spread"])  # graceful fallback


def get_spot_price(tkr: str) -> float:
    """Get latest spot price via yfinance (fast_info if available)."""
    if yf is None:
        return np.nan
    try:
        tk = yf.Ticker(tkr)
        px = None
        try:
            px = tk.fast_info.last_price
        except Exception:
            pass
        if px is None or (isinstance(px, float) and not np.isfinite(px)):
            hist = tk.history(period="1d")
            if not hist.empty:
                px = float(hist["Close"].iloc[-1])
        return float(px) if px is not None else np.nan
    except Exception:
        return np.nan


def compute_atm_iv_from_chain(tkr: str, expiry: date) -> float:
    """Estimate ATM IV (percent) by averaging call/put IV at strike closest to spot."""
    if yf is None:
        return np.nan
    try:
        spot = get_spot_price(tkr)
        if not np.isfinite(spot):
            return np.nan
        tk = yf.Ticker(tkr)
        expiry_str = pd.Timestamp(expiry).strftime("%Y-%m-%d")
        chain = tk.option_chain(expiry_str)
        calls = chain.calls.dropna(subset=["strike","impliedVolatility"]).copy()
        puts = chain.puts.dropna(subset=["strike","impliedVolatility"]).copy()
        calls["dist"] = (calls["strike"] - spot).abs()
        puts["dist"] = (puts["strike"] - spot).abs()
        c_iv = float(calls.sort_values("dist").iloc[0]["impliedVolatility"]) if not calls.empty else np.nan
        p_iv = float(puts.sort_values("dist").iloc[0]["impliedVolatility"]) if not puts.empty else np.nan
        ivs = [v for v in [c_iv, p_iv] if np.isfinite(v) and 0 < v < MAX_IV_FOR_PLOTTING]
        if not ivs:
            return np.nan
        return np.mean(ivs) * 100.0
    except Exception:
        return np.nan



# Utility to ensure rel_spread column exists for CSV-subset data
def ensure_rel_spread_cols(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    if "rel_spread" in df.columns:
        return df
    if {"bid","ask"}.issubset(df.columns):
        mid = (df["bid"].astype(float) + df["ask"].astype(float)) / 2.0
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = (df["ask"].astype(float) - df["bid"].astype(float)) / mid.replace(0, np.nan)
        df["rel_spread"] = rel.clip(lower=0)
    else:
        df["rel_spread"] = np.nan
    return df

def prep_smile_points(df_src: pd.DataFrame) -> pd.DataFrame:
    """Group by type/strike and average IV for plotting. Assumes columns type/strike/iv_pct present."""
    if df_src.empty:
        return df_src

    df = df_src.copy()
    # Normalize type values and ensure strike is numeric
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip().str.lower()
    else:
        df["type"] = ""
    # coerce strike to float and drop non-numeric
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df = df.dropna(subset=["strike", "iv_pct"]).copy()

    # volume-weighted when available
    has_vol = "volume" in df.columns
    if has_vol:
        df["_w"] = df["volume"].fillna(0).clip(lower=1)
        piv = (
            df.groupby(["type", "strike"], as_index=False)
              .apply(lambda g: pd.Series({"iv_pct": np.average(g["iv_pct"], weights=g["_w"]) }))
              .reset_index()
              .sort_values(["type", "strike"])
        )
        # after reset_index, apply yields columns ['type','strike',0?,'iv_pct'] depending on pandas; normalize
        if "iv_pct" not in piv.columns and piv.shape[1] >= 3:
            piv.columns = ["type","strike","iv_pct"]
    else:
        piv = (
            df.groupby(["type", "strike"], as_index=False)
              .agg(iv_pct=("iv_pct", "mean"))
              .sort_values(["type", "strike"])
        )

    # final normalization
    piv["type"] = piv["type"].astype(str).str.strip().str.lower()
    return piv

def plot_smile(df_all: pd.DataFrame, tkr: str, expiry: date, outdir: Path):
    """Draw IV smile (cleaned). If USE_FULL_CHAIN and yfinance available, use full chain; otherwise fall back to CSV subset."""
    if USE_FULL_CHAIN:
        df_plot = fetch_full_chain_df(tkr, expiry)
        if df_plot.empty:
            sub = df_all[(df_all["ticker"]==tkr) & (df_all["expiry"]==expiry)].copy()
            df_plot = sub.copy() if not sub.empty else pd.DataFrame()
    else:
        sub = df_all[(df_all["ticker"]==tkr) & (df_all["expiry"]==expiry)].copy()
        df_plot = sub.copy() if not sub.empty else pd.DataFrame()

    if df_plot.empty:
        return None
    # If coming from CSV, compute rel_spread when missing
    df_plot = ensure_rel_spread_cols(df_plot)
    # keep liquidity columns if missing
    for c in ["volume","openInterest"]:
        if c not in df_plot.columns:
            df_plot[c] = 0
    # Clean: IV range, liquidity, spread
    df_plot = df_plot.dropna(subset=["type","strike","iv_pct"]).copy()
    df_plot = df_plot[(df_plot["iv_pct"] >= MIN_IV_PCT) & (df_plot["iv_pct"] <= MAX_IV_PCT)]
    df_plot = df_plot[(df_plot["volume"].fillna(0) > 0) | (df_plot["openInterest"].fillna(0) > 0)]
    df_plot = df_plot[(df_plot["rel_spread"].isna()) | (df_plot["rel_spread"] <= REL_SPREAD_MAX)]
    # df_plot = df_plot[(df_plot["iv_pct"]>0) & (df_plot["iv_pct"]<MAX_IV_FOR_PLOTTING*100)]
    piv = prep_smile_points(df_plot)

    # Draw current spot as a vertical dashed line for reference
    spot = get_spot_price(tkr)
    spot_label = None
    if np.isfinite(spot):
        spot_label = f"SPOT {spot:.2f}"

    # Normalize pivot 'type' and ensure strike numeric (defensive)
    piv["type"] = piv["type"].astype(str).str.strip().str.lower()
    piv["strike"] = pd.to_numeric(piv["strike"], errors="coerce")

    # Minimum points per side
    pts_call = piv[piv["type"]=="call"]["strike"].nunique()
    pts_put  = piv[piv["type"]=="put"]["strike"].nunique()
    if pts_call < MIN_POINTS_PER_SIDE and pts_put < MIN_POINTS_PER_SIDE:
        return None

    # Debug summary to help identify swapped sides / bad data
    try:
        for st in ["put","call"]:
            s = piv[piv["type"]==st]
            if not s.empty:
                print(f"{tkr} {expiry} {st}: count={s['strike'].nunique()}, min={s['strike'].min():.2f}, max={s['strike'].max():.2f}")
    except Exception:
        pass

    plt.figure(figsize=(9,6))
    if spot_label is not None:
        plt.axvline(spot, linestyle='--', alpha=0.6, label=spot_label)

    # Explicit plotting order and colors: put (left) then call (right)
    colors = {"put": "C0", "call": "C1"}
    for side_type in ["put","call"]:
        side = piv[piv["type"]==side_type].sort_values("strike")
        if side["strike"].nunique() >= max(3, MIN_POINTS_PER_SIDE//2):
            plt.plot(side["strike"], side["iv_pct"], label=side_type.upper(), color=colors.get(side_type))

    plt.title(f"{tkr} IV Smile ({expiry})")
    plt.xlabel("Strike (vertical dashed = spot)")
    plt.ylabel("IV (%)")
    plt.legend()
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"smile_{tkr}_{expiry}.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()
    return outfile

def run(csv_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    # Ensure expected columns exist
    required = {"ticker","expiry","type","strike","iv_pct","volume","openInterest","adtv_10d","iv_x_vol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少欄位: {missing}")

    # Build metrics
    metrics = aggregate_metrics(df)

    # Compute ATM IV from full chain (if available) for each ticker's nearest expiry
    atm_rows = []
    for _, r in metrics.iterrows():
        tkr = r["ticker"]
        ne = r["nearest_expiry"] if "nearest_expiry" in r else None
        if pd.isna(ne):
            atm_iv = np.nan
        else:
            atm_iv = compute_atm_iv_from_chain(tkr, ne) if USE_FULL_CHAIN else np.nan
        atm_rows.append({"ticker": tkr, "atm_iv_pct": atm_iv})
    atm_df = pd.DataFrame(atm_rows)
    metrics = metrics.merge(atm_df, on="ticker", how="left")

    # Cross-sectional baselines (group by theme if present; else group = 'all')
    if "theme" in df.columns:
        tmap = (
            df[["ticker","theme"]]
              .drop_duplicates("ticker")
              .set_index("ticker")["theme"]
              .to_dict()
        )
        metrics["_grp"] = metrics["ticker"].map(tmap).fillna("all")
    else:
        metrics["_grp"] = "all"

    grp_stats = metrics.groupby("_grp").agg(
        mean_atm_iv=("atm_iv_pct","mean"),
        std_atm_iv=("atm_iv_pct","std"),
        mean_optvol_near=("opt_vol_near","mean")
    ).reset_index()

    metrics = metrics.merge(grp_stats, on="_grp", how="left")

    # Z-scores / ratios with safe denominators
    metrics["std_atm_iv"] = metrics["std_atm_iv"].replace(0, np.nan)
    metrics["atm_iv_z"] = (metrics["atm_iv_pct"] - metrics["mean_atm_iv"]) / metrics["std_atm_iv"]
    metrics["vol_ratio"] = metrics["opt_vol_near"] / metrics["mean_optvol_near"].replace(0, np.nan)

    # Suspicious flag: low ADTV + IV up + volume surge + spreads not too wide
    metrics["suspicious_flag"] = (
        metrics["low_adtv_flag"].fillna(False)
        & (metrics["atm_iv_z"] >= ATM_IV_Z_MIN)
        & (metrics["vol_ratio"] >= VOL_RATIO_MIN)
        & (metrics["med_rel_spread_near"].fillna(1e9) <= SPREAD_MAX_FOR_ALERT)
    )

    # Save per-ticker metrics
    metrics_path = output_dir / "ticker_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    # Screen strict then relaxed
    strict, relaxed = screen_candidates(metrics)

    strict_path = output_dir / "candidates_strict.csv"
    relaxed_path = output_dir / "candidates_relaxed.csv"
    strict.to_csv(strict_path, index=False)
    relaxed.to_csv(relaxed_path, index=False)

    suspicious_path = output_dir / "suspicious_unusual_activity.csv"
    metrics.loc[metrics["suspicious_flag"]].to_csv(suspicious_path, index=False)

    # Decide which set to plot
    chosen = strict if len(strict) > 0 else relaxed
    smiles_dir = output_dir / ("smiles_strict" if len(strict) > 0 else "smiles_relaxed")
    generated = []

    for _, row in chosen.iterrows():
        tkr = row["ticker"]
        expiry = row["nearest_expiry"]
        if pd.isna(expiry):
            continue
        out = plot_smile(df, tkr, expiry, smiles_dir)
        if out is not None:
            generated.append(str(out))

    # manifest
    pd.Series(generated, name="smile_files").to_csv(output_dir / "smile_manifest.csv", index=False)

    print(f"Saved metrics to: {metrics_path}")
    print(f"Strict candidates: {strict_path} ({len(strict)})")
    print(f"Relaxed candidates: {relaxed_path} ({len(relaxed)})")
    print(f"Smile images in: {smiles_dir} (count={len(generated)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="/iv_earnings_candidates_themes.csv",
                    help="Path to iv_earnings_candidates_themes.csv")
    ap.add_argument("--outdir", type=str, default=str(OUTPUT_DIR),
                    help="Output root directory")
    ap.add_argument("--no-full-chain", action="store_true", help="Do not fetch full option chain; use CSV subset only")
    args = ap.parse_args()
    if args.no_full_chain:
        USE_FULL_CHAIN = False
    run(Path(args.csv), Path(args.outdir))
