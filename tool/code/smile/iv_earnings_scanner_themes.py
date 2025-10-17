#!/usr/bin/env python3
# iv_earnings_scanner_themes.py (expanded THEMES: >=50 tickers per theme)
# Usage:
#   pip install yfinance pandas matplotlib
#   python iv_earnings_scanner_themes.py --themes ai,chips,defense,nuclear,mining

import argparse
import datetime as dt
import math
from typing import Dict, Optional, Tuple, List, Set

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo
from tabulate import tabulate

# ---------------------- THEME UNIVERSE (>=50 each) ----------------------
THEMES = {
    "ai": [
        # Core AI / Infra / Cloud / Data / Security / Consumer AI
        "NVDA","MSFT","GOOGL","META","AMZN","SNOW","DDOG","MDB","CRWD","NOW",
        "PLTR","AI","PATH","ZS","NET","SMCI","ORCL","SAP","ADBE","INTU",
        "TEAM","HUBS","SHOP","OKTA","U","ESTC","DT","PANW","FTNT","S",
        "BRZE","FROG","NEWR","AKAM","TWLO","BILL","APP","TTD","RBLX","ABNB",
        "IONQ","BBAI","SOUN","UPST","DOCN","AYX","VRNS","WDAY","BOX","NICE",
        "DOCU","VERX","PD","EE","INFA","CFLT","COUR","DUOL","GLBE","AFRM",
        "ROKU","OKTA","ZSAN","NET","OKTA",  # 重點雲端/安全重複者已去重，這行留 OKTA/NET 覆蓋率
        # 來自你的財報清單（屬 AI/雲/資料中心）
        "NFLX","IBM","VRT","DLR","FFIV", "INTC", "EQIX", "QBTS"
    ][:70],

    "chips": [
        # Semiconductors & equipment
        "NVDA","AMD","TSM","AVGO","ASML","MU","INTC","QCOM","NXPI","TXN",
        "LRCX","AMAT","KLAC","MRVL","ADI","ON","ARM","GFS","WOLF","MCHP",
        "MPWR","MTSI","DIOD","SYNA","TER","COHR","ACLS","AEHR","ICHR","CAMT",
        "NVTS","LSCC","SITM","SLAB","SMTC","QRVO","SWKS","SKYT","UMC","ASX",
        "AMKR","ENTG","TOELY","UCTT","FORM","VECO","POWI","RMBS","ALGM","STM",
        "IFNNY","MXL","IMOS","HIMX","ROHCY","CRUS","KLIC","TTMI","SANM","SGH",
        "AAOI","KN","AVGO","ON","MRVL",    # 加強主流覆蓋；去重由 set 處理
        # 來自你的財報清單（屬晶片/EDA/車用）
        "CDNS","MBLY","APH"
    ][:70],

    "defense": [
        # Defense / Aerospace / Military Systems（移除 CWCO、OCN、ATEN、ATEC、FREY 等非軍工）
        "LMT","NOC","RTX","GD","HII","LHX","TDG","HEI","TXT","BWXT",
        "KTOS","AVAV","AXON","BA","CW","MRCY","TDY","SAIC","LDOS","BAH",
        "HXL","SPR","VVX","KBR","CACI","PSN","PLTR","ACHR","JOBY","RKLB",
        "SPCE","SPIR","SATL","IRDM","VSAT","TGI","DRS","AIR","MOG-A","MOG-B",
        "ATRO","HWM","CWAN","AER","COLL","WLDN","RGR","SWBI","MNTX","CVU",
        # 你的財報清單中軍工/航太/工業大廠
        "GE","HON"
    ][:70],

    "nuclear": [
        # Uranium / fuel cycle / US utilities with nuclear / SMR
        "CCJ","UEC","UUUU","DNN","UROY","NXE","URG","LEU","SMR","BWXT",
        "GEV","NRG","CEG","EXC","PEG","DUK","SO","NEE","EIX","AEP",
        "ED","XEL","PCG","PPL","FE","SRE","VST","LTBR","NRG","GNRC",
        "IDR","NEP","AES","DTE","CMS","PGE","CNP","EVRG","NI","OGE",
        "ORA","PNW","WEC","AEE","AEP","ETR","D","AEP","AES","ED",
        # 去掉外盤/難用後綴（KAP.L、RR.L、AREVA.PA、SHEC.F 等）與非核（TELL、EVGN）
        # 你的清單中對應美股核能/公用事業
        "DTE", "OKLO", "CCJ"
    ][:70],

    "mining": [
        # Mining / Metals / Battery materials（保留美股主要市場/常用 ADR；去重）
        "BHP","RIO","VALE","FCX","TECK","AA","CLF","SCCO","GOLD","NEM",
        "WPM","FNV","AEM","HL","PAAS","AG","CDE","SAND","KGC","BTG",
        "AU","SBSW","HBM","IAG","EGO","EQX","GFI","NGD","SA","GORO",
        "TMC","CMP","MOS","NTR","LAC","ALB","SQM","LTHM","PLL","SGML",
        "LAAC","IVPAF","MAG","FSM","EXK","ASM","CGAU","ORLA","HAYN","NC",
        "MT","X","ATI","SID","GGB","CMC","RS","STLD","NUE","TS",
        # 你的策略觀察對象
        "CRS","MP","CRML","USAR"
    ][:70],
}
# THEMES = {
#     "ai": ["AMZN", "NFLX"], "chips": ["NVDA"], "defense": ["LMT"], "nuclear": ["CCJ"], "mining": ["BHP"]
# }
# ---------------------- SETTINGS ----------------------
MAX_STOCK_ADTV = 2_000_000
NEAREST_N_EXPIRIES = 2
MIN_OPTION_VOLUME = 300
MIN_IV = 0.45
MIN_IV_X_VOL = 120.0
OUT_CSV = "iv_earnings_candidates_themes.csv"

# ---------------------- HELPERS ----------------------
def next_week_window(today: Optional[dt.date] = None, tz: str = "America/Chicago", days: int = 7) -> Tuple[dt.datetime, dt.datetime]:
    """Return [start, end) window starting tomorrow for the next `days` days in given timezone.
    Times returned are timezone-naive UTC equivalents for comparison.
    """
    if today is None:
        # Use current date in the specified timezone
        now_local = dt.datetime.now(ZoneInfo(tz))
        today = now_local.date()
    # Define window as the next `days` calendar days starting tomorrow (local)
    start_local = dt.datetime.combine(today + dt.timedelta(days=1), dt.time.min, tzinfo=ZoneInfo(tz))
    end_local = start_local + dt.timedelta(days=days)
    # Convert to UTC and drop tzinfo (naive UTC for comparisons with yfinance naive timestamps)
    start_utc_naive = (start_local.astimezone(ZoneInfo("UTC"))).replace(tzinfo=None)
    end_utc_naive = (end_local.astimezone(ZoneInfo("UTC"))).replace(tzinfo=None)
    return start_utc_naive, end_utc_naive

def get_next_earnings_date(ticker: str) -> Optional[pd.Timestamp]:
    """Best-effort retrieval of the next earnings date using multiple yfinance fallbacks.
    Returns a timezone-naive UTC timestamp if found, else None.
    """
    try:
        t = yf.Ticker(ticker)
        now = pd.Timestamp.utcnow().tz_localize(None)

        # 1) Primary: get_earnings_dates (try a bigger limit)
        try:
            df = t.get_earnings_dates(limit=30)
            if df is not None and not df.empty:
                idx = df.index
                # Some yfinance versions return tz-aware index; normalize to naive UTC
                if hasattr(idx, "tz") and idx.tz is not None:
                    idx = idx.tz_convert("UTC").tz_localize(None)
                future = idx[idx >= now]
                if len(future) > 0:
                    return future.min()
        except Exception:
            pass

        # 2) Fallback: t.calendar or t.get_calendar() variations
        # Different yfinance versions expose calendar in different shapes
        try:
            cal = None
            if hasattr(t, "calendar"):
                cal = t.calendar
            if cal is None and hasattr(t, "get_calendar"):
                try:
                    cal = t.get_calendar()
                except Exception:
                    cal = None
            if cal is not None and not getattr(cal, "empty", False):
                # Common shapes contain an "Earnings Date" row or column
                # Try to extract any datetime-like value that is in the future
                possible = []
                # yfinance can return a DataFrame/Series-like object, or a plain dict.
                # Handle dict case first (common): {'Earnings Date': [date, ...], ...}
                if isinstance(cal, dict):
                    if "Earnings Date" in cal:
                        vals = cal.get("Earnings Date") or []
                        # vals might be a single value or a list
                        if isinstance(vals, (list, tuple)):
                            possible.extend(vals)
                        else:
                            possible.append(vals)
                else:
                    # If columns/index contain 'Earnings Date'
                    try:
                        if "Earnings Date" in cal.index:
                            vals = cal.loc["Earnings Date"].values.tolist()
                            possible.extend(vals)
                    except Exception:
                        pass
                    try:
                        if "Earnings Date" in cal.columns:
                            vals = cal["Earnings Date"].values.tolist()
                            possible.extend(vals)
                    except Exception:
                        pass

                # Flatten and parse any candidate values into Timestamps (naive UTC)
                ts = []
                for v in possible:
                    if isinstance(v, (list, tuple)):
                        for sub in v:
                            try:
                                tstamp = pd.to_datetime(sub, utc=True)
                                # convert to tz-naive UTC
                                tstamp = tstamp.tz_convert("UTC").tz_localize(None) if getattr(tstamp, 'tz', None) is not None else tstamp
                                ts.append(tstamp)
                            except Exception:
                                continue
                    else:
                        try:
                            tstamp = pd.to_datetime(v, utc=True)
                            tstamp = tstamp.tz_convert("UTC").tz_localize(None) if getattr(tstamp, 'tz', None) is not None else tstamp
                            ts.append(tstamp)
                        except Exception:
                            continue

                ts = [x for x in ts if isinstance(x, pd.Timestamp) and x >= now]
                if ts:
                    return min(ts)
        except Exception:
            pass

        # 3) Nothing found
        return None
    except Exception as e:
        print(f"[WARN] {ticker}: earnings fetch failed: {e}")
        return None

def avg_volume_10d(ticker: str) -> Optional[float]:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        vol = info.get("averageVolume10days", None)
        if vol is None:
            hist = t.history(period="1mo", interval="1d")
            if hist is None or hist.empty: return None
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
    last_err = None
    for _ in range(3):
        try:
            t = yf.Ticker(ticker)
            oc = t.option_chain(expiry)
            calls = oc.calls.copy(); calls["type"]="call"
            puts  = oc.puts.copy();  puts["type"]="put"
            df = pd.concat([calls, puts], ignore_index=True, sort=False)
            keep = ["strike","type","impliedVolatility","volume","lastPrice","bid","ask","openInterest"]
            return df[keep].dropna(subset=["impliedVolatility"])
        except Exception as e:
            last_err = e
    print(f"[WARN] {ticker}: option chain fetch failed for {expiry}: {last_err}")
    return None

def summarize_options(df: pd.DataFrame, min_option_volume: int, min_iv: float, min_iv_x_vol: float) -> pd.DataFrame:
    df = df.copy()
    df["iv_x_vol"] = df["impliedVolatility"] * df["volume"].astype(float)
    df["pass_filters"] = (
        (df["volume"] >= min_option_volume) &
        (df["impliedVolatility"] >= min_iv) &
        (df["iv_x_vol"] >= min_iv_x_vol)
    )
    return df.sort_values("iv_x_vol", ascending=False)



# ---------------------- HELPERS ----------------------

def build_universe_with_map(theme_names: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Build the ticker universe from the requested themes and also return a
    mapping from ticker -> primary theme (based on the first theme in the
    user's theme order that contains the ticker). Duplicates are de-duped.
    """
    seen: Set[str] = set()
    universe: List[str] = []
    tkr2theme: Dict[str, str] = {}
    for name in theme_names:
        key = name.strip().lower()
        if key not in THEMES:
            print(f"[WARN] theme '{name}' not recognized. Available: {', '.join(THEMES.keys())}")
            continue
        for raw in THEMES[key]:
            tkr = raw.strip().upper()
            if not tkr:
                continue
            if tkr not in seen:
                seen.add(tkr)
                universe.append(tkr)
                # first theme wins for this ticker (user-specified order)
                tkr2theme.setdefault(tkr, name.strip().lower())
            else:
                # already present; do not overwrite earlier theme choice
                pass
    return universe, tkr2theme

# ---------------------- MAIN ----------------------
def main():
    parser = argparse.ArgumentParser(description="Scan themed universe for next-week earnings + options IV/volume")
    parser.add_argument("--themes", type=str, default="ai,chips,defense,nuclear,mining")
    parser.add_argument("--max-stock-adtv", type=int, default=MAX_STOCK_ADTV, help="10D average shares/day upper bound; set 0 to disable volume cap")
    parser.add_argument("--min-option-volume", type=int, default=MIN_OPTION_VOLUME)
    parser.add_argument("--min-iv", type=float, default=MIN_IV)
    parser.add_argument("--min-ivxvol", type=float, default=MIN_IV_X_VOL)
    parser.add_argument("--expiries", type=int, default=NEAREST_N_EXPIRIES)
    parser.add_argument("--per-expiry-top", type=int, default=0, help="take top N contracts per expiry by IVxVol; 0 = take all after filters")
    parser.add_argument("--per-ticker-top", type=int, default=0, help="take top N contracts per ticker across expiries; 0 = take all after filters")
    parser.add_argument("--min-oi", type=int, default=0, help="minimum open interest per contract; 0 = disable")
    parser.add_argument("--only-pass", action="store_true", help="only include contracts that pass thresholds (volume/IV/IVxVol)")
    parser.add_argument("--exclude-earnings", action="store_true", help="exclude tickers that have earnings within the lookahead window (use with --days)")
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--days", type=int, default=7, help="lookahead days starting tomorrow (local tz)")
    parser.add_argument("--tz", type=str, default="America/Chicago", help="IANA timezone for earnings window")
    args = parser.parse_args()

    max_stock_adtv = args.max_stock_adtv
    min_option_volume = args.min_option_volume
    min_iv = args.min_iv
    min_iv_x_vol = args.min_ivxvol
    expiries_count = args.expiries
    make_plots = args.make_plots
    per_expiry_top = args.per_expiry_top
    per_ticker_top = args.per_ticker_top
    min_oi = args.min_oi
    only_pass = args.only_pass
    exclude_earnings = args.exclude_earnings

    themes = [t.strip() for t in args.themes.split(",") if t.strip()]
    tickers, tkr2theme = build_universe_with_map(themes)
    if not tickers:
        print("No tickers in universe. Check your --themes parameter."); return

    start, end = next_week_window(tz=args.tz, days=args.days)
    print(f"Themes: {themes}")
    print(f"Universe size: {len(tickers)} tickers")
    print(f"Scanning earnings window (local {args.tz}): {start.date()} -> {end.date()} ({args.days} days)")
    print(f"Equity ADTV <= {max_stock_adtv:,} (0=disable) | Option filters: Vol>={min_option_volume}, IV>={min_iv:.0%}, IVxVol>={min_iv_x_vol}, OI>={min_oi} | per-expiry-top={per_expiry_top}, per-ticker-top={per_ticker_top}")

    rows = []

    for tk in tickers:
        print(f"\n=== {tk} ===")
        ed = get_next_earnings_date(tk)
        if ed is None:
            print("  No upcoming earnings date found."); continue
        ed_naive = ed.tz_localize(None) if getattr(ed, 'tz', None) is not None else ed
        # If user requested exclusion, skip tickers that have earnings inside the lookahead window.
        if exclude_earnings and (start <= ed_naive < end):
            print(f"  Skipping: earnings within next {args.days} days: {ed_naive}"); continue
        # Default behaviour: only include tickers whose earnings fall inside the window
        if not exclude_earnings and not (start <= ed_naive < end):
            print(f"  Earnings not in window ({args.days}d): {ed_naive}"); continue

        adtv = avg_volume_10d(tk)
        if adtv is None:
            print("  Could not determine 10D ADTV; skipping."); continue
        if max_stock_adtv and max_stock_adtv > 0 and adtv > max_stock_adtv:
            print(f"  Skipping due to larger ADTV ({adtv:,.0f} > {max_stock_adtv:,.0f})");
            continue

        print(f"  Earnings: {ed.date()} | 10D ADTV: {adtv:,.0f}")
        expiries = nearest_expiries(tk, expiries_count)
        if not expiries:
            print("  No option expiries found; skipping."); continue

        best_records = []
        total_pass_vol = 0
        for ex in expiries:
            oc = load_option_chain(tk, ex)
            if oc is None or oc.empty: continue
            oc2 = summarize_options(oc, min_option_volume, min_iv, min_iv_x_vol)
            total_pass_vol += oc2.loc[oc2["pass_filters"], "volume"].sum()

            # Apply optional open interest floor and pass-filters selection
            sel = oc2.copy()
            if min_oi and min_oi > 0:
                sel = sel[sel.get("openInterest", 0) >= min_oi]
            if only_pass:
                sel = sel[sel["pass_filters"]]
            # Determine how many to take per expiry
            if per_expiry_top and per_expiry_top > 0:
                sel = sel.head(per_expiry_top)
            # Accumulate selected rows
            for _, r in sel.iterrows():
                best_records.append({
                    "theme": tkr2theme.get(tk, ""),
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

            if make_plots:
                try:
                    plt.figure(figsize=(7,4.5))
                    for typ in ["call","put"]:
                        sub = oc[oc["type"]==typ]
                        if not sub.empty:
                            plt.plot(sub["strike"], sub["impliedVolatility"]*100, label=typ.upper())
                    plt.title(f"{tk} IV Smile ({ex})")
                    plt.xlabel("Strike"); plt.ylabel("IV (%)")
                    plt.legend(); plt.grid(True); plt.tight_layout()
                    plt.savefig(f"smile_{tk}_{ex}.png", dpi=140); plt.close()
                except Exception as e:
                    print(f"  [WARN] plot failed: {e}")

        if best_records:
            best_records = sorted(
                best_records,
                key=lambda x: (x["iv_x_vol"], x["volume"], x.get("openInterest", 0)),
                reverse=True,
            )
            if per_ticker_top and per_ticker_top > 0:
                best_records = best_records[:per_ticker_top]
            rows.extend(best_records)
            print(f"  Selected {len(best_records)} contracts; total pass volume (sum): {int(total_pass_vol)}")
        else:
            print("  No contracts met the filters.")

    if not rows:
        print("\nNo candidates found under current thresholds."); return

    out = pd.DataFrame(rows)
    out["iv_pct"] = (out["iv"] * 100.0).round(2)
    out["iv_x_vol"] = out["iv_x_vol"].round(2)
    out = out[["theme","ticker","earnings_date","expiry","type","strike","iv_pct","volume","iv_x_vol","bid","ask","openInterest","adtv_10d"]]
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    try:
        print(tabulate(out, headers="keys", tablefmt="github", showindex=False))
    except Exception:
        print(out)

if __name__ == "__main__":
    main()
