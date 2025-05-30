# screener.py – Tradier IV + defensive Yahoo history
import os, datetime as dt, requests, numpy as np, pandas as pd, yfinance as yf
from scipy.interpolate import interp1d

TRADIER_TOKEN = os.getenv("TRADIER_TOKEN")
HEADERS = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
BASE    = "https://sandbox.tradier.com/v1"

# ──────────────────────────────────────────────────────────────────────
def expirations(sym):
    r = requests.get(f"{BASE}/markets/options/expirations",
                     headers=HEADERS, params={"symbol": sym})
    if r.ok and "expirations" in r.json():
        return r.json()["expirations"]["date"]
    return []

def atm_mid_iv(sym, expiry, spot):
    r = requests.get(f"{BASE}/markets/options/chains",
                     headers=HEADERS,
                     params={"symbol": sym, "expiration": expiry, "greeks": "true"})
    if not (r.ok and "options" in r.json()):
        return None
    df = pd.json_normalize(r.json()["options"]["option"])
    if df.empty:
        return None
    row = df.iloc[(df.strike - spot).abs().idxmin()]
    return row["greeks.mid_iv"] or None

def yang_zhang(df, window=30, periods=252):
    if len(df) < window + 1:
        return np.nan
    log_ho = np.log(df.High/df.Open)
    log_lo = np.log(df.Low /df.Open)
    log_co = np.log(df.Close/df.Open)
    log_oc = np.log(df.Open/df.Close.shift(1))
    log_cc = np.log(df.Close/df.Close.shift(1))
    rs = log_ho*(log_ho-log_co) + log_lo*(log_lo-log_co)
    ov = log_oc.pow(2).rolling(window).sum()/(window-1)
    cv = log_cc.pow(2).rolling(window).sum()/(window-1)
    rv = rs      .rolling(window).sum()/(window-1)
    k  = 0.34/(1.34 + (window+1)/(window-1))
    return np.sqrt(ov + k*cv + (1-k)*rv).iloc[-1] * np.sqrt(periods)

# ──────────────────────────────────────────────────────────────────────
def score_ticker(raw_sym: str):
    sym = raw_sym.strip().upper()           # ← sanitize ticker
    if not sym:
        return None

    # ---- get last price via Tradier quote (faster, avoids Yahoo) ----
    q = requests.get(f"{BASE}/markets/quotes",
                     headers=HEADERS, params={"symbols": sym}).json()
    try:
        spot = float(q["quotes"]["quote"]["last"])
    except (KeyError, TypeError):
        return None

    # ---- build IV term structure -----------------------------------
    today, dtes, ivs = dt.date.today(), [], []
    for exp in expirations(sym):
        dte = (dt.datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        if dte < 45:
            continue
        iv = atm_mid_iv(sym, exp, spot)
        if iv is not None:
            dtes.append(dte); ivs.append(iv)

    if len(ivs) < 2:
        return None
    curve = interp1d(dtes, ivs, kind="linear", fill_value="extrapolate")
    ts_slope = (curve(45) - curve(dtes[0])) / (45 - dtes[0])
    iv30     = curve(30)

    # ---- realised vol & volume from Yahoo (skip on error) ----------
    try:
        yf_hist = yf.Ticker(sym).history(period="4mo").dropna()
        rv30    = yang_zhang(yf_hist)
        avg_vol = yf_hist.Volume.rolling(30).mean().iloc[-1]
    except Exception:
        return None

    if np.isnan(rv30) or rv30 == 0:
        return None

    return dict(
        avg_volume    = avg_vol,
        iv30_rv30     = iv30 / rv30,
        ts_slope_0_45 = ts_slope,
    )

def passes(m):
    return (
        m["avg_volume"]    >= 1_500_000 and
        m["iv30_rv30"]     >= 1.25      and
        m["ts_slope_0_45"] <= -0.00406
    )
