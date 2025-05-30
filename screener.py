# screener.py – pulls option IV from Tradier, price/history from yfinance
import os, datetime, requests, numpy as np, pandas as pd, yfinance as yf
from scipy.interpolate import interp1d

TRADIER_TOKEN = os.getenv("TRADIER_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {TRADIER_TOKEN}",
    "Accept": "application/json"
}
BASE = "https://sandbox.tradier.com/v1"

# ─── realised-vol (unchanged) ─────────────────────────────────────────
def yang_zhang(df, window=30, periods=252):
    if len(df) < window + 1:
        return np.nan
    log_ho = np.log(df.High  / df.Open)
    log_lo = np.log(df.Low   / df.Open)
    log_co = np.log(df.Close / df.Open)
    log_oc = np.log(df.Open  / df.Close.shift(1))
    log_cc = np.log(df.Close / df.Close.shift(1))

    rs = log_ho*(log_ho-log_co) + log_lo*(log_lo-log_co)
    ov = log_oc.pow(2).rolling(window).sum()/(window-1)
    cv = log_cc.pow(2).rolling(window).sum()/(window-1)
    rv = rs      .rolling(window).sum()/(window-1)

    k = 0.34/(1.34 + (window+1)/(window-1))
    sigma = np.sqrt(ov + k*cv + (1-k)*rv)*np.sqrt(periods)
    return sigma.iloc[-1]

# ─── Tradier helpers ──────────────────────────────────────────────────
def expirations(symbol):
    url = f"{BASE}/markets/options/expirations"
    r = requests.get(url, headers=HEADERS,
                     params={"symbol": symbol, "includeAllRoots": "true"})
    if r.status_code != 200 or "expirations" not in r.json():
        return []
    return r.json()["expirations"]["date"]       # list of YYYY-MM-DD

def atm_mid_iv(symbol, expiry, spot):
    """Return ATM mid-IV for <symbol, expiry> or None."""
    url = f"{BASE}/markets/options/chains"
    r = requests.get(url, headers=HEADERS,
                     params={"symbol": symbol, "expiration": expiry,
                             "greeks": "true"})
    if r.status_code != 200 or "options" not in r.json():
        return None
    df = pd.json_normalize(r.json()["options"]["option"])
    if df.empty:
        return None
    # choose strike closest to spot
    row = df.iloc[(df.strike - spot).abs().idxmin()]
    iv  = row["greeks.mid_iv"]
    return iv if iv else None

def term_structure(symbol, spot, dte_min=45):
    today = datetime.date.today()
    dtes, ivs = [], []
    for exp in expirations(symbol):
        exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte < dte_min:
            continue
        iv = atm_mid_iv(symbol, exp, spot)
        if iv is not None:
            dtes.append(dte)
            ivs .append(iv)
    return dtes, ivs

# ─── main scoring function ───────────────────────────────────────────
def score_ticker(sym: str):
    # spot price & historical bars from yfinance
    yf_tkr  = yf.Ticker(sym)
    price   = yf_tkr.history(period="1d").Close.iloc[-1]

    dtes, ivs = term_structure(sym, price)
    if len(ivs) < 2:
        return None

    curve     = interp1d(dtes, ivs, kind="linear", fill_value="extrapolate")
    ts_slope  = (curve(45) - curve(dtes[0])) / (45 - dtes[0])
    iv30      = curve(30)

    hist      = yf_tkr.history(period="4mo").dropna()
    rv30      = yang_zhang(hist)
    if np.isnan(rv30) or rv30 == 0:
        return None

    avg_vol   = hist.Volume.rolling(30).mean().iloc[-1]

    return dict(
        avg_volume    = avg_vol,
        iv30_rv30     = iv30 / rv30,
        ts_slope_0_45 = ts_slope,
    )

# ─── pass/fail rule (unchanged) ───────────────────────────────────────
def passes(m):
    return (
        m["avg_volume"]    >= 1_500_000 and
        m["iv30_rv30"]     >= 1.25      and
        m["ts_slope_0_45"] <= -0.00406
    )
