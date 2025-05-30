"""
screener.py  ·  Calendar-spread metrics, 100 % Tradier sandbox

Requires repo secret TRADIER_TOKEN.
"""

import os, datetime as dt, numpy as np, pandas as pd, requests
from scipy.interpolate import interp1d

TOKEN = os.environ["TRADIER_TOKEN"]
HEAD  = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}
BASE  = "https://sandbox.tradier.com/v1"


# ────────────────────────── Tradier helper calls ─────────────────────
def quote_price(sym: str):
    r = requests.get(f"{BASE}/markets/quotes", headers=HEAD,
                     params={"symbols": sym}).json()
    try:
        return float(r["quotes"]["quote"]["last"])
    except (KeyError, TypeError):
        return None


def expirations(sym: str):
    r = requests.get(f"{BASE}/markets/options/expirations",
                     headers=HEAD, params={"symbol": sym})
    if r.ok and "expirations" in r.json():
        return r.json()["expirations"]["date"]
    return []


def atm_iv(sym: str, expiry: str, spot: float):
    r = requests.get(f"{BASE}/markets/options/chains", headers=HEAD,
                     params={"symbol": sym, "expiration": expiry,
                             "greeks": "true"})
    if not (r.ok and "options" in r.json()):
        return None
    df = pd.json_normalize(r.json()["options"]["option"])
    if df.empty:
        return None
    row = df.iloc[(df.strike - spot).abs().idxmin()]
    return row["greeks.mid_iv"] or None


def history_df(sym: str, months=4):
    start = (dt.date.today() - dt.timedelta(days=30 * months)).isoformat()
    r = requests.get(f"{BASE}/markets/history", headers=HEAD,
                     params={"symbol": sym, "interval": "daily",
                             "session_filter": "all",
                             "start": start})
    if not (r.ok and "history" in r.json()):
        return pd.DataFrame()
    rows = r.json()["history"]["day"]
    return pd.DataFrame(rows).assign(
        date=lambda df: pd.to_datetime(df.date)
    ).set_index("date").astype(float)


# ────────────────────────── realised volatility ──────────────────────
def yang_zhang(df, window=30, periods=252):
    if len(df) < window + 1:
        return np.nan
    log_ho = np.log(df.high / df.open)
    log_lo = np.log(df.low  / df.open)
    log_co = np.log(df.close/ df.open)
    log_oc = np.log(df.open / df.close.shift(1))
    log_cc = np.log(df.close/ df.close.shift(1))

    rs = log_ho*(log_ho-log_co) + log_lo*(log_lo-log_co)
    ov = log_oc.pow(2).rolling(window).sum()/(window-1)
    cv = log_cc.pow(2).rolling(window).sum()/(window-1)
    rv = rs      .rolling(window).sum()/(window-1)

    k = 0.34/(1.34 + (window+1)/(window-1))
    sigma = np.sqrt(ov + k*cv + (1-k)*rv).iloc[-1] * np.sqrt(periods)
    return sigma


# ────────────────────────── main scoring fn ──────────────────────────
def score_ticker(raw_sym: str):
    sym = raw_sym.strip().upper()
    if not sym:
        return None

    spot = quote_price(sym)
    if spot is None:
        return None

    today, dtes, ivs = dt.date.today(), [], []
    for exp in expirations(sym):
        dte = (dt.datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        if dte < 45:
            continue
        iv = atm_iv(sym, exp, spot)
        if iv is not None:
            dtes.append(dte); ivs.append(iv)
    if len(ivs) < 2:
        return None

    curve     = interp1d(dtes, ivs, kind="linear", fill_value="extrapolate")
    ts_slope  = (curve(45) - curve(dtes[0])) / (45 - dtes[0])
    iv30      = curve(30)

    hist = history_df(sym)
    rv30 = yang_zhang(hist)
    if np.isnan(rv30) or rv30 == 0:
        return None

    avg_vol = hist.volume.rolling(30).mean().iloc[-1]

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

