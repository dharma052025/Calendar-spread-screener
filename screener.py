# screener.py
import numpy as np, yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d

# ---------- helpers --------------------------------------------------
def filter_dates(dates):
    today  = datetime.utcnow().date()
    cutoff = today + timedelta(days=45)
    keep   = sorted(
        d for d in (datetime.strptime(x, "%Y-%m-%d").date() for x in dates)
        if d >= cutoff
    )
    return [d.strftime("%Y-%m-%d") for d in keep[:1]]  # nearest ≥45-d expiry

def yang_zhang(df, window=30, periods=252):
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

def term_curve(days, ivs):
    i = np.argsort(days)
    return interp1d(np.asarray(days)[i], np.asarray(ivs)[i],
                    kind="linear", fill_value="extrapolate")

# ---------- main scoring function -----------------------------------
def score_ticker(sym: str):
    tk = yf.Ticker(sym)
    if not tk.options:
        return None                     # skip symbols w/o options

    # ---- build ATM IV term structure ----
    atm_ivs, dtes = [], []
    today = datetime.utcnow().date()
    for exp in filter_dates(tk.options):
        chain = tk.option_chain(exp)
        if chain.calls.empty or chain.puts.empty:
            continue
        price = tk.history(period="1d").Close.iloc[-1]
        calls, puts = chain.calls, chain.puts
        atm_call = calls.iloc[(calls.strike-price).abs().idxmin()]
        atm_put  =  puts.iloc[(puts.strike -price).abs().idxmin()]
        atm_iv   = (atm_call.impliedVolatility + atm_put.impliedVolatility)/2
        atm_ivs.append(atm_iv)
        dtes.append((datetime.strptime(exp, "%Y-%m-%d").date() - today).days)

    if not atm_ivs:
        return None

    curve     = term_curve(dtes, atm_ivs)
    ts_slope  = (curve(45) - curve(dtes[0])) / (45 - dtes[0])
    iv30      = curve(30)

    hist      = tk.history(period="3mo")
    rv30      = yang_zhang(hist)
    avg_vol   = hist.Volume.rolling(30).mean().iloc[-1]

    return dict(
        avg_volume     = avg_vol,
        iv30_rv30      = iv30/rv30,
        ts_slope_0_45  = ts_slope,
    )

def passes(m):
    return (
        m["avg_volume"]    >= 1_500_000 and
        m["iv30_rv30"]     >= 1.25      and
        m["ts_slope_0_45"] <= -0.00406
    )
