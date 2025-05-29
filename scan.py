# scan.py  –  calendar-spread screener + Telegram alert
import csv, os, datetime, requests, screener

# ── Telegram credentials (set as repo secrets) ────────────────────────
TG_TOKEN = os.environ["TG_TOKEN"]
TG_CHAT  = os.environ["TG_CHAT"]

def alert(text: str):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TG_CHAT, "text": text})

def fmt_metrics(m):
    """Return a one-liner with the three numeric metrics."""
    return (f"avgVol={m['avg_volume']/1_000_000:.1f} M | "
            f"IV30/RV30={m['iv30_rv30']:.2f} | "
            f"TS_slope={m['ts_slope_0_45']:.5f}")

def main():
    today     = datetime.date.today()
    out_file  = f"list{today:%Y%m%d}.csv"

    passed, all_scores = [], []          # two buckets

    # ── iterate through the ticker universe ────────────────────────────
    with open("tickers.csv") as f:
        for sym in map(str.strip, f):
            if not sym:
                continue
            m = screener.score_ticker(sym)
            if not m:
                continue                 # skip symbols w/o option data
            all_scores.append((sym, m))
            if screener.passes(m):
                passed.append((sym, m))

    # ── write CSV with the *passing* names only (can be empty) ─────────
    with open(out_file, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["Symbol", "AvgVol", "IV30/RV30", "TS_slope_0_45"])
        for sym, m in passed:
            w.writerow([sym, m["avg_volume"], m["iv30_rv30"], m["ts_slope_0_45"]])

    # ── build Telegram message ─────────────────────────────────────────
    if passed:
        body = "\n".join(f"{sym:<6} | {fmt_metrics(m)}" for sym, m in passed)
        msg  = (f"📆 {today:%Y-%m-%d} Calendar-spread scan "
                f"({len(passed)} hits)\n\n{body}")
    else:
        # no passes → show the scorecard for every ticker we evaluated
        scorecard = "\n".join(f"{sym:<6} | {fmt_metrics(m)}"
                              for sym, m in all_scores)
        msg = (f"📆 {today:%Y-%m-%d}: **no ticker met all three filters**\n\n"
               f"Scores for reference:\n{scorecard}")

    alert(msg)

if __name__ == "__main__":
    main()

