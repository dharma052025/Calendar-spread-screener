# scan.py
import csv, os, datetime, requests, screener

TG_TOKEN = os.environ["TG_TOKEN"]   # set as repo secret
TG_CHAT  = os.environ["TG_CHAT"]

def alert(txt: str):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TG_CHAT, "text": txt})

def main():
    today     = datetime.date.today()
    out_file  = f"list{today:%Y%m%d}.csv"
    hits, rows = [], []

    with open("tickers.csv") as f:
        for sym in map(str.strip, f):
            if not sym: continue
            m = screener.score_ticker(sym)
            if m and screener.passes(m):
                rows.append([sym, m["avg_volume"], m["iv30_rv30"], m["ts_slope_0_45"]])
                hits.append(f"{sym:<6} | avgVol={m['avg_volume']/1_000_000:.1f} M | "
                            f"IV30/RV30={m['iv30_rv30']:.2f} | "
                            f"TS_slope={m['ts_slope_0_45']:.5f}")

    if rows:
        with open(out_file, "w", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["Symbol","AvgVol","IV30/RV30","TS_slope_0_45"])
            w.writerows(rows)
        msg = f"📆 {today:%Y-%m-%d} Calendar-spread scan ({len(rows)} hits)\n\n" + "\n".join(hits)
    else:
        msg = f"📆 {today:%Y-%m-%d}: no tickers met all three filters."
    alert(msg)

if __name__ == "__main__":
    main()
