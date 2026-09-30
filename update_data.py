"""
Downloads fresh screener data and saves it to data/screener_data.csv.

GitHub Actions runs this on a schedule (see .github/workflows/update-data.yml),
so the app never has to download anything while you wait.
Uses Finnhub for company financials if FINNHUB_API_KEY is set, otherwise Yahoo.
"""
import os
import sys
import datetime as dt
from pathlib import Path

import stock_screener as sc

OUT = Path(__file__).parent / "data" / "screener_data.csv"


def main():
    key = os.environ.get("FINNHUB_API_KEY") or None
    print(f"Financials source: {'Finnhub' if key else 'Yahoo'}")

    def progress(done, total, label):
        if total <= 1 or done % 50 == 0 or done == total:
            print(f"  {label} {done}/{total}", flush=True)

    raw = sc.fetch_all(progress, key)   # raises (and fails the run) if data is bad
    raw["fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes")
    OUT.parent.mkdir(exist_ok=True)
    raw.to_csv(OUT, index=False)
    have = int(raw["data_error"].isna().sum()) if "data_error" in raw else len(raw)
    print(f"Saved {len(raw)} stocks ({have} with financials) to {OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)   # a failed run keeps the previous day's data in place
