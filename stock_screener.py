#!/usr/bin/env python3
"""
Strict "Quality at a Reasonable Price" stock screener.

A stock is flagged only if it passes EVERY check in all four gates:

  1. TREND     - in an uptrend, not a falling knife, not overextended
  2. QUALITY   - high returns on capital, fat margins, modest debt, real cash flow,
                 and profitable in every recent year
  3. VALUE     - not expensive relative to its earnings and cash flow
  4. STRENGTH  - Piotroski F-score >= 7 (fundamentals improving year over year)

Universe: S&P 500, excluding Financials and Real Estate (bank/REIT accounting
makes these metrics misleading). By default it also excludes Energy and
Materials: commodity producers look cheapest and highest-quality at the PEAK of
their cycle, which this kind of screen can't tell apart from a great business.

This file works two ways:
  - as a command-line script:  python stock_screener.py
  - as the engine behind app.py (the Streamlit app)

It finds stocks that match a disciplined rule set. It does NOT predict returns;
a pass is a reason to research a company, not to buy it. Data comes from Yahoo
Finance via the unofficial `yfinance` library and can be wrong or rate-limited.
"""

import time
import datetime as dt
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# ---------------------------------------------------------------------------
# DEFAULT THRESHOLDS
# ---------------------------------------------------------------------------
DEFAULT_T = {
    # Trend
    "max_above_200dma": 0.15,     # price no more than 15% above its 200-day average
    "min_momentum_12_1": 0.00,    # 12-month return (skipping last month) must be positive
    # Quality
    "min_roe": 0.15,
    "min_roa": 0.07,              # guards against buyback-inflated ROE
    "min_op_margin": 0.15,
    "max_debt_to_equity": 100,    # Yahoo reports in percent: 100 = debt equals equity
    "min_current_ratio": 1.2,
    "min_fcf_conversion": 0.80,   # free cash flow >= 80% of net income
    "min_revenue_growth": 0.00,
    "min_earnings_growth": 0.00,
    # Value
    "max_trailing_pe": 25,
    "max_forward_pe": 20,
    "max_ev_ebitda": 14,
    "min_fcf_yield": 0.05,
    # Financial strength
    "min_piotroski": 7,
}

ALWAYS_EXCLUDED = {"Financials", "Real Estate"}
CYCLICAL_SECTORS = {"Energy", "Materials"}
WORKERS = 6
RETRIES = 3

try:
    import yfinance as yf
except ImportError:
    yf = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def retry(fn):
    for attempt in range(RETRIES):
        try:
            return fn()
        except Exception:
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 * (attempt + 1))


def num(x):
    """Float, or None for missing/NaN/non-numeric values."""
    try:
        x = float(x)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
def get_universe():
    """S&P 500 constituents from Wikipedia. Falls back to tickers.txt."""
    try:
        import requests
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
        df = pd.read_html(StringIO(html))[0]
        df = df.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
        df = df[~df["sector"].isin(ALWAYS_EXCLUDED)]
        return df[["ticker", "name", "sector"]].reset_index(drop=True)
    except Exception as e:
        print(f"Couldn't load S&P 500 list ({e}). Trying tickers.txt ...")
        with open("tickers.txt") as f:
            tickers = [t.strip().upper() for t in f if t.strip()]
        return pd.DataFrame({"ticker": tickers, "name": "", "sector": ""})


# ---------------------------------------------------------------------------
# Data: prices
# ---------------------------------------------------------------------------
def download_prices(tickers):
    """Bulk download, then retry any tickers that failed one at a time."""
    px = yf.download(tickers, period="14mo", auto_adjust=True, progress=False, threads=True)
    if isinstance(px.columns, pd.MultiIndex):
        closes = px["Close"].copy()
    else:
        closes = px[["Close"]].rename(columns={"Close": tickers[0]})
    missing = [t for t in tickers if t not in closes or closes[t].dropna().empty]
    for t in missing:
        try:
            one = retry(lambda: yf.download(t, period="14mo", auto_adjust=True,
                                            progress=False, threads=False))
            c = one["Close"]
            closes[t] = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
        except Exception:
            pass
    return closes


def trend_metrics(close):
    close = close.dropna()
    if len(close) < 253:
        return {}
    price = close.iloc[-1]
    sma50 = close.iloc[-50:].mean()
    sma200 = close.iloc[-200:].mean()
    return {"price": price, "sma50": sma50, "sma200": sma200,
            "pct_above_200dma": price / sma200 - 1,
            "momentum_12_1": close.iloc[-22] / close.iloc[-253] - 1}


# ---------------------------------------------------------------------------
# Data: fundamentals
# ---------------------------------------------------------------------------
def fundamental_metrics(info):
    mcap = num(info.get("marketCap"))
    fcf = num(info.get("freeCashflow"))
    ni = num(info.get("netIncomeToCommon"))
    return {
        "roe": num(info.get("returnOnEquity")),
        "roa": num(info.get("returnOnAssets")),
        "op_margin": num(info.get("operatingMargins")),
        "debt_to_equity": num(info.get("debtToEquity")),
        "current_ratio": num(info.get("currentRatio")),
        "fcf_conversion": fcf / ni if fcf is not None and ni and ni > 0 else None,
        "revenue_growth": num(info.get("revenueGrowth")),
        "earnings_growth": num(info.get("earningsGrowth")),
        "trailing_pe": num(info.get("trailingPE")),
        "forward_pe": num(info.get("forwardPE")),
        "ev_ebitda": num(info.get("enterpriseToEbitda")),
        "fcf_yield": fcf / mcap if fcf is not None and mcap else None,
        "market_cap_bn": mcap / 1e9 if mcap else None,
    }


def _row(df, *names):
    """[latest_year, prior_year] values for the first matching line item."""
    if df is None or df.empty:
        return None
    df = df.reindex(sorted(df.columns, reverse=True), axis=1)  # newest first
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce").iloc[:2]
            if len(s) == 2 and s.notna().all():
                return s.tolist()
    return None


def piotroski(income, balance, cashflow):
    """Return (score 0-9, number of signals that could be computed)."""
    ni = _row(income, "Net Income", "Net Income Common Stockholders")
    rev = _row(income, "Total Revenue")
    gp = _row(income, "Gross Profit")
    ta = _row(balance, "Total Assets")
    ca = _row(balance, "Current Assets")
    cl = _row(balance, "Current Liabilities")
    ltd = _row(balance, "Long Term Debt", "Long Term Debt And Capital Lease Obligation")
    shares = _row(balance, "Ordinary Shares Number", "Share Issued")
    if shares is None:
        shares = _row(income, "Diluted Average Shares")
    cfo = _row(cashflow, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")

    signals = []

    def sig(cond_fn, *needed):
        signals.append(None if any(x is None for x in needed) else bool(cond_fn()))

    # A company with no long-term debt line has zero leverage.
    lev = None
    if ta is not None:
        lev = [ltd[0] / ta[0], ltd[1] / ta[1]] if ltd is not None else [0.0, 0.0]

    sig(lambda: ni[0] / ta[0] > 0, ni, ta)                        # 1 profitable
    sig(lambda: cfo[0] > 0, cfo)                                  # 2 positive cash flow
    sig(lambda: ni[0] / ta[0] > ni[1] / ta[1], ni, ta)            # 3 ROA improving
    sig(lambda: cfo[0] > ni[0], cfo, ni)                          # 4 cash > accounting profit
    sig(lambda: lev[0] <= lev[1], lev)                            # 5 leverage not rising
    sig(lambda: ca[0] / cl[0] > ca[1] / cl[1], ca, cl)            # 6 liquidity improving
    sig(lambda: shares[0] <= shares[1] * 1.005, shares)           # 7 no dilution
    sig(lambda: gp[0] / rev[0] > gp[1] / rev[1], gp, rev)         # 8 gross margin improving
    sig(lambda: rev[0] / ta[0] > rev[1] / ta[1], rev, ta)         # 9 asset turnover improving

    return sum(1 for s in signals if s), sum(1 for s in signals if s is not None)


def profitable_every_year(income):
    """True if net income > 0 in every annual report Yahoo provides (usually 4).
    None if fewer than 3 years are available."""
    if income is None or income.empty:
        return None
    for n in ("Net Income", "Net Income Common Stockholders"):
        if n in income.index:
            s = pd.to_numeric(income.loc[n], errors="coerce").dropna()
            return bool((s > 0).all()) if len(s) >= 3 else None
    return None


def fetch_one(ticker):
    t = yf.Ticker(ticker)
    info = retry(lambda: t.info)
    m = fundamental_metrics(info)
    m.update(piotroski=None, piotroski_computed=None, profitable_all_years=None)
    # Stocks with no profit or no free cash flow fail the quality gate at ANY
    # threshold, so skip their annual reports to save time.
    ni, fcf = num(info.get("netIncomeToCommon")), num(info.get("freeCashflow"))
    if ni and ni > 0 and fcf and fcf > 0:
        inc = retry(lambda: t.income_stmt)
        score, computed = piotroski(inc, retry(lambda: t.balance_sheet), retry(lambda: t.cashflow))
        m.update(piotroski=score, piotroski_computed=computed,
                 profitable_all_years=profitable_every_year(inc))
    return m


def fetch_all(progress=None):
    """Download everything the screen needs. Slow (several minutes); run once,
    then call evaluate() as many times as you like with different thresholds.
    progress: optional callback(done, total, label)."""
    if yf is None:
        raise RuntimeError("Missing library. Run: pip install yfinance pandas lxml requests")
    say = progress or (lambda d, t, label: None)

    uni = get_universe()
    tickers = uni["ticker"].tolist()
    say(0, 1, "Downloading prices")
    closes = download_prices(tickers)

    rows = {r.ticker: {"ticker": r.ticker, "name": r.name, "sector": r.sector,
                       **trend_metrics(closes[r.ticker] if r.ticker in closes else pd.Series(dtype=float))}
            for r in uni.itertuples()}

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_one, tk): tk for tk in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            tk = futures[fut]
            try:
                rows[tk].update(fut.result())
            except Exception as e:
                rows[tk]["data_error"] = type(e).__name__
            say(i, len(tickers), "Checking fundamentals")

    return pd.DataFrame(rows.values())


# ---------------------------------------------------------------------------
# Rules: apply thresholds to already-downloaded data (instant)
# ---------------------------------------------------------------------------
def failed_checks(r, T, require_consistent_profit=True):
    f = []
    g = lambda k: num(r.get(k))

    def check(label, key, ok):
        v = g(key)
        if v is None:
            f.append(f"{label}: no data")
        elif not ok(v):
            f.append(label)

    if r.get("data_error") and isinstance(r.get("data_error"), str):
        return [f"data error: {r['data_error']}"]

    # Trend
    if g("price") is None:
        f.append("trend: not enough price history")
    else:
        if g("price") <= g("sma200"):
            f.append("price below 200-day avg")
        if g("sma50") <= g("sma200"):
            f.append("50-day avg below 200-day avg")
        if g("pct_above_200dma") > T["max_above_200dma"]:
            f.append("overextended above 200-day avg")
        if g("momentum_12_1") <= T["min_momentum_12_1"]:
            f.append("negative 12-month momentum")

    # Quality
    check("ROE too low", "roe", lambda v: v >= T["min_roe"])
    check("ROA too low", "roa", lambda v: v >= T["min_roa"])
    check("operating margin too low", "op_margin", lambda v: v >= T["min_op_margin"])
    check("too much debt", "debt_to_equity", lambda v: 0 <= v <= T["max_debt_to_equity"])
    check("current ratio too low", "current_ratio", lambda v: v >= T["min_current_ratio"])
    check("weak cash conversion", "fcf_conversion", lambda v: v >= T["min_fcf_conversion"])
    check("revenue shrinking", "revenue_growth", lambda v: v > T["min_revenue_growth"])
    check("earnings shrinking", "earnings_growth", lambda v: v > T["min_earnings_growth"])
    if require_consistent_profit:
        p = r.get("profitable_all_years")
        if p is None or p != p:
            f.append("profit history: no data")
        elif not p:
            f.append("lost money in a recent year")

    # Value
    check("trailing P/E too high", "trailing_pe", lambda v: 0 < v <= T["max_trailing_pe"])
    check("forward P/E too high", "forward_pe", lambda v: 0 < v <= T["max_forward_pe"])
    check("EV/EBITDA too high", "ev_ebitda", lambda v: 0 < v <= T["max_ev_ebitda"])
    check("FCF yield too low", "fcf_yield", lambda v: v >= T["min_fcf_yield"])

    # Strength
    check("Piotroski F-score too low", "piotroski", lambda v: v >= T["min_piotroski"])
    return f


def evaluate(raw, T=None, exclude_cyclicals=True, require_consistent_profit=True):
    """Return raw data plus PASSED / num_failed / failed_checks, best first."""
    T = {**DEFAULT_T, **(T or {})}
    df = raw.copy()
    if exclude_cyclicals:
        df = df[~df["sector"].isin(CYCLICAL_SECTORS)]
    fails = [failed_checks(r, T, require_consistent_profit) for r in df.to_dict("records")]
    df["PASSED"] = [not f for f in fails]
    df["num_failed"] = [len(f) for f in fails]
    df["failed_checks"] = ["; ".join(f) for f in fails]
    if "fcf_yield" not in df:
        df["fcf_yield"] = None
    return df.sort_values(["PASSED", "num_failed", "fcf_yield"],
                          ascending=[False, True, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Command-line use
# ---------------------------------------------------------------------------
def main():
    def progress(done, total, label):
        if label != "Checking fundamentals" or done % 50 == 0 or done == total:
            print(f"  {label} ... {done}/{total}" if total > 1 else f"  {label} ...")

    print("Fetching data (this takes several minutes) ...")
    raw = fetch_all(progress)
    df = evaluate(raw)

    fname = f"screen_results_{dt.date.today():%Y-%m-%d}.csv"
    df.to_csv(fname, index=False, float_format="%.4f")

    passed = df[df["PASSED"]]
    print("\n" + "=" * 70)
    if passed.empty:
        print("No stocks passed every check today. That's normal for a strict screen.")
    else:
        print(f"{len(passed)} STOCK(S) PASSED ALL CHECKS:\n")
        print(passed[["ticker", "name", "sector", "price", "trailing_pe", "forward_pe",
                      "fcf_yield", "roe", "piotroski"]].round(3).to_string(index=False))

    near = df[(~df["PASSED"]) & (df["num_failed"] == 1)]
    if not near.empty:
        print("\nNear misses (failed exactly one check):")
        for _, r in near.head(10).iterrows():
            print(f"  {r['ticker']:<6} {str(r['name'])[:30]:<30}  failed: {r['failed_checks']}")
    print(f"\nFull results saved to {fname}")
    print("Reminder: a pass means 'worth researching', not 'buy'.")


if __name__ == "__main__":
    main()
