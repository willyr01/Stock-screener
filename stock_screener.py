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
import random
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
WORKERS = 3
MIN_COVERAGE = 0.5   # refuse results if under half the stocks got fundamentals
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


class NoFundamentals(Exception):
    """Yahoo answered but sent back no fundamentals (usually a soft block)."""


def _get_info(t):
    info = t.info or {}
    # A real response has dozens of fields. A near-empty one means Yahoo
    # withheld the data, so treat it as a failure and retry instead of
    # silently recording every metric as missing.
    if len(info) < 10 or num(info.get("marketCap")) is None:
        raise NoFundamentals()
    return info


def fetch_one(ticker):
    time.sleep(random.uniform(0.2, 0.8))  # be gentle; bursts get blocked
    t = yf.Ticker(ticker)
    info = retry(lambda: _get_info(t))
    m = fundamental_metrics(info)
    m.update(piotroski=None, piotroski_computed=None, piotroski_max=9, profitable_all_years=None)
    # Stocks with no profit or no free cash flow fail the quality gate at ANY
    # threshold, so skip their annual reports to save time.
    ni, fcf = num(info.get("netIncomeToCommon")), num(info.get("freeCashflow"))
    if ni and ni > 0 and fcf and fcf > 0:
        inc = retry(lambda: t.income_stmt)
        score, computed = piotroski(inc, retry(lambda: t.balance_sheet), retry(lambda: t.cashflow))
        m.update(piotroski=score, piotroski_computed=computed,
                 profitable_all_years=profitable_every_year(inc))
    return m


# ---------------------------------------------------------------------------
# Data: fundamentals from Finnhub (used when an API key is provided)
# ---------------------------------------------------------------------------
FINNHUB_URL = "https://finnhub.io/api/v1/stock/metric"
FINNHUB_PACE = 1.05          # seconds between calls; free tier allows ~60/minute
PCT_FIELDS = ["roe", "roa", "op_margin", "revenue_growth", "earnings_growth"]


class FinnhubKeyError(Exception):
    pass


def _pick(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _series(src, name):
    """Values of an annual/quarterly Finnhub series, newest first."""
    pts = [p for p in (src or {}).get(name) or [] if num(p.get("v")) is not None]
    pts.sort(key=lambda p: str(p.get("period")), reverse=True)
    return [float(p["v"]) for p in pts]


def finnhub_metrics(js):
    """Turn one /stock/metric response into the screen's fields.
    Percent-style fields are left in Finnhub's units here; fetch_all()
    detects the unit across all stocks and converts to fractions."""
    m = js.get("metric") or {}
    ann = (js.get("series") or {}).get("annual") or {}
    qtr = (js.get("series") or {}).get("quarterly") or {}
    g = lambda *keys: _pick(*(num(m.get(k)) for k in keys))
    latest = lambda src, name: (_series(src, name) or [None])[0]

    pe = _pick(g("peTTM", "peExclExtraTTM", "peBasicExclExtraTTM"), latest(qtr, "peTTM"))
    pfcf = _pick(g("pfcfShareTTM"), latest(qtr, "pfcfTTM"))
    mcap = g("marketCapitalization")  # millions of dollars

    out = {
        "roe": g("roeTTM", "roeRfy"),
        "roa": g("roaTTM", "roaRfy"),
        "op_margin": g("operatingMarginTTM", "operatingMarginAnnual"),
        "debt_to_equity": g("totalDebt/totalEquityQuarterly", "totalDebt/totalEquityAnnual"),
        "current_ratio": g("currentRatioQuarterly", "currentRatioAnnual"),
        "revenue_growth": g("revenueGrowthTTMYoy", "revenueGrowthQuarterlyYoy"),
        "earnings_growth": g("epsGrowthTTMYoy", "epsGrowthQuarterlyYoy"),
        "trailing_pe": pe,
        "forward_pe": g("forwardPE", "peForward"),          # usually absent on free tier
        "ev_ebitda": g("currentEv/ebitdaTTM", "evEbitdaTTM"),  # usually absent on free tier
        # Free-cash-flow yield = 1 / (price / FCF). Negative FCF gives a negative yield.
        "fcf_yield": 1 / pfcf if pfcf else None,
        # FCF / earnings = (price/earnings) / (price/FCF). Unit-free.
        "fcf_conversion": pe / pfcf if pe and pfcf and pe > 0 else None,
        "market_cap_bn": mcap / 1000 if mcap else None,
    }

    # Piotroski-style score from Finnhub's annual history. Finnhub's free data has
    # no share counts, so the dilution test is left out: the score is out of 8.
    roa, nm, gm = _series(ann, "roa"), _series(ann, "netMargin"), _series(ann, "grossMargin")
    fcfm, cr, ltd = _series(ann, "fcfMargin"), _series(ann, "currentRatio"), _series(ann, "longtermDebtTotalAsset")
    two = lambda s: len(s) >= 2
    signals = []

    def sig(ok_fn, *need):
        signals.append(bool(ok_fn()) if all(need) else None)

    sig(lambda: roa[0] > 0, roa)                              # profitable
    sig(lambda: fcfm[0] > 0, fcfm)                            # positive cash flow
    sig(lambda: roa[0] > roa[1], two(roa))                    # ROA improving
    sig(lambda: fcfm[0] > nm[0], fcfm, nm)                    # cash beats accounting profit
    sig(lambda: ltd[0] <= ltd[1], two(ltd))                   # leverage not rising
    sig(lambda: cr[0] > cr[1], two(cr))                       # liquidity improving
    sig(lambda: gm[0] > gm[1], two(gm))                       # gross margin improving
    turn = [r / n for r, n in zip(roa, nm) if n]              # sales/assets = ROA / net margin
    sig(lambda: turn[0] > turn[1], two(turn))                 # asset turnover improving
    if not ltd:   # company reports no long-term debt: leverage can't have risen
        signals[4] = True

    out["piotroski"] = sum(1 for s in signals if s)
    out["piotroski_computed"] = sum(1 for s in signals if s is not None)
    out["piotroski_max"] = 8
    recent = nm[:4]
    out["profitable_all_years"] = bool(all(v > 0 for v in recent)) if len(recent) >= 3 else None
    return out


def fetch_finnhub(symbol, key, session):
    sym = symbol.replace("-", ".")  # BRK-B -> BRK.B
    for attempt in range(4):
        r = session.get(FINNHUB_URL, params={"symbol": sym, "metric": "all", "token": key},
                        timeout=20)
        if r.status_code == 401:
            raise FinnhubKeyError("Finnhub rejected the API key. Check the FINNHUB_API_KEY "
                                  "line in the app's Secrets settings.")
        if r.status_code == 429:           # over the per-minute limit: back off
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        js = r.json()
        if not (js.get("metric") or {}):
            raise NoFundamentals()
        return finnhub_metrics(js)
    raise RuntimeError("Finnhub rate limit")


def normalize_units(df):
    """Finnhub reports percentages (15 = 15%); the rules use fractions (0.15).
    Detect the unit from the whole universe rather than assuming it."""
    for c in PCT_FIELDS:
        if c in df and df[c].notna().any() and df[c].abs().median() > 1.5:
            df[c] = df[c] / 100
    # Debt-to-equity rules are in percent (100 = debt equals equity).
    c = "debt_to_equity"
    if c in df and df[c].notna().any() and df[c].abs().median() < 10:
        df[c] = df[c] * 100
    return df


def fetch_all(progress=None, finnhub_key=None):
    """Download everything the screen needs. Slow (several minutes); run once,
    then call evaluate() as many times as you like with different thresholds.
    Prices always come from Yahoo. Company financials come from Finnhub when a
    key is given, otherwise from Yahoo.
    progress: optional callback(done, total, label)."""
    if yf is None:
        raise RuntimeError("Missing library. Run: pip install yfinance pandas lxml requests")
    say = progress or (lambda d, t, label: None)
    source = "Finnhub" if finnhub_key else "Yahoo"

    uni = get_universe()
    tickers = uni["ticker"].tolist()
    say(0, 1, "Downloading prices")
    closes = download_prices(tickers)

    rows = {r.ticker: {"ticker": r.ticker, "name": r.name, "sector": r.sector,
                       **trend_metrics(closes[r.ticker] if r.ticker in closes else pd.Series(dtype=float))}
            for r in uni.itertuples()}

    ok = failed = 0
    first_error = None

    def record(tk, result=None, err=None):
        nonlocal ok, failed, first_error
        if err is None:
            rows[tk].update(result)
            ok += 1
        else:
            rows[tk]["data_error"] = (f"{source} sent no fundamentals"
                                      if isinstance(err, NoFundamentals) else type(err).__name__)
            first_error = first_error or err
            failed += 1

    def blocked():
        return ok == 0 and failed >= (15 if finnhub_key else 30)

    if finnhub_key:
        import requests
        session = requests.Session()
        for i, tk in enumerate(tickers, 1):
            start = time.time()
            try:
                record(tk, fetch_finnhub(tk, finnhub_key, session))
            except FinnhubKeyError:
                raise
            except Exception as e:
                record(tk, err=e)
            say(i, len(tickers), "Checking fundamentals (Finnhub)")
            if blocked():
                raise RuntimeError(f"Finnhub isn't returning data ({first_error!r}).")
            time.sleep(max(0.0, FINNHUB_PACE - (time.time() - start)))
    else:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(fetch_one, tk): tk for tk in tickers}
            for i, fut in enumerate(as_completed(futures), 1):
                tk = futures[fut]
                try:
                    record(tk, fut.result())
                except Exception as e:
                    record(tk, err=e)
                say(i, len(tickers), "Checking fundamentals")
                if blocked():
                    for f in futures:
                        f.cancel()
                    raise RuntimeError(
                        "Yahoo Finance is refusing to send company fundamentals to this "
                        "server (prices still work). This is a block on Yahoo's side.")

    if ok < MIN_COVERAGE * len(tickers):
        raise RuntimeError(
            f"{source} only sent fundamentals for {ok} of {len(tickers)} stocks, too few "
            "for trustworthy results.")

    df = pd.DataFrame(rows.values())
    if finnhub_key:
        df = normalize_units(df)
    df["source"] = source
    return df


# ---------------------------------------------------------------------------
# Rules: apply thresholds to already-downloaded data (instant)
# ---------------------------------------------------------------------------
CHECK_FIELDS = ["roe", "roa", "op_margin", "debt_to_equity", "current_ratio", "fcf_conversion",
                "revenue_growth", "earnings_growth", "trailing_pe", "forward_pe", "ev_ebitda",
                "fcf_yield", "piotroski"]


def unavailable_checks(raw):
    """Fields the data source didn't supply for ANY stock. Those checks are
    skipped (not failed), since failing every stock on them would be meaningless."""
    return [c for c in CHECK_FIELDS
            if c not in raw or pd.to_numeric(raw[c], errors="coerce").notna().sum() == 0]


def failed_checks(r, T, require_consistent_profit=True, skip=()):
    f = []
    g = lambda k: num(r.get(k))

    def check(label, key, ok):
        if key in skip:
            return
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
    skip = unavailable_checks(raw)
    fails = [failed_checks(r, T, require_consistent_profit, skip) for r in df.to_dict("records")]
    df["PASSED"] = [not f for f in fails]
    df["num_failed"] = [len(f) for f in fails]
    df["failed_checks"] = ["; ".join(f) for f in fails]
    if "fcf_yield" not in df:
        df["fcf_yield"] = None
    return df.sort_values(["PASSED", "num_failed", "fcf_yield"],
                          ascending=[False, True, False]).reset_index(drop=True)


def default_thresholds(raw):
    """Defaults, adjusted for the data source. Finnhub's F-score is out of 8
    (no dilution test), so the equivalent strict bar is 6 instead of 7."""
    T = dict(DEFAULT_T)
    if "piotroski_max" in raw and pd.to_numeric(raw["piotroski_max"], errors="coerce").max() == 8:
        T["min_piotroski"] = 6
    return T


# ---------------------------------------------------------------------------
# Command-line use
# ---------------------------------------------------------------------------
def main():
    import os
    key = os.environ.get("FINNHUB_API_KEY")

    def progress(done, total, label):
        if not label.startswith("Checking") or done % 50 == 0 or done == total:
            print(f"  {label} ... {done}/{total}" if total > 1 else f"  {label} ...")

    print(f"Fetching data with financials from {'Finnhub' if key else 'Yahoo'} "
          "(this takes several minutes) ...")
    raw = fetch_all(progress, key)
    df = evaluate(raw, default_thresholds(raw))
    skipped = unavailable_checks(raw)
    if skipped:
        print(f"  Skipped (not provided by data source): {', '.join(skipped)}")

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
