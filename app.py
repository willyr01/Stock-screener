"""
Stock Screener - Streamlit app.

Run locally:   streamlit run app.py
Deploy free:   Streamlit Community Cloud (share.streamlit.io), pointing at this file.

Company financials come from Finnhub when FINNHUB_API_KEY is set in the app's
Secrets, otherwise from Yahoo. Prices always come from Yahoo.
"""
import datetime as dt
from collections import Counter

import pandas as pd
import streamlit as st

import stock_screener as sc

st.set_page_config(page_title="Stock Screener", page_icon="📈", layout="wide")


def finnhub_key():
    try:
        return st.secrets.get("FINNHUB_API_KEY") or None
    except Exception:          # no Secrets configured at all
        return None


# ---------------------------------------------------------------------------
# Data (downloaded once, cached for 12 hours, shared by everyone using the app)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_data(key, _progress=None):
    return sc.fetch_all(_progress, key), dt.datetime.now()


def get_data(key):
    bar = st.progress(0.0, text="Loading market data. First load takes about 8 minutes...")

    def progress(done, total, label):
        bar.progress(min(done / total, 1.0), text=f"{label}... {done}/{total}" if total > 1 else label)

    raw, fetched_at = load_data(key, progress)
    bar.empty()
    return raw, fetched_at


st.title("📈 Stock Screener")
st.caption("Great businesses at fair prices, already in an uptrend. S&P 500, "
           "excluding banks and REITs. A pass means *worth researching*, not *buy*.")

KEY = finnhub_key()
try:
    raw, fetched_at = get_data(KEY)
except Exception as e:
    hint = ("Check that the **FINNHUB_API_KEY** line in Secrets is correct, then tap "
            "**Refresh market data**." if isinstance(e, sc.FinnhubKeyError) else
            "Nothing was saved, so the app will try again fresh. Wait 15–30 minutes, then "
            "tap **Refresh market data** in the sidebar.")
    st.error(f"**Couldn't load market data.** {e}\n\n{hint}")
    if st.sidebar.button("Refresh market data", use_container_width=True):
        load_data.clear()
        st.rerun()
    st.stop()

SOURCE = raw["source"].iloc[0] if "source" in raw else "Yahoo"
SKIPPED = set(sc.unavailable_checks(raw))
D = sc.default_thresholds(raw)
F_MAX = int(pd.to_numeric(raw.get("piotroski_max"), errors="coerce").max()) \
    if "piotroski_max" in raw and raw["piotroski_max"].notna().any() else 9

# ---------------------------------------------------------------------------
# Sidebar: rules
# ---------------------------------------------------------------------------
# (key, data field it applies to, label, min, max, step, shown as percent?)
CONTROLS = {
    "Trend": [
        ("max_above_200dma", None, "Max % above 200-day avg", 0, 50, 1, True),
        ("min_momentum_12_1", None, "Min 12-month momentum %", -20, 30, 1, True),
    ],
    "Quality": [
        ("min_roe", "roe", "Min return on equity %", 0, 40, 1, True),
        ("min_roa", "roa", "Min return on assets %", 0, 20, 1, True),
        ("min_op_margin", "op_margin", "Min operating margin %", 0, 40, 1, True),
        ("max_debt_to_equity", "debt_to_equity", "Max debt-to-equity %", 0, 300, 10, False),
        ("min_current_ratio", "current_ratio", "Min current ratio", 0.5, 3.0, 0.1, False),
        ("min_fcf_conversion", "fcf_conversion", "Min cash conversion (FCF ÷ earnings) %", 0, 150, 5, True),
        ("min_revenue_growth", "revenue_growth", "Min revenue growth %", -20, 30, 1, True),
        ("min_earnings_growth", "earnings_growth", "Min earnings growth %", -20, 30, 1, True),
    ],
    "Value": [
        ("max_trailing_pe", "trailing_pe", "Max trailing P/E", 5, 60, 1, False),
        ("max_forward_pe", "forward_pe", "Max forward P/E", 5, 50, 1, False),
        ("max_ev_ebitda", "ev_ebitda", "Max EV/EBITDA", 4, 40, 1, False),
        ("min_fcf_yield", "fcf_yield", "Min free-cash-flow yield %", 0, 12, 0.5, True),
    ],
    "Financial strength": [
        ("min_piotroski", "piotroski", f"Min Piotroski F-score (0–{F_MAX})", 0, F_MAX, 1, False),
    ],
}


def reset_rules():
    for group in CONTROLS.values():
        for key, *_, pct in group:
            st.session_state[key] = round(D[key] * 100, 2) if pct else D[key]
    st.session_state["exclude_cyclicals"] = True
    st.session_state["consistent_profit"] = True
    st.session_state["rules_for"] = SOURCE


# Set defaults on first visit, and again if the data source changed.
if st.session_state.get("rules_for") != SOURCE:
    reset_rules()

with st.sidebar:
    st.header("Rules")
    exclude_cyclicals = st.toggle(
        "Exclude Energy & Materials", key="exclude_cyclicals",
        help="Commodity producers look cheapest at the peak of their cycle, "
             "which fools this kind of screen.")
    consistent = st.toggle(
        "Require profit every year", key="consistent_profit",
        help="Positive profit in each of the last 3–4 annual reports.")

    T = {}
    for group, items in CONTROLS.items():
        with st.expander(group, expanded=False):
            for key, field, label, lo, hi, step, pct in items:
                if field in SKIPPED:
                    continue
                if isinstance(step, int) and not pct and isinstance(D[key], int):
                    v = st.slider(label, int(lo), int(hi), step=int(step), key=key)
                else:
                    v = st.slider(label, float(lo), float(hi), step=float(step), key=key)
                T[key] = v / 100 if pct else v

    st.button("Reset to defaults", on_click=reset_rules, use_container_width=True)
    if st.button("Refresh market data", use_container_width=True):
        load_data.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
have = int(raw["data_error"].isna().sum()) if "data_error" in raw else len(raw)
df = sc.evaluate(raw, {**D, **T}, exclude_cyclicals, consistent)
passed = df[df["PASSED"]]
near = df[(~df["PASSED"]) & (df["num_failed"] == 1)]

c1, c2, c3 = st.columns(3)
c1.metric("Stocks scanned", len(df))
c2.metric("Passed every check", len(passed))
c3.metric("Missed by one check", len(near))
st.caption(f"Data from {fetched_at:%b %d, %I:%M %p} · prices from Yahoo, financials from "
           f"{SOURCE} for {have} of {len(raw)} stocks. Changing rules updates instantly.")

PCT_COLS = ["pct_above_200dma", "momentum_12_1", "roe", "roa", "op_margin", "fcf_yield",
            "fcf_conversion", "revenue_growth", "earnings_growth"]
LABELS = {
    "ticker": "Ticker", "name": "Company", "sector": "Sector", "price": "Price",
    "trailing_pe": "P/E", "forward_pe": "Fwd P/E", "fcf_yield": "FCF yield", "roe": "ROE",
    "roa": "ROA", "op_margin": "Op margin", "debt_to_equity": "Debt/Eq",
    "current_ratio": "Current ratio", "fcf_conversion": "Cash conversion",
    "revenue_growth": "Revenue growth", "earnings_growth": "EPS growth",
    "piotroski": "F-score", "failed_checks": "Failed", "num_failed": "# failed",
}


def show(table, cols):
    cols = [c for c in cols if c not in SKIPPED and c in table]
    t = table[cols].copy()
    for c in PCT_COLS:
        if c in t:
            t[c] = pd.to_numeric(t[c], errors="coerce") * 100
    fmt = {"price": "$%.2f", "trailing_pe": "%.1f", "forward_pe": "%.1f",
           "debt_to_equity": "%.0f%%", "current_ratio": "%.2f", "piotroski": "%d"}
    cfg = {}
    for c in cols:
        if c in ("ticker", "name", "sector"):
            cfg[c] = st.column_config.TextColumn(LABELS[c])
        elif c == "failed_checks":
            cfg[c] = st.column_config.TextColumn(LABELS[c], width="large")
        else:
            cfg[c] = st.column_config.NumberColumn(
                LABELS.get(c, c), format="%.1f%%" if c in PCT_COLS else fmt.get(c, "%.2f"))
    st.dataframe(t, column_config=cfg, hide_index=True, use_container_width=True)


st.subheader("Passed every check")
if passed.empty:
    st.info("Nothing passes today. That's normal for a strict screen. Check the near misses, "
            "or loosen a rule in the sidebar to see what's close.")
else:
    show(passed, ["ticker", "name", "sector", "price", "trailing_pe", "forward_pe",
                  "fcf_yield", "roe", "op_margin", "piotroski"])

st.subheader("Near misses")
if near.empty:
    st.write("None.")
else:
    show(near, ["ticker", "name", "sector", "failed_checks", "trailing_pe",
                "fcf_yield", "roe", "debt_to_equity", "piotroski"])

with st.expander("Which rules knock out the most stocks?"):
    counts = Counter(
        c.replace(": no data", " (no data)") for fc in df["failed_checks"] for c in fc.split("; ") if c)
    if counts:
        chart = pd.DataFrame(counts.most_common(), columns=["rule", "stocks failed"]).set_index("rule")
        st.bar_chart(chart, horizontal=True)

with st.expander("All stocks"):
    show(df, ["ticker", "name", "sector", "num_failed", "failed_checks", "price",
              "trailing_pe", "fcf_yield", "roe"])
    st.download_button("Download full results (CSV)", df.to_csv(index=False).encode(),
                       f"screen_results_{dt.date.today():%Y-%m-%d}.csv", "text/csv",
                       use_container_width=True)

with st.expander("Data check"):
    st.write("How many stocks have each number. A field near zero means the data source "
             "isn't supplying it.")
    cov = pd.DataFrame(
        [(LABELS.get(c, c), int(pd.to_numeric(raw[c], errors="coerce").notna().sum()) if c in raw else 0)
         for c in sc.CHECK_FIELDS], columns=["Number", f"Stocks with data (of {len(raw)})"])
    st.dataframe(cov, hide_index=True, use_container_width=True)
    if SKIPPED:
        st.caption("Skipped because the data source doesn't provide them: "
                   + ", ".join(LABELS.get(c, c) for c in SKIPPED) + ".")
    st.write("Spot-check a few well-known companies against another site. ROE and margins "
             "should look like normal percentages, for example 15%, not 1500% or 0.15%.")
    sample = raw[raw["ticker"].isin(["AAPL", "MSFT", "JNJ", "PG", "HD"])]
    show(sample, ["ticker", "name", "roe", "op_margin", "debt_to_equity", "current_ratio",
                  "trailing_pe", "fcf_yield", "revenue_growth", "piotroski"])

with st.expander("How this works"):
    st.markdown(f"""
A stock passes only if it clears **every** rule in four gates:

- **Trend:** above its 200-day average, 50-day above 200-day, positive 12-month momentum,
  and not stretched too far above the 200-day (avoids chasing).
- **Quality:** high returns on equity and assets, strong margins, modest debt, enough cash
  to cover short-term bills, earnings backed by real cash, growing sales and profits, and
  profitable every recent year.
- **Value:** reasonable P/E and a solid free-cash-flow yield (plus forward P/E and EV/EBITDA
  when the data source provides them).
- **Strength:** a Piotroski F-score, {F_MAX} yes/no tests of whether the business improved vs.
  last year.

Missing data counts as a fail. Prices come from Yahoo and company financials from {SOURCE}.
Verify a company's numbers before acting. This is a research filter, not a prediction,
and it hasn't been backtested.
""")
