"""
Stock Screener - Streamlit app.

Run locally:   streamlit run app.py
Deploy free:   Streamlit Community Cloud (share.streamlit.io), pointing at this file.
"""
import datetime as dt
from collections import Counter

import pandas as pd
import streamlit as st

import stock_screener as sc

st.set_page_config(page_title="Stock Screener", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# Data (downloaded once, cached for 12 hours, shared by everyone using the app)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_data(_progress=None):
    return sc.fetch_all(_progress), dt.datetime.now()


def get_data():
    bar = st.progress(0.0, text="Loading market data. First load takes several minutes...")

    def progress(done, total, label):
        bar.progress(min(done / total, 1.0), text=f"{label}... {done}/{total}" if total > 1 else label)

    raw, fetched_at = load_data(progress)
    bar.empty()
    return raw, fetched_at


# ---------------------------------------------------------------------------
# Sidebar: rules
# ---------------------------------------------------------------------------
D = sc.DEFAULT_T
# (key, label, min, max, step, shown as percent?)
CONTROLS = {
    "Trend": [
        ("max_above_200dma", "Max % above 200-day avg", 0, 50, 1, True),
        ("min_momentum_12_1", "Min 12-month momentum %", -20, 30, 1, True),
    ],
    "Quality": [
        ("min_roe", "Min return on equity %", 0, 40, 1, True),
        ("min_roa", "Min return on assets %", 0, 20, 1, True),
        ("min_op_margin", "Min operating margin %", 0, 40, 1, True),
        ("max_debt_to_equity", "Max debt-to-equity %", 0, 300, 10, False),
        ("min_current_ratio", "Min current ratio", 0.5, 3.0, 0.1, False),
        ("min_fcf_conversion", "Min cash conversion (FCF ÷ earnings) %", 0, 150, 5, True),
        ("min_revenue_growth", "Min revenue growth %", -20, 30, 1, True),
        ("min_earnings_growth", "Min earnings growth %", -20, 30, 1, True),
    ],
    "Value": [
        ("max_trailing_pe", "Max trailing P/E", 5, 60, 1, False),
        ("max_forward_pe", "Max forward P/E", 5, 50, 1, False),
        ("max_ev_ebitda", "Max EV/EBITDA", 4, 40, 1, False),
        ("min_fcf_yield", "Min free-cash-flow yield %", 0, 12, 0.5, True),
    ],
    "Financial strength": [
        ("min_piotroski", "Min Piotroski F-score (0–9)", 0, 9, 1, False),
    ],
}


def reset_rules():
    for group in CONTROLS.values():
        for key, *_ , pct in group:
            st.session_state[key] = round(D[key] * 100, 2) if pct else D[key]
    st.session_state["exclude_cyclicals"] = True
    st.session_state["consistent_profit"] = True


if "min_roe" not in st.session_state:
    reset_rules()

with st.sidebar:
    st.header("Rules")
    exclude_cyclicals = st.toggle(
        "Exclude Energy & Materials", key="exclude_cyclicals",
        help="Commodity producers look cheapest at the peak of their cycle, "
             "which fools this kind of screen.")
    consistent = st.toggle(
        "Require profit every year", key="consistent_profit",
        help="Net income positive in every annual report Yahoo has (usually 4 years).")

    T = {}
    for group, items in CONTROLS.items():
        with st.expander(group, expanded=False):
            for key, label, lo, hi, step, pct in items:
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
# Main page
# ---------------------------------------------------------------------------
st.title("📈 Stock Screener")
st.caption("Great businesses at fair prices, already in an uptrend. S&P 500, "
           "excluding banks and REITs. A pass means *worth researching*, not *buy*.")

try:
    raw, fetched_at = get_data()
except Exception as e:
    st.error(f"Couldn't load market data: {e}\n\nYahoo may be rate-limiting. "
             "Wait a few minutes, then tap **Refresh market data**.")
    st.stop()

df = sc.evaluate(raw, T, exclude_cyclicals, consistent)
passed = df[df["PASSED"]]
near = df[(~df["PASSED"]) & (df["num_failed"] == 1)]

c1, c2, c3 = st.columns(3)
c1.metric("Stocks scanned", len(df))
c2.metric("Passed every check", len(passed))
c3.metric("Missed by one check", len(near))
st.caption(f"Market data from {fetched_at:%b %d, %I:%M %p}. Changing rules updates instantly.")

PCT_COLS = ["pct_above_200dma", "momentum_12_1", "roe", "roa", "op_margin", "fcf_yield",
            "fcf_conversion", "revenue_growth", "earnings_growth"]


def show(table, cols):
    t = table[cols].copy()
    for c in PCT_COLS:
        if c in t:
            t[c] = pd.to_numeric(t[c], errors="coerce") * 100
    cfg = {
        "ticker": st.column_config.TextColumn("Ticker"),
        "name": st.column_config.TextColumn("Company"),
        "sector": st.column_config.TextColumn("Sector"),
        "price": st.column_config.NumberColumn("Price", format="$%.2f"),
        "trailing_pe": st.column_config.NumberColumn("P/E", format="%.1f"),
        "forward_pe": st.column_config.NumberColumn("Fwd P/E", format="%.1f"),
        "fcf_yield": st.column_config.NumberColumn("FCF yield", format="%.1f%%"),
        "roe": st.column_config.NumberColumn("ROE", format="%.1f%%"),
        "op_margin": st.column_config.NumberColumn("Op margin", format="%.1f%%"),
        "debt_to_equity": st.column_config.NumberColumn("Debt/Eq", format="%.0f%%"),
        "piotroski": st.column_config.NumberColumn("F-score", format="%d"),
        "failed_checks": st.column_config.TextColumn("Failed", width="large"),
    }
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

with st.expander("How this works"):
    st.markdown("""
A stock passes only if it clears **every** rule in four gates:

- **Trend:** above its 200-day average, 50-day above 200-day, positive 12-month momentum,
  and not stretched too far above the 200-day (avoids chasing).
- **Quality:** high returns on equity and assets, strong margins, modest debt, enough cash
  to cover short-term bills, earnings backed by real cash, growing sales and profits, and
  profitable every recent year.
- **Value:** reasonable P/E, forward P/E, EV/EBITDA, and a solid free-cash-flow yield.
- **Strength:** Piotroski F-score, 9 yes/no tests of whether the business improved vs. last year.

Missing data counts as a fail. Data comes from Yahoo Finance (unofficial, sometimes wrong),
so verify a company's numbers before acting. This is a research filter, not a prediction,
and it hasn't been backtested.
""")
