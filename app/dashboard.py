import os
import sys

import pandas as pd
import streamlit as st

# Make sure Python can find the project root (where risk_engine lives) 
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from risk_engine import run_backtest

st.set_page_config(
    page_title="Macro Risk-Conditioned Allocation Dashboard",
    layout="wide",
)

st.title("Macro Risk-Conditioned Allocation Dashboard")

st.write(
    "Backtest of a dynamic Trend / Mean-Reversion allocator with "
    "volatility targeting, VaR cap, and drawdown breaker on a multi-asset universe."
)

st.sidebar.header("Backtest Window")
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2005-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("today"))

st.sidebar.header("Risk Parameters")
target_vol = st.sidebar.slider("Target sleeve vol (ann.)", 0.05, 0.25, 0.10, step=0.01)
var_limit = st.sidebar.slider("VaR limit (1d 95%, fraction of equity)", 0.01, 0.05, 0.02, step=0.005)
dd_threshold = st.sidebar.slider("Drawdown threshold (5-day)", 0.01, 0.10, 0.03, step=0.005)
rho_threshold = st.sidebar.slider("Autocorr threshold (|ρ| to classify trend vs MR)", 0.0, 0.3, 0.10, step=0.01)

# Signal parameter controls
st.sidebar.header("Signal Parameters")
trend_sma = st.sidebar.slider("Trend SMA window (days)", 50, 200, 100, step=10)
mr_window = st.sidebar.slider("MR window (days)", 10, 60, 30, step=5)
mr_num_std = st.sidebar.slider("MR band width (std dev)", 1.0, 3.0, 2.5, step=0.25)

# Build config dict *after* reading all sliders
config = {
    "target_vol_sleeve": target_vol,
    "rho_threshold": rho_threshold,
    "var_limit": var_limit,
    "dd_threshold": dd_threshold,
    "lam_port": 0.94,
    "risk_free_annual": 0.03,
    "high_vol_quantile": 0.75,
    # pass tuned signal params into the engine
    "trend_sma_window": trend_sma,
    "mr_window": mr_window,
    "mr_num_std": mr_num_std,
}



@st.cache_data
def run_model(start_date_str: str, end_date_str: str, cfg: dict):
    """Wrapper so Streamlit can cache the backtest results."""
    return run_backtest(
        start_date=start_date_str,
        end_date=end_date_str,
        config=cfg,
    )


results = run_model(
    start_date.strftime("%Y-%m-%d"),
    end_date.strftime("%Y-%m-%d"),
    config,
)

equity_spy = results["equity_spy"]
equity_50_50 = results["equity_50_50"]
equity_rm = results["equity_rm"]
stats_df = results["stats_df"]

alloc_trend = results["alloc_trend"]
alloc_mr = results["alloc_mr"]
vol_market = results["vol_market"]
rho_market = results["rho_market"]

var_used = results["var_used"]
var_scale = results["var_scale"]

# Implied cash allocation
alloc_cash = 1.0 - (alloc_trend + alloc_mr)
alloc_cash = alloc_cash.clip(lower=0.0)

# Top row: Equity curves + KPIs / stats
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Equity Curves")

    equity_df = pd.concat([equity_spy, equity_50_50, equity_rm], axis=1)
    equity_df.columns = ["SPY (Buy & Hold)", "50/50 Trend+MR", "Risk-Managed Allocator"]

    st.line_chart(equity_df)

with col_right:
    st.subheader("Key Metrics (Risk-Managed vs SPY)")

    rm_row = stats_df.loc["Risk-Managed Allocator"]
    spy_row = stats_df.loc["Buy & Hold SPY"]

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("CAGR (RM)", f"{rm_row['CAGR']:.2%}")
    with k2:
        st.metric("Ann. Vol (RM)", f"{rm_row['AnnVol']:.2%}")
    with k3:
        st.metric("Max DD (RM)", f"{rm_row['MaxDD']:.1%}")

    st.markdown("**Full Performance & Risk Table**")
    st.dataframe(
        stats_df.style.format(
            {
                "CAGR": "{:.2%}",
                "AnnVol": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Sortino": "{:.2f}",
                "MaxDD": "{:.1%}",
                "Worst5d": "{:.1%}",
            }
        )
    )
# Zoomed-in recent period
st.markdown("---")
st.subheader("Recent Performance (2022–Now)")

zoom_start = "2022-01-01"
equity_zoom = equity_df[equity_df.index >= zoom_start]
st.line_chart(equity_zoom)

st.markdown("---")
st.subheader("Risk-Conditioned Allocation & Market Risk State")

col_alloc, col_risk, col_var = st.columns(3)

with col_alloc:
    st.markdown("**Dynamic Allocation Weights**")

    alloc_df = pd.concat([alloc_trend, alloc_mr, alloc_cash], axis=1)
    alloc_df.columns = ["Trend", "Mean Reversion", "Cash"]

    st.area_chart(alloc_df)

with col_risk:
    st.markdown("**Market Risk State (Volatility & Autocorrelation)**")

    risk_df = pd.concat([vol_market, rho_market], axis=1)
    risk_df.columns = ["EWMA Vol (ann.)", "20d Autocorr"]

    st.line_chart(risk_df)

with col_var:
    st.markdown("**VaR Usage Monitor (1d, 95%)**")

    var_limit_series = pd.Series(var_limit, index=var_used.index, name="VaR Limit")
    var_df = pd.concat([var_used, var_limit_series], axis=1)
    var_df.columns = ["Estimated VaR", "VaR Limit"]

    st.line_chart(var_df)

    # Breach indicator (1 when VaR cap is binding)
    breaches = (var_used > var_limit).astype(float)
    breaches.name = "VaR cap active (1 = on)"

    st.area_chart(breaches)
    st.caption(
        "Shaded periods show days where the estimated 1d 95% VaR exceeds the limit "
        "and the VaR cap forces position scaling."
    )