import datetime as dt
from collections import deque

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


DEFAULT_ASSET_TICKERS = {
    "SPY": "Equities - S&P 500",
    "TLT": "Rates - US 20Y+ Treasuries",
    "GLD": "Commodities - Gold",
    "USO": "Commodities - Crude Oil (proxy)",
    "UUP": "FX - US Dollar Index (proxy for DXY)",
}

# default risk parameters
DEFAULT_CONFIG = {
    "target_vol_sleeve": 0.10,
    "rho_threshold": 0.10,
    "var_limit": 0.02,
    "dd_threshold": 0.03,
    "lam_port": 0.94,
    "risk_free_annual": 0.03,
    "high_vol_quantile": 0.75,

    # tuned parameters
    "trend_sma_window": 100,
    "mr_window": 30,
    "mr_num_std": 2.5,
}


#  Helper functions 

def download_price_data(ticker_list, start, end):
    """
    Download adjusted close prices for a list of tickers between start and end dates.
    Returns a DataFrame with one column per ticker.
    """
    price_data = yf.download(
        tickers=ticker_list,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    adj_close = price_data["Adj Close"].copy()
    adj_close = adj_close.dropna(how="all")
    return adj_close

# Simple moving average 
def compute_sma(price_series, window):
    return price_series.rolling(window=window, min_periods=window).mean()


def compute_bollinger_bands(price_series, window=20, num_std=2.0):
    middle_band = compute_sma(price_series, window=window)
    rolling_std = price_series.rolling(window=window, min_periods=window).std()
    upper_band = middle_band + num_std * rolling_std
    lower_band = middle_band - num_std * rolling_std
    return middle_band, upper_band, lower_band


def generate_trend_signal(price_series, sma_window=100, allow_shorts=False):
    """
    Simple trend signal: +1 if price> SMA, 0 if not (unless allow_shorts=True).
    """
    sma = compute_sma(price_series, window=sma_window)
    signal = pd.Series(index=price_series.index, dtype=float)
    signal[price_series > sma] = 1.0
    if allow_shorts:
        signal[price_series < sma] = -1.0
    else:
        signal[price_series <= sma] = 0.0
    signal[sma.isna()] = 0.0
    return signal


def generate_mean_reversion_signal(price_series, window=20, num_std=2.0, allow_shorts=True):
    """
    Bollinger-band mean reversion signal.
    """
    middle_band, upper_band, lower_band = compute_bollinger_bands(
        price_series, window=window, num_std=num_std
    )
    signal = pd.Series(index=price_series.index, dtype=float)
    signal[price_series < lower_band] = 1.0
    if allow_shorts:
        signal[price_series > upper_band] = -1.0
    signal[(price_series >= lower_band) & (price_series <= upper_band)] = 0.0
    signal[middle_band.isna()] = 0.0
    return signal


def compute_ewma_vol(returns_series, lam=0.94, annualize=True, trading_days=252):
    """
    EWMA volatility 
    """
    ewma_var = pd.Series(index=returns_series.index, dtype=float)
    seed_window = min(100, len(returns_series))
    init_var = returns_series.iloc[:seed_window].var()

    prev_var = init_var
    for t, r in enumerate(returns_series, start=0):
        prev_var = lam * prev_var + (1 - lam) * (r ** 2)
        ewma_var.iloc[t] = prev_var

    ewma_vol = np.sqrt(ewma_var)
    if annualize:
        ewma_vol = ewma_vol * np.sqrt(trading_days)
    return ewma_vol

def rolling_autocorr(series, window=20, lag=1):
    s1 = series
    s2 = series.shift(lag)
    return s1.rolling(window=window, min_periods=window).corr(s2)

def equity_and_drawdown(returns_series, initial_equity=1.0):
    equity = initial_equity * (1 + returns_series).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return equity, drawdown


def compute_performance_stats(returns, name, trading_days=252):
    """
    Performance stats helper
    """
    returns = returns.dropna()
    if len(returns) == 0:
        raise ValueError(f"No data for {name}")

    total_return = (1 + returns).prod()
    n_years = len(returns) / trading_days
    cagr = total_return ** (1 / n_years) - 1

    ann_vol = returns.std() * np.sqrt(trading_days)
    sharpe = cagr / ann_vol if ann_vol > 0 else np.nan

    downside = returns.copy()
    downside[downside > 0] = 0
    downside_dev = downside.std() * np.sqrt(trading_days)
    sortino = cagr / downside_dev if downside_dev > 0 else np.nan

    equity, drawdown = equity_and_drawdown(returns)
    max_dd = drawdown.min()

    roll_5d = (1 + returns).rolling(window=5).apply(np.prod, raw=True) - 1
    worst_5d = roll_5d.min()

    return {
        "Strategy": name,
        "CAGR": cagr,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": max_dd,
        "Worst5d": worst_5d,
    }


def signal_to_noise_allocation(vol, rho, high_vol_threshold, rho_thresh=0.10):
    """
    Your mapping from (vol, autocorr) -> Trend/MR weights.
    """
    if np.isnan(vol) or np.isnan(rho):
        return 0.0, 0.0

    # High-volatility regime
    if vol > high_vol_threshold:
        if rho > rho_thresh:
            return 1.0, 0.0       # pure trend
        elif rho < -rho_thresh:
            return 0.0, 1.0       # pure MR
        else:
            return 0.0, 0.0       # noise -> cash
    else:
        # Low-vol: diversified mix
        return 0.5, 0.5


# Streamlit Main Function 
def run_backtest(
    asset_tickers=None,
    start_date="2005-01-01",
    end_date=None,
    config=None,
):
    """
    Run the full risk-conditioned allocation backtest and return
    equity curves + stats for the dashboard.
    """

    # defaults
    if asset_tickers is None:
        asset_tickers = DEFAULT_ASSET_TICKERS

    if end_date is None:
        end_date = dt.date.today().strftime("%Y-%m-%d")

    if config is None:
        config = DEFAULT_CONFIG.copy()

    target_vol_sleeve = config.get("target_vol_sleeve", 0.10)
    rho_threshold     = config.get("rho_threshold", 0.10)
    var_limit         = config.get("var_limit", 0.02)
    dd_threshold      = config.get("dd_threshold", 0.03)
    lam_port          = config.get("lam_port", 0.94)
    risk_free_annual  = config.get("risk_free_annual", 0.03)
    high_vol_quantile = config.get("high_vol_quantile", 0.75)

    trend_sma_window = config.get("trend_sma_window", 100)
    mr_window        = config.get("mr_window", 20)
    mr_num_std       = config.get("mr_num_std", 2.0)

    risk_free_daily = risk_free_annual / 252.0

    #  Prices & returns
    price_df = download_price_data(list(asset_tickers.keys()), start_date, end_date)
    price_df = price_df.ffill().dropna(how="any")

    returns_df = np.log(price_df / price_df.shift(1)).dropna(how="any")

    # Signals: Trend & MR 
    trend_signal_df = pd.DataFrame(index=price_df.index, columns=price_df.columns, dtype=float)
    mr_signal_df    = pd.DataFrame(index=price_df.index, columns=price_df.columns, dtype=float)

    for ticker in price_df.columns:
        trend_signal_df[ticker] = generate_trend_signal(
            price_df[ticker],
            sma_window=trend_sma_window,   
            allow_shorts=False,
        )
        mr_signal_df[ticker] = generate_mean_reversion_signal(
            price_df[ticker],
            window=mr_window,              
            num_std=mr_num_std,           
            allow_shorts=True,
        )

    # Lag signals by 1 day to avoid look-ahead
    lagged_trend_signals = trend_signal_df.shift(1).reindex(returns_df.index).fillna(0.0)
    lagged_mr_signals    = mr_signal_df.shift(1).reindex(returns_df.index).fillna(0.0)

    trend_returns_per_asset = lagged_trend_signals * returns_df
    mr_returns_per_asset    = lagged_mr_signals * returns_df

    # Equal-weight across assets
    trend_returns = trend_returns_per_asset.mean(axis=1)
    mr_returns    = mr_returns_per_asset.mean(axis=1)
    trend_returns.name = "Trend"
    mr_returns.name    = "MeanReversion"

    # Equal-weight "market" proxy
    market_returns = returns_df.mean(axis=1)
    market_returns.name = "Market"

    # Vol & risk state
    ewma_vol_market = compute_ewma_vol(market_returns, lam=0.94, annualize=True)
    autocorr_market = rolling_autocorr(market_returns, window=20, lag=1)
    high_vol_thresh = ewma_vol_market.quantile(high_vol_quantile)

    ewma_vol_trend = compute_ewma_vol(trend_returns, lam=0.94, annualize=True)
    ewma_vol_mr    = compute_ewma_vol(mr_returns,    lam=0.94, annualize=True)

    # Align all series
    common_index = (
        trend_returns.index
        .intersection(mr_returns.index)
        .intersection(ewma_vol_trend.index)
        .intersection(ewma_vol_mr.index)
        .intersection(ewma_vol_market.index)
        .intersection(autocorr_market.index)
    )

    trend_r      = trend_returns.reindex(common_index).fillna(0.0)
    mr_r         = mr_returns.reindex(common_index).fillna(0.0)
    vol_trend    = ewma_vol_trend.reindex(common_index).ffill().bfill()
    vol_mr       = ewma_vol_mr.reindex(common_index).ffill().bfill()
    vol_market   = ewma_vol_market.reindex(common_index).ffill().bfill()
    rho_market   = autocorr_market.reindex(common_index).fillna(0.0)

    # Containers for simulation results
    port_returns       = pd.Series(index=common_index, dtype=float)
    alloc_trend_series = pd.Series(index=common_index, dtype=float)
    alloc_mr_series    = pd.Series(index=common_index, dtype=float)
    scale_trend_series = pd.Series(index=common_index, dtype=float)
    scale_mr_series    = pd.Series(index=common_index, dtype=float)
    var_used_series    = pd.Series(index=common_index, dtype=float)
    var_scale_series   = pd.Series(index=common_index, dtype=float)
    dd_scale_series    = pd.Series(index=common_index, dtype=float)
    total_scale_series = pd.Series(index=common_index, dtype=float)
    port_vol_series    = pd.Series(index=common_index, dtype=float)

    equity = 1.0
    equity_series = pd.Series(index=common_index, dtype=float)
    equity_series.iloc[0] = equity

    # initial daily variance guess
    daily_var_port_prev = (trend_r.std() ** 2 + mr_r.std() ** 2) / 4.0
    z_95 = norm.ppf(0.95)

    dd_window = 5
    equity_window = deque([equity], maxlen=dd_window)

    # Main risk-conditioned allocation loop
    for i, date in enumerate(common_index):
        if i == 0:
            port_returns.iloc[i]       = 0.0
            alloc_trend_series.iloc[i] = 0.0
            alloc_mr_series.iloc[i]    = 0.0
            scale_trend_series.iloc[i] = 0.0
            scale_mr_series.iloc[i]    = 0.0
            var_used_series.iloc[i]    = 0.0
            var_scale_series.iloc[i]   = 1.0
            dd_scale_series.iloc[i]    = 1.0
            total_scale_series.iloc[i] = 1.0
            port_vol_series.iloc[i]    = np.sqrt(daily_var_port_prev * 252)
            continue

        # risk state
        v_mkt = vol_market.loc[date]
        rho   = rho_market.loc[date]

        alloc_trend, alloc_mr = signal_to_noise_allocation(
            v_mkt,
            rho,
            high_vol_threshold=high_vol_thresh,
            rho_thresh=rho_threshold,
        )

        vt = max(vol_trend.loc[date], 1e-6)
        vm = max(vol_mr.loc[date],    1e-6)

        scale_trend = alloc_trend * (target_vol_sleeve / vt)
        scale_mr    = alloc_mr    * (target_vol_sleeve / vm)

        # pre-overlay portfolio return from sleeves
        r_trend = trend_r.loc[date]
        r_mr    = mr_r.loc[date]
        r_pre   = scale_trend * r_trend + scale_mr * r_mr

        # VaR scaling (using prev day's port variance)
        daily_vol_prev = np.sqrt(daily_var_port_prev)
        var_prev = z_95 * daily_vol_prev
        var_scale = 1.0
        if var_prev > var_limit:
            var_scale = var_limit / var_prev

        # Drawdown scaling (using last dd_window days of equity)
        equity_prev = equity_window[-1]
        peak_prev   = max(equity_window)
        dd_prev     = equity_prev / peak_prev - 1.0

        dd_scale = 1.0
        if dd_prev < -dd_threshold:
            dd_scale = 0.5

        total_scale = min(var_scale, dd_scale)

        # Combine risky sleeves + cash 
        if alloc_trend == 0.0 and alloc_mr == 0.0:
            r_risky = 0.0
            cash_weight = 1.0
        else:
            r_risky = total_scale * r_pre
            cash_weight = 1.0 - total_scale
            cash_weight = max(0.0, min(1.0, cash_weight))

        r_cash = cash_weight * risk_free_daily
        r_port = r_risky + r_cash

        equity = equity * (1.0 + r_port)

        # update EWMA portfolio variance
        daily_var_port_prev = lam_port * daily_var_port_prev + (1 - lam_port) * (r_port ** 2)

        equity_window.append(equity)

        # store
        port_returns.iloc[i]       = r_port
        equity_series.iloc[i]      = equity
        alloc_trend_series.iloc[i] = alloc_trend
        alloc_mr_series.iloc[i]    = alloc_mr
        scale_trend_series.iloc[i] = scale_trend
        scale_mr_series.iloc[i]    = scale_mr
        var_used_series.iloc[i]    = var_prev
        var_scale_series.iloc[i]   = var_scale
        dd_scale_series.iloc[i]    = dd_scale
        total_scale_series.iloc[i] = total_scale
        port_vol_series.iloc[i]    = np.sqrt(daily_var_port_prev * 252)

    # Benchmark series / stats
    spy_returns   = returns_df["SPY"].reindex(common_index).fillna(0.0)
    fixed_50_50   = 0.5 * trend_r + 0.5 * mr_r

    equity_spy,   dd_spy   = equity_and_drawdown(spy_returns)
    equity_50_50, dd_50_50 = equity_and_drawdown(fixed_50_50)
    equity_rm,    dd_rm    = equity_and_drawdown(port_returns)

    stats_list = [
        compute_performance_stats(spy_returns,   "Buy & Hold SPY"),
        compute_performance_stats(fixed_50_50,   "Fixed 50/50 Trend+MR"),
        compute_performance_stats(port_returns,  "Risk-Managed Allocator"),
    ]
    stats_df = pd.DataFrame(stats_list).set_index("Strategy")

    # what the dashboard consumes
    return {
        "equity_spy": equity_spy,
        "equity_50_50": equity_50_50,
        "equity_rm": equity_rm,
        "stats_df": stats_df,
        # extra series we can use for more charts later:
        "alloc_trend": alloc_trend_series,
        "alloc_mr": alloc_mr_series,
        "equity_rm_series": equity_series,
        "var_used": var_used_series,
        "var_scale": var_scale_series,
        "dd_scale": dd_scale_series,
        "total_scale": total_scale_series,
        "vol_market": ewma_vol_market,
        "rho_market": autocorr_market,
    }