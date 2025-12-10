````markdown
# Risk-Conditioned Allocation Engine

Systematic allocation engine that dynamically rotates capital between **trend-following**, **mean-reversion**, and **cash** based on **volatility**, **autocorrelation**, and **risk constraints**.

The project is implemented as a backtestable research library in Python and is intended as a transparent, fully-commented example of how a professional systematic allocator might be structured.

---

## 1. High-Level Idea

The engine combines two sleeves:

- **Trend sleeve** – long-only, price–SMA momentum.
- **Mean-reversion sleeve** – Bollinger-band style, long/short.

At each date it:

1. Measures **market volatility** (EWMA) and **autocorrelation** of an equal-weight “market” basket.
2. Classifies the environment into:
   - **Trending high-vol** → allocate to trend.
   - **Mean-reverting high-vol** → allocate to MR.
   - **Noisy high-vol** → de-risk to cash.
   - **Low-vol** → diversify 50/50 across trend + MR.
3. Sizes each sleeve using:
   - **Volatility targeting** (per-sleeve target vol).
   - **Portfolio-level EWMA VaR cap**.
   - **Drawdown breaker** based on recent equity curve.
4. Blends the risky sleeve(s) with **cash at a risk-free rate**, then updates portfolio equity.

Outputs include equity curves and performance stats versus:

- Buy & Hold **SPY**
- Fixed **50/50 Trend + MR**
- **Risk-Managed Allocator**

---

## 2. Asset Universe & Data

By default the engine trades a small multi-asset universe:

```python
DEFAULT_ASSET_TICKERS = {
    "SPY": "Equities - S&P 500",
    "TLT": "Rates - US 20Y+ Treasuries",
    "GLD": "Commodities - Gold",
    "USO": "Commodities - Crude Oil (proxy)",
    "UUP": "FX - US Dollar Index (proxy for DXY)",
}
````

* Prices are pulled via [`yfinance`](https://github.com/ranaroussi/yfinance).
* Backtest frequency: **daily**.
* Returns are computed as **log returns** on adjusted close.

You can replace `DEFAULT_ASSET_TICKERS` with any set of liquid tickers supported by Yahoo Finance.

---

## 3. Model Components

### 3.1 Signal Layer

**Trend-Following**

* Signal: `+1` when price > SMA, `0` otherwise (long-only).
* SMA window (default `100` days) controlled by:

```python
trend_sma_window = config.get("trend_sma_window", 100)
```

**Mean-Reversion**

* Signal: `+1` when price < lower Bollinger band.
* Signal: `-1` when price > upper Bollinger band.
* Flat inside the band.
* Controlled by:

```python
mr_window  = config.get("mr_window", 30)
mr_num_std = config.get("mr_num_std", 2.5)
```

Signals are **lagged by 1 day** before being applied to returns to avoid look-ahead bias.

### 3.2 Volatility & Risk State

* **EWMA volatility** with decay `λ` (default `0.94`) for:

  * Market basket
  * Trend sleeve
  * MR sleeve
* **Autocorrelation** of the market basket at lag 1 over a rolling window (default `20` days).
* “High vol” is defined as the upper quantile of market vol:

```python
high_vol_quantile = config.get("high_vol_quantile", 0.75)
high_vol_thresh   = ewma_vol_market.quantile(high_vol_quantile)
```

### 3.3 Signal-to-Noise Allocator

The function:

```python
signal_to_noise_allocation(vol, rho, high_vol_threshold, rho_thresh=0.10)
```

maps `(market_vol, market_autocorr)` → `(weight_trend, weight_mr)`:

* `vol > high_vol_thresh` and `rho > +ρ*` → **1.0 trend, 0.0 MR**
* `vol > high_vol_thresh` and `rho < −ρ*` → **0.0 trend, 1.0 MR**
* `vol > high_vol_thresh` and `|rho| ≤ ρ*` → **0.0 trend, 0.0 MR (cash)**
* `vol ≤ high_vol_thresh` → **0.5 trend, 0.5 MR** (diversified low-vol regime)

where `ρ* = rho_threshold` (default `0.10`).

### 3.4 Risk Overlays

1. **Per-sleeve vol targeting**

   For each sleeve:

   ```python
   scale_trend = alloc_trend * (target_vol_sleeve / vol_trend)
   scale_mr    = alloc_mr    * (target_vol_sleeve / vol_mr)
   ```

   with `target_vol_sleeve` default `10%` annualised.

2. **Portfolio EWMA VaR cap**

   * Track EWMA variance of portfolio returns (`lam_port`, default `0.94`).

   * Compute one-day VaR at 95%:

     ```python
     z_95     = norm.ppf(0.95)
     var_prev = z_95 * daily_vol_prev
     ```

   * If `VaR > var_limit` (default `2%` of equity), scale risk down:

     ```python
     var_scale = min(1.0, var_limit / var_prev)
     ```

3. **Drawdown breaker**

   * Track equity over last `dd_window` days (default `5`).
   * If running drawdown exceeds `dd_threshold` (default `-3%`), halve the risk:

     ```python
     if dd_prev < -dd_threshold:
         dd_scale = 0.5
     ```

4. **Total scaling and cash weight**

   ```python
   total_scale = min(var_scale, dd_scale)

   if alloc_trend == 0 and alloc_mr == 0:
       r_risky = 0.0
       cash_weight = 1.0
   else:
       r_risky = total_scale * r_pre     # sleeves
       cash_weight = 1.0 - total_scale  # remaining in cash
   ```

   Cash accrues at a configurable risk-free rate (default `3%` annual).

---

## 4. Backtest API

The main entrypoint is:

```python
results = run_backtest(
    asset_tickers=None,      # dict of tickers -> description (defaults to DEFAULT_ASSET_TICKERS)
    start_date="2005-01-01",
    end_date=None,           # default: today
    config=None,             # dict overriding DEFAULT_CONFIG
)
```

### 4.1 Configuration

Key parameters (with defaults):

```python
DEFAULT_CONFIG = {
    "target_vol_sleeve": 0.10,
    "rho_threshold":     0.10,
    "var_limit":         0.02,
    "dd_threshold":      0.03,
    "lam_port":          0.94,
    "risk_free_annual":  0.03,
    "high_vol_quantile": 0.75,

    # signal parameters
    "trend_sma_window": 100,
    "mr_window":        30,
    "mr_num_std":       2.5,
}
```

You can pass a partial dict to override any subset, e.g.:

```python
my_config = {
    "target_vol_sleeve": 0.12,
    "trend_sma_window":  75,
    "mr_window":         20,
}
results = run_backtest(config=my_config)
```

### 4.2 Outputs

`run_backtest` returns a dictionary:

```python
{
    "equity_spy":          pd.Series,
    "equity_50_50":        pd.Series,
    "equity_rm":           pd.Series,   # risk-managed allocator
    "stats_df":            pd.DataFrame,

    # additional diagnostics
    "alloc_trend":         pd.Series,
    "alloc_mr":            pd.Series,
    "equity_rm_series":    pd.Series,
    "var_used":            pd.Series,
    "var_scale":           pd.Series,
    "dd_scale":            pd.Series,
    "total_scale":         pd.Series,
    "vol_market":          pd.Series,
    "rho_market":          pd.Series,
}
```

`stats_df` contains standard performance metrics (annualised, daily frequency):

* CAGR
* Annualised volatility
* Sharpe ratio
* Sortino ratio
* Maximum drawdown
* Worst 5-day return

Example usage:

```python
from risk_allocator import run_backtest   # adjust module name to your file

results      = run_backtest(start_date="2010-01-01")
stats        = results["stats_df"]
equity_rm    = results["equity_rm"]
equity_spy   = results["equity_spy"]

print(stats)
equity_rm.plot(title="Risk-Managed Allocator vs SPY")
```

---

## 5. Repository Layout 

```text
risk-conditioned-allocator/
├── risk_allocator.py       # contains run_backtest and helper functions
├── notebooks/
│   └── exploration.ipynb   # optional research / sanity checks
├── requirements.txt
└── README.md
```

If you are using a Streamlit dashboard, you might also have:

```text
├── app.py                  # Streamlit UI that calls run_backtest
└── .streamlit/
    └── config.toml
```
