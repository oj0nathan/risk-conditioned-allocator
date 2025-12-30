# Risk-Conditioned Allocation Engine for Trend / Mean-Reversion Sleeves
This project tests whether a simple **risk-conditioned allocator** can systematically improve the way capital is split between trend-following, mean-reversion, and cash. The engine trades a small multi-asset universe, builds trend and mean-reversion signals on each asset, and then uses market volatility, autocorrelation, and portfolio risk constraints (VaR and drawdown) to scale exposures dynamically.

The code is written in Python using pandas, NumPy, SciPy, yfinance, and is structured as a reusable backtest function (`run_backtest`) that returns equity curves and performance statistics.

---

TL;DR

* Construct long-only **trend** signals from price vs SMA (e.g. 100-day moving average) and **mean-reversion** signals from Bollinger bands (price vs rolling mean ± k·σ).
* Apply signals cross-sectionally across a small multi-asset universe (SPY, TLT, GLD, USO, UUP), then equal-weight across assets to form:

  * a **Trend sleeve** and
  * a **Mean-Reversion sleeve**.
* Estimate **EWMA volatility** for the market basket, trend sleeve, and MR sleeve, plus **rolling autocorrelation** of the market basket.
* Define a **signal-to-noise map** from (market volatility, autocorrelation) to sleeve weights:

  * High vol & positive autocorr → allocate to **trend**
  * High vol & negative autocorr → allocate to **mean reversion**
  * High vol & near-zero autocorr → treat as **noise**, sit in cash
  * Low vol → **50/50** trend + MR mix.
* Vol-target each sleeve to a **per-sleeve target vol** (e.g. 10% annualised) and then apply:

  * a **portfolio EWMA VaR cap** (one-day 95% VaR ≤ 2% of equity), and
  * a **drawdown breaker** (risk cuts when recent drawdown exceeds 3% over 5 days).
* Combine risk-scaled sleeves with cash at a configurable **risk-free rate** and backtest the resulting portfolio vs:

  * Buy & hold SPY
  * Fixed 50/50 Trend + MR.
* Output equity curves and performance stats (CAGR, vol, Sharpe, Sortino, max drawdown, worst 5-day loss) for all three strategies.

This is a research/learning project – not investment advice.

---

What this project demonstrates

* A clean implementation of a **multi-sleeve allocation engine** that routes capital between trend, mean-reversion, and cash using simple, interpretable rules.
* Use of **EWMA volatility** and **autocorrelation** as a compact “market state” proxy, separating:

  * *what* the signals want to do (trend vs MR) from
  * *how much* risk the portfolio should run.
* Practical **risk management overlays**:

  * per-sleeve volatility targeting,
  * portfolio-level EWMA VaR cap,
  * drawdown-based risk throttling.
* Careful handling of **look-ahead bias**:

  * signals are lagged by one day before being applied to returns,
  * EWMA risk metrics and VaR use information available only up to the previous day.
* A benchmarked comparison vs naive alternatives (SPY buy-and-hold and fixed 50/50 Trend+MR), with a consistent performance-statistics framework.

---

Data & Universe

Asset universe (downloaded from Yahoo Finance via `yfinance`):

* **SPY** – Equities: S&P 500
* **TLT** – Rates: US 20Y+ Treasuries
* **GLD** – Commodities: Gold
* **USO** – Commodities: Crude Oil proxy
* **UUP** – FX: US Dollar Index proxy

Implementation details:

* Uses **adjusted close** prices from Yahoo Finance.
* Computes **daily log returns**.
* Trend and mean-reversion signals are built on prices; portfolio construction and risk management operate on daily returns.
* You can override the default universe by passing a custom `asset_tickers` dict into `run_backtest`.

---

Model Overview

**Signal Layer**

* Trend signal per asset:

  * SMA-based rule: `+1` when price > SMA, `0` otherwise (long-only by default).
  * SMA horizon configurable via `trend_sma_window`.
* Mean-reversion signal per asset:

  * Bollinger-band rule:

    * `+1` when price < lower band (oversold),
    * `-1` when price > upper band (overbought),
    * `0` inside the band.
  * Controlled by `mr_window` (lookback) and `mr_num_std` (band width).
* Signals are **lagged by 1 day** before they are multiplied with returns → avoids trading on information from the same day’s close.
* Sleeve returns:

  * Trend sleeve = equal-weight of all asset-level trend-signal returns.
  * MR sleeve = equal-weight of all asset-level MR-signal returns.
  * Market basket = equal-weight of raw asset returns.

**Market State & Signal-to-Noise Allocator**

* Compute **EWMA volatility** for the market basket (and sleeves) with decay λ (e.g. 0.94).

* Compute **rolling autocorrelation** of market returns (lag 1, e.g. 20-day window).

* Define a “high volatility” threshold as a quantile (e.g. 75th percentile) of EWMA vol.

* Map `(vol_market, rho_market)` → `(weight_trend, weight_mr)`:

  * If `vol > high_vol_thresh`:

    * `rho > +ρ*` → (1.0, 0.0)  → **pure trend**
    * `rho < −ρ*` → (0.0, 1.0)  → **pure mean reversion**
    * `|rho| ≤ ρ*` → (0.0, 0.0) → **noise → cash**
  * If `vol ≤ high_vol_thresh`:

    * `(0.5, 0.5)` → **balanced mix**.

* Thresholds `ρ*` and the vol quantile are configurable via the `config` dict.

**Risk Overlays**

1. **Per-sleeve Volatility Targeting**

   * Compute EWMA vol for trend and MR sleeves.
   * Scale sleeves to hit a per-sleeve target vol, e.g.:

     `scale_trend = alloc_trend * target_vol_sleeve / vol_trend`
     `scale_mr    = alloc_mr    * target_vol_sleeve / vol_mr`

2. **Portfolio EWMA VaR Cap**

   * Maintain EWMA variance of daily portfolio returns using parameter `lam_port`.
   * Compute one-day 95% VaR: `VaR = z_95 * daily_vol_prev`.
   * If `VaR > var_limit` (e.g. 2% of equity), multiply risky exposure by `var_limit / VaR`.

3. **Drawdown Breaker**

   * Maintain a rolling window of recent equity values (e.g. 5 days).
   * Compute running peak and current drawdown.
   * If drawdown < −`dd_threshold` (e.g. −3%), cut risk (e.g. halve exposures).

4. **Cash & Risk-Free Return**

   * If both sleeves are turned off (trend=0, MR=0), portfolio sits fully in cash.
   * Otherwise, risky sleeves are scaled by the minimum of VaR- and DD-scales; residual capital sits in cash and earns a configurable risk-free rate (default 3% annualised).

**Backtest & Benchmarks**

* The main function `run_backtest` orchestrates:

  1. Downloading prices and computing daily log returns.
  2. Building trend and MR signals and sleeves.
  3. Estimating EWMA vol, autocorrelation, and portfolio variance.
  4. Running the day-by-day allocation loop with risk overlays.
  5. Computing equity curves and performance statistics.

* Benchmarks:

  * **Buy & Hold SPY**.
  * **Fixed 50/50 Trend + MR** (equal risk exposure to both sleeves, no dynamic risk overlays).
  * **Risk-Managed Allocator** (full engine).

* Performance metrics (daily → annualised) include:

  * CAGR
  * Annualised volatility
  * Sharpe ratio
  * Sortino ratio
  * Maximum drawdown
  * Worst 5-day return

---

Data & Implementation Details

* Prices from Yahoo Finance via `yfinance.download`.
* Pandas Series/DataFrames used for all time series.
* NumPy used for vectorised operations and rolling calculations.
* SciPy (`scipy.stats.norm`) used for normal quantiles in VaR.

The whole backtest is encapsulated in `run_backtest(...)`, which returns a dictionary of equity curves, risk diagnostics, and performance statistics that can be easily plugged into notebooks or a Streamlit dashboard.

---

Disclaimer

This repository is for **educational and research purposes only**. It is not investment advice, an offer, or a recommendation to buy or sell any security or to implement this strategy in live trading. The backtests ignore many real-world frictions (transaction costs, liquidity constraints, taxes, slippage, borrow availability). Use at your own risk.
