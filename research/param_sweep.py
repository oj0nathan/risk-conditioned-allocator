import itertools

import pandas as pd

import os
import sys

# Make sure Python can see the project root (where risk_engine lives)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from risk_engine.core import run_backtest, DEFAULT_CONFIG

import itertools

import pandas as pd

from risk_engine.core import run_backtest, DEFAULT_CONFIG


def main():
    # Define the grid of parameters to test
    trend_windows = [50, 100, 150]         # SMA lengths
    mr_windows    = [10, 20, 30]           # Bollinger window
    mr_stds       = [1.5, 2.0, 2.5]        # Bollinger width

    results = []

    for sma_w, mr_w, mr_std in itertools.product(trend_windows, mr_windows, mr_stds):
        cfg = DEFAULT_CONFIG.copy()
        cfg["trend_sma_window"] = sma_w
        cfg["mr_window"] = mr_w
        cfg["mr_num_std"] = mr_std

        print(f"Running SMA={sma_w}, MR_win={mr_w}, MR_std={mr_std}...")

        res = run_backtest(
            start_date="2005-01-01",
            end_date=None,            # up to today
            config=cfg,
        )

        stats_df = res["stats_df"]
        rm_row = stats_df.loc["Risk-Managed Allocator"]

        results.append(
            {
                "trend_sma_window": sma_w,
                "mr_window": mr_w,
                "mr_num_std": mr_std,
                "CAGR": rm_row["CAGR"],
                "AnnVol": rm_row["AnnVol"],
                "Sharpe": rm_row["Sharpe"],
                "Sortino": rm_row["Sortino"],
                "MaxDD": rm_row["MaxDD"],
                "Worst5d": rm_row["Worst5d"],
            }
        )

    # Collect into a DataFrame and sort
    results_df = pd.DataFrame(results)

    # Example: sort by Sharpe, then by MaxDD (higher Sharpe, less negative DD)
    results_df = results_df.sort_values(
        by=["Sharpe", "MaxDD"],
        ascending=[False, True],
    )

    print("\nTop parameter sets by Sharpe (best first):")
    print(
        results_df.head(15).to_string(
            index=False,
            float_format=lambda x: f"{x:0.2%}" if abs(x) < 1 else f"{x:0.3f}",
        )
    )

    # Optionally save to CSV for later inspection in Excel
    results_df.to_csv("param_sweep_results.csv", index=False)


if __name__ == "__main__":
    main()
