

import pandas as pd
from datetime import timedelta, datetime as dt
from statsmodels.tsa.stattools import coint
from pairs_trader_core import PairsTrader, read_yf_data
from concurrent.futures import ProcessPoolExecutor, as_completed
"""
Backtesting Framework for Pairs Trading Strategies

This script loads stock pairs, applies a statistical arbitrage strategy using the 
PairsTrader class, and outputs historical performance metrics. Supports parallel 
execution for speed and scalability.

Expected Inputs:
    - stable_pairs.csv: List of stock pairs to evaluate.
    - yf_data.csv: Historical price data from Yahoo Finance.

Outputs:
    - backtest_results.csv: Performance summary for each pair.
"""

def single_pair_backtest_worker(args):
    """
    Executes the full backtest workflow for a single stock pair.

    Args:
        args: Tuple of parameters (stock1, stock2, yf_data, start_date, end_date, 
              lookback, entry_threshold, exit_threshold, max_duration)

    Returns:
        Dictionary of backtest metrics (Sharpe, returns, trade stats), or None on failure.
    """
    stock1, stock2, yf_data, start_date, end_date, lookback, entry_threshold, exit_threshold, max_duration = args
    try:
        trader = PairsTrader(
            stock1=stock1,
            stock2=stock2,
            start_date=start_date,
            end_date=end_date,
            lookback=lookback,
            yf_data=yf_data
        )
        trader.fetch_data()
        trader.run_rolling_analysis()
        trader.backtest(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            max_duration=max_duration
        )
        score = trader.score_strategy()
        score.update({
            "Stock 1": stock1,
            "Stock 2": stock2
        })
        return score
    except Exception as e:
        # Optionally log or print the error:
        # print(f"Backtest failed for {stock1}-{stock2}: {e}")
        return None

def batch_pairs_backtest_parallel(
    pairs,
    yf_data,
    start_date,
    end_date,
    lookback=130,
    entry_threshold=2,
    exit_threshold=0.3,
    max_duration=30,
    max_workers=8
):
    """
    Runs backtests for a list of stock pairs in parallel using multiple processes.

    Args:
        pairs: List of (stock1, stock2) tuples.
        yf_data: Price DataFrame with stock tickers as columns.
        start_date: Start date for backtesting (string or pd.Timestamp).
        end_date: End date for backtesting (string or pd.Timestamp).
        lookback: Rolling window size for hedge ratio estimation.
        entry_threshold: Z-score threshold for trade entry.
        exit_threshold: Z-score threshold for trade exit.
        max_duration: Max duration (in days) to hold a trade.
        max_workers: Number of parallel workers.

    Returns:
        DataFrame with backtest metrics for each pair.
    """

    args_list = [
        (stock1, stock2, yf_data, start_date, end_date, lookback, entry_threshold, exit_threshold, max_duration)
        for stock1, stock2 in pairs
    ]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(single_pair_backtest_worker, args) for args in args_list]
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)

    if results:
        df = pd.DataFrame(results)
        df = df[[
            "Stock 1", "Stock 2", "Sharpe", "Volatility", "Signal Frequency", "Total Return",
            "Average Return Per Trade", "Number of Trades", "Average Trade Duration", "Correlation"
        ]]
        return df
    else:
        return pd.DataFrame()
    
def batch_pairs_backtest(
    pairs,
    yf_data,
    start_date,
    end_date,
    lookback=130,
    entry_threshold=2,
    exit_threshold=0.3,
    max_duration=30
):
    """
    Runs backtests sequentially for a list of stock pairs.

    Args:
        pairs: List of (stock1, stock2) tuples.
        yf_data: Price DataFrame with stock tickers as columns.
        start_date: Start date for backtesting (string or pd.Timestamp).
        end_date: End date for backtesting (string or pd.Timestamp).
        lookback: Rolling window size for hedge ratio estimation.
        entry_threshold: Z-score threshold for trade entry.
        exit_threshold: Z-score threshold for trade exit.
        max_duration: Max duration (in days) to hold a trade.

    Returns:
        DataFrame with backtest metrics for each pair.
    """

    results = []
    i = 0
    for stock1, stock2 in pairs:
        try:
            trader = PairsTrader(
                stock1=stock1,
                stock2=stock2,
                start_date=start_date,
                end_date=end_date,
                lookback=lookback,
                yf_data=yf_data
            )
            trader.fetch_data()
            trader.run_rolling_analysis()
            trader.backtest(
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                max_duration=max_duration
            )
            score = trader.score_strategy()
            score.update({
                "Stock 1": stock1,
                "Stock 2": stock2
            })
            results.append(score)
        except Exception as e:
            # Optionally log or print error here
            print(f"Error processing pair ({stock1}, {stock2}): {e}")
        if i % 10 == 0:
            print(f"Processed {i} pairs...")
        i += 1

    if results:
        df = pd.DataFrame(results)
        df = df[[
            "Stock 1", "Stock 2", "Sharpe", "Volatility", "Signal Frequency", "Total Return",
            "Average Return Per Trade", "Number of Trades", "Average Trade Duration", "Correlation"
        ]]
        return df
    else:
        return pd.DataFrame()  # Empty result

if __name__ == "__main__":
    # Example usage
    pairs = pd.read_csv("stable_pairs.csv").values.tolist()  # Load pairs from CSV
    # Assuming yf_data is a DataFrame with historical prices for these stocks
    yf_data = read_yf_data(file_name="yf_data1.csv")
    
    end_date = dt.now().strftime("%Y-%m-%d")
    start_date = (pd.to_datetime(end_date) - timedelta(days=2*365)).strftime("%Y-%m-%d")
    
    # results_df = batch_pairs_backtest(
    #     pairs=pairs,
    #     yf_data=yf_data,
    #     start_date=start_date,
    #     end_date=end_date
    # )   
    results_df = batch_pairs_backtest_parallel(
        pairs=pairs,
        yf_data=yf_data,
        start_date=start_date,
        end_date=end_date
    )
    print(results_df)
    results_df.to_csv("backtest_results.csv", index=False)  # Save results to CSV