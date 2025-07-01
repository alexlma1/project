
from pairs_analysis_utils import find_stable_pairs_parallel_fast
from pairs_trader_core import PairsTrader,read_yf_data
import pandas as pd
from datetime import timedelta, datetime as dt


def convert_str_to_earnings_date(date_str):
    """
    Convert a string date in the format May 28/a to a datetime object.
    Where May 28/a represents Month, day and a for am and p for pm

    """

    if pd.isna(date_str):
        return None
    if date_str=="-":
        return None
    try:
        # find the month
        month_str, day_str = date_str.split(" ")
        month = dt.strptime(month_str, "%b").month
        # find the day
        day = int(day_str[:2])  # Remove the trailing 'a' or 'p'
        # find the hour
        hour = 0
        if day_str[-1] == 'p':
            hour = 4
        elif day_str[-1] == 'a':
            hour = 9
        year = dt.now().year  # Use the current year
        return dt(year, month, day, hour)
    except ValueError as e:
        print(f"Error converting date '{date_str}': {e}")
        return None

if __name__ == "__main__":


    stocks_table = pd.read_csv("finviz_data.csv")
    stocks_table = stocks_table[stocks_table["Market Cap"] > 5e9]
    stocks_table["Earnings"] = stocks_table["Earnings"].apply(convert_str_to_earnings_date)
    
    stocks_table["Volume"] = stocks_table["Volume"].str.replace(",", "").astype(float)
    stocks_table = stocks_table[stocks_table["Volume"] > 1e6]
    stocks_table["Volatility M"] = stocks_table["Volatility M"].str.replace("%", "").str.replace("-", "0").astype(float)
    stocks_table = stocks_table[stocks_table["Volatility M"] > 1.0]
    stocks_in_universe = read_yf_data(file_name="yf_data1.csv").columns.to_list()
    stock_universe = [x for x in stocks_table["Ticker"].values if x in stocks_in_universe]
    print(f"Using {len(stock_universe)} stocks from the universe.")


    yf_data = read_yf_data(file_name="yf_data1.csv")
    yf_data = yf_data[stock_universe]

    # stable_pairs = find_stable_pairs_by_crosscorr_parallel(
    #     yf_data,
    #     lags=range(-5, 6),
    #     min_corr=0.6,
    #     min_obs=120,
    #     min_corr_ratio=0.5,
    #     window=120,
    #     max_workers=8
    # )
    stable_pairs = find_stable_pairs_parallel_fast(
        yf_data,
        min_corr=0.4,
        max_pvalue=0.10,
        window=60,
        min_corr_ratio=0.2,
        min_coint_ratio=0.2,
        max_workers=8,
        lags=[1]
    )
    print(f"Found {len(stable_pairs)} stable pairs.")
    print(stable_pairs)
    stable_pairs.to_csv("stable_pairs.csv", index=False)
    # Set n_trials and max_workers as desired
    # results_table = monte_carlo_pairs_test_parallel(stock_universe, n_trials=100, max_workers=8)
    # results_table.to_csv("pairs_trading_results_with_durations.csv", index=False)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # print(results_table)