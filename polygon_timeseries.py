import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

def fast_parse(df):
    """
    Vectorized OCC option symbol parser.
    """
    tickers = df['ticker'].astype(str)
    df['root_symbol'] = tickers.str.slice(0, -15)
    df['expiration_date'] = pd.to_datetime(tickers.str.slice(-15, -9), format='%y%m%d', errors='coerce')
    df['option_type'] = tickers.str.slice(-9, -8)
    df['strike_price'] = tickers.str.slice(-8, step=1).astype(float) / 1000.0
    
    return df

def build_time_series_fast(data_dir, hdf_path="contract_timeseries_2025.h5"):
    """
    Fast builder: appends parsed daily data to a single table (queryable by ticker/date).
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("2025-*.csv.gz"))
    store = pd.HDFStore(hdf_path, mode='a')

    for file in tqdm(files, desc="📅 Processing daily files"):
        date = pd.to_datetime(file.stem[:10])
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"❌ Error reading {file.name}: {e}")
            continue

        if df.empty or 'ticker' not in df.columns:
            continue

        df['date'] = date
        try:
            df = fast_parse(df)
        except Exception as e:
            print(f"❌ Parse error in {file.name}: {e}")
            continue

        try:
            store.append("/full_dataset", df, format='table', data_columns=["ticker", "date"])
        except Exception as e:
            print(f"❌ Failed to write {file.name}: {e}")

    store.close()
    print(f"\n✅ Appended all data to: {hdf_path}")

def build_combined_time_series(data_dirs, hdf_path="contract_timeseries_combined.h5"):
    """Build combined HDF5 file from multiple year directories."""
    store = pd.HDFStore(hdf_path, mode='a')
    for year_dir in data_dirs:
        data_dir = Path(year_dir)
        files = sorted(data_dir.glob("20*.csv.gz"))
        for file in tqdm(files, desc=f"📅 Processing {data_dir.name}"):
            date = pd.to_datetime(file.stem[:10])
            try:
                df = pd.read_csv(file)
            except Exception as e:
                print(f"❌ Error reading {file.name}: {e}")
                continue

            if df.empty or 'ticker' not in df.columns:
                continue

            df['date'] = date
            try:
                df = fast_parse(df)
            except Exception as e:
                print(f"❌ Parse error in {file.name}: {e}")
                continue

            try:
                store.append("/full_dataset", df, format='table', data_columns=["ticker", "date", "root_symbol"])
            except Exception as e:
                print(f"❌ Failed to write {file.name}: {e}")
    store.close()
    print(f"\n✅ Combined data written to: {hdf_path}")
def fast_filter_contract_data(hdf_input, hdf_output, spot_path):
    store_in = pd.HDFStore(hdf_input, mode='r')
    store_out = pd.HDFStore(hdf_output, mode='w')
    spot_df = pd.read_csv(spot_path, parse_dates=['Date']).set_index('Date')

    # Read only dates with unique values once
    available_dates = pd.to_datetime(store_in.select_column("/full_dataset", "date")).dropna().unique()

    for date in tqdm(available_dates, desc="📅 Filtering by date"):
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        try:
            daily_df = store_in.select("/full_dataset", where=f'date == "{date_str}"')
        except Exception:
            continue

        if daily_df.empty or date not in spot_df.index:
            continue

        spot_row = spot_df.loc[date]
        spot_row = spot_row.dropna()
        if spot_row.empty:
            continue

        # Filter only relevant root symbols
        symbols = spot_row.index.tolist()
        root_symbols = [f"O:{s}" for s in symbols]
        daily_df = daily_df[daily_df["root_symbol"].isin(root_symbols)].copy()

        if daily_df.empty:
            continue

        daily_df["dte"] = (daily_df["expiration_date"] - date).dt.days

        # Merge spot price for moneyness calculation
        spot_map = {f"O:{sym}": spot_row[sym] for sym in symbols}
        daily_df["spot_price"] = daily_df["root_symbol"].map(spot_map)
        daily_df = daily_df.dropna(subset=["spot_price"])
        daily_df['strike_price'] = daily_df['strike_price']*10
        # Compute moneyness ratio
        daily_df["moneyness_ratio"] = daily_df["strike_price"] / daily_df["spot_price"]
        daily_df['strike_price'] = daily_df['strike_price'] / 10
        # Apply filters
        filtered = daily_df[
            (daily_df["dte"] <= 90) &
            (daily_df["moneyness_ratio"] >= 0.1) &
            (daily_df["moneyness_ratio"] <= 1.8)
        ]

        if not filtered.empty:
            store_out.append("/full_dataset", filtered, format='table', data_columns=["ticker", "date", "root_symbol"])

    store_in.close()
    store_out.close()
    print(f"\n✅ Done. Filtered data written to {hdf_output}")

if __name__ == "__main__":
    #build_time_series_fast("./polygon_data_2024", hdf_path="contract_timeseries_2024.h5")
    # build_combined_time_series(
    #    ["./polygon_data_2024", "./polygon_data_2025"],
    #    hdf_path="test_contract_timeseries.h5"
    # )
    fast_filter_contract_data(
        hdf_input="contract_timeseries.h5",
        hdf_output="filtered_contract_timeseries.h5",
        spot_path="./yf_data1.csv"
    )