import re
from matplotlib import pyplot as plt
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from typing import Tuple

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

def append_latest_contract_file( data_dir="~/data/polygon_data_2025", hdf_path="~/data/contract_timeseries.h5",
                                file_name_provided=None):
    data_dir = Path(os.path.expanduser(data_dir))
    hdf_path = os.path.expanduser(hdf_path)
    
    # Find the latest .csv.gz file based on date in filename
    files = sorted(data_dir.glob("2025-*.csv.gz"))
    if not files:
        raise FileNotFoundError("No CSV files found in the specified data directory.")

    latest_file = files[-1]
    latest_date = pd.to_datetime(latest_file.stem[:10])
    print(f"📄 Appending latest file: {latest_file.name}")
    
    
    if file_name_provided: 
        latest_file = Path(os.path.expanduser(data_dir)) / file_name_provided
        if not latest_file.exists():
            raise FileNotFoundError(f"Provided file {latest_file} does not exist.")
        latest_date = pd.to_datetime(latest_file.stem[:10])
        print(f"📄 Using provided file: {latest_file.name} with date {latest_date.date()}")

        print(f"📅 Date: {latest_date.date()}")
        
    try:
        df = pd.read_csv(latest_file)
    except Exception as e:
        print(f"❌ Error reading {latest_file.name}: {e}")
        return

    if df.empty or 'ticker' not in df.columns:
        print("⚠️ Skipping empty or invalid file.")
        return

    df['date'] = latest_date

    try:
        df = fast_parse(df)
    except Exception as e:
        print(f"❌ Parse error: {e}")
        return

    try:
        with pd.HDFStore(hdf_path, mode='a') as store:
            store.append("/full_dataset", df, format='table', data_columns=["ticker", "date", "root_symbol"])
        print(f"✅ Appended {len(df)} rows for {latest_date.date()} to {hdf_path}")
    except Exception as e:
        print(f"❌ Failed to write to HDF5: {e}")



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

def drop_columns_contract_data(tickers, hdf_input, hdf_output):
    store_in = pd.HDFStore(hdf_input, mode='r')
    store_out = pd.HDFStore(hdf_output, mode='w')

    # Read only dates with unique values once
    available_dates = pd.to_datetime(store_in.select_column("/full_dataset", "date")).dropna().unique()

    for date in tqdm(available_dates, desc="📅 Filtering by date"):
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        try:
            daily_df = store_in.select("/full_dataset", where=f'date == "{date_str}"')
        except Exception:
            continue

        if daily_df.empty:
            continue

        root_symbols = [f"O:{s}" for s in tickers]
        daily_df = daily_df[daily_df["root_symbol"].isin(root_symbols)].copy()
        
        if daily_df.empty:
            continue
        daily_df = daily_df[['ticker','close','volume','date']]
        filtered = daily_df.copy()
        if not filtered.empty:
            store_out.append("/full_dataset", filtered, format='table', data_columns=["ticker", "date", "root_symbol"])

    store_in.close()
    store_out.close()
    print(f"\n✅ Done. Filtered data written to {hdf_output}")

def update_filtered_contract_for_date(date, hdf_input="~/data/contract_timeseries.h5",
                                      hdf_output="~/data/filtered_contract_timeseries.h5",
                                      spot_path="~/data/yf_data1.csv"):
    """
    Filter and append one day's options data to the filtered HDF5 file.
    """
    date = pd.to_datetime(date)
    date_str = date.strftime("%Y-%m-%d")
    hdf_input = os.path.expanduser(hdf_input)
    hdf_output = os.path.expanduser(hdf_output)
    spot_path = os.path.expanduser(spot_path)

    # Load spot data
    spot_df = pd.read_csv(spot_path, parse_dates=['Date']).set_index('Date')
    
    if date not in spot_df.index:
        print(f"⚠️ Spot data not available for {date_str}")
        return

    spot_row = spot_df.loc[date].dropna()
    if spot_row.empty:
        print(f"⚠️ Spot prices for {date_str} are all NaN")
        return

    try:
        with pd.HDFStore(hdf_input, mode='r') as store_in:
            daily_df = store_in.select("/full_dataset", where=f'date == "{date_str}"')
    except Exception as e:
        print(f"❌ Failed to read {date_str} from {hdf_input}: {e}")
        return

    if daily_df.empty:
        print(f"⚠️ No options data for {date_str}")
        return

    symbols = spot_row.index.tolist()
    root_symbols = [f"O:{s}" for s in symbols]
    daily_df = daily_df[daily_df["root_symbol"].isin(root_symbols)].copy()

    if daily_df.empty:
        print(f"⚠️ No matching root symbols on {date_str}")
        return

    daily_df["dte"] = (daily_df["expiration_date"] - date).dt.days
    spot_map = {f"O:{sym}": spot_row[sym] for sym in symbols}
    daily_df["spot_price"] = daily_df["root_symbol"].map(spot_map)
    daily_df = daily_df.dropna(subset=["spot_price"])
    
    daily_df["moneyness_ratio"] = daily_df["strike_price"] / daily_df["spot_price"]
    daily_df['strike_price'] /= 10

    filtered = daily_df[
        (daily_df["dte"] <= 90) &
        (daily_df["moneyness_ratio"] >= 0.1) &
        (daily_df["moneyness_ratio"] <= 1.8)
    ]

    if filtered.empty:
        print(f"⚠️ No options passed filtering on {date_str}")
        return

    try:
        with pd.HDFStore(hdf_output, mode='a') as store_out:
            store_out.append("/full_dataset", filtered, format='table',
                             data_columns=["ticker", "date", "root_symbol"])
        print(f"✅ Appended {len(filtered)} rows for {date_str} to {hdf_output}")
    except Exception as e:
        print(f"❌ Failed to write filtered data: {e}")
def read_data(hdf_path="~/data/filtered_contract_timeseries_500.h5"):
    """
    Read and return the filtered contract data from HDF5.
    """
    new_hdf_path = Path("~/data/temp.h5").expanduser()
    new_store = pd.HDFStore(new_hdf_path, mode='r')
    df = new_store.select("/full_dataset")
    return df
    # return df
    # hdf_path = os.path.expanduser(hdf_path)
    # new_store = pd.HDFStore(new_hdf_path, mode='w')

    # try:
    #     with pd.HDFStore(hdf_path, mode='r') as store:
    #         df = store.select("/full_dataset")
    #         df = parse_ticker(df)
    #         df = df[df['root'].isin(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])]
    #         # write to new store
    #         new_store.put("/full_dataset", df, format='table',data_columns=df.columns.tolist())
    #     new_store.close()
    #     return df
    # except Exception as e:
    #     print(f"❌ Failed to read data from {hdf_path}: {e}")
    #     return pd.DataFrame()


# ───────────────────────────────────────────────────────────────
# 2.  Black-Scholes helpers
# ───────────────────────────────────────────────────────────────
def _bs_price(cp: str, S: float, K: float, T: float, r: float,
              sigma: float, q: float = 0.0) -> float:
    """European Black-Scholes price."""
    if T <= 0:
        return max(0.0, (S - K) if cp == "C" else (K - S))

    disc = np.exp(-r * T)
    dq   = np.exp(-q * T)
    sig_sqrt = sigma * np.sqrt(T)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sig_sqrt
    d2 = d1 - sig_sqrt

    if cp == "C":
        return dq * S * norm.cdf(d1) - disc * K * norm.cdf(d2)
    else:  # Put
        return disc * K * norm.cdf(-d2) - dq * S * norm.cdf(-d1)


def implied_vol(cp: str, S: float, K: float, T: float, r: float,
                price: float, q: float = 0.0,
                bounds: Tuple[float, float] = (1e-4, 5.0)) -> float:
    """
    Solve for sigma such that Black-Scholes price == market `price`.
    Returns `np.nan` if no solution in `bounds`.
    """
    if price <= 0 or T <= 0:
        return np.nan
    try:
        return brentq(
            lambda vol: _bs_price(cp, S, K, T, r, vol, q) - price,
            bounds[0], bounds[1], maxiter=200
        )
    except ValueError:
        return np.nan


# ───────────────────────────────────────────────────────────────
# 3.  Core pipeline pieces
# ───────────────────────────────────────────────────────────────
def collapse_duplicates(opt: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (ticker, date).
    Sums `volume`, keeps last `close`.
    """
    return (opt.groupby(["ticker", "date"], as_index=False)
                .agg(close=("close", "last"),
                     volume=("volume", "sum")))


def filter_long_lived(opt: pd.DataFrame,
                      min_obs: int = 90,
                      min_span: int = 60) -> pd.DataFrame:
    """
    Keep contracts with ≥ `min_obs` observations AND
    covering ≥ `min_span` calendar days.
    """
    life = (opt.groupby("ticker")["date"]
              .agg(obs="size",
                   span=lambda s: (s.max() - s.min()).days + 1))

    keep = life.query("obs >= @min_obs and span >= @min_span").index
    return opt[opt["ticker"].isin(keep)].copy()


def flag_high_volume(opt: pd.DataFrame, window: int = 20,
                     z_thresh: float = 2.0) -> pd.Series:
    """
    Return boolean Series where volume > universe mean + z_thresh * std
    over a rolling window of length `window`.
    """
    roll_mean = opt["volume"].rolling(window, min_periods=1).mean()
    roll_std  = opt["volume"].rolling(window, min_periods=1).std(ddof=0)
    return opt["volume"] > roll_mean + z_thresh * roll_std


def attach_underlying(opt: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    # 1 ▸ be sure the index & columns are named
    px = (px.rename_axis(index="date")        # row index
             .rename_axis(columns="root"))    # column index ← key trick

    # 2 ▸ long-to-wide → tidy 3-col frame
    stacked = (
        px.stack()                # MultiIndex (date, root)
          .rename("S")            # price column
          .reset_index()          # → DataFrame ['date','root','S']
    )

    # 3 ▸ harmonise option table’s date col name
    opt = opt.rename(columns={"Date": "date"})   # harmless if already lower-case

    # 4 ▸ merge
    return opt.merge(stacked, on=["date", "root"], how="left")

def add_bs_implied_vol(opt: pd.DataFrame, r: float = 0.0005,
                       q: float = 0.0) -> pd.DataFrame:
    """
    Compute Black-Scholes implied vol for each row and return dataframe
    with an extra 'iv' column (σ in annual decimal terms).
    Requires columns:
        • cp       — 'C' or 'P'
        • S        — underlying close
        • strike   — strike price
        • T        — time to expiry (years)
        • close    — option market price
    """
    opt["iv"] = opt.apply(
        lambda row: implied_vol(row["cp"], row["S"], row["strike"],
                                row["T"], r, row["close"], q),
        axis=1
    )
    return opt


# ───────────────────────────────────────────────────────────────
# 4.  Convenience “run-it-all” wrapper
# ───────────────────────────────────────────────────────────────
def screen_high_vol(opt_raw: pd.DataFrame,
                    px: pd.DataFrame,
                    *,
                    min_obs: int = 40,
                    min_span: int = 60,
                    window: int = 20,
                    z_thresh: float = 2.0,
                    r: float = 0.0005,
                    q: float = 0.0) -> pd.DataFrame:
    """
    Run the full pipeline:

        1. deduplicate rows ➜ collapse_duplicates
        2. parse OCC label   ➜ parse_occ
        3. long-lived filter ➜ filter_long_lived
        4. attach S, T       ➜ attach_underlying & time-to-mat
        5. high-volume flag  ➜ flag_high_volume
        6. implied vol       ➜ add_bs_implied_vol

    Returns DataFrame of rows that meet the high-volume criterion.
    """

    # 1. remove duplicates
    opt = opt_raw.drop_duplicates(subset=["ticker", "date"], keep="last")


    print(f"Parsed {len(opt)} option contracts.")
    # 3. long-lived filter
    #opt = filter_long_lived(opt, min_obs=min_obs, min_span=min_span)
    print(f"Filtered to {len(opt)} long-lived contracts.")
    # 4. attach underlying close & time-to-mat
    opt = attach_underlying(opt, px)
    opt["T"] = ((opt["expiry"] - opt["date"]).dt.days.clip(lower=0) / 365.25)
    opt = opt.query("T > 0 & S.notnull()", engine="python")
    print(f"Attached underlying prices; {len(opt)} contracts remain.")
    print(opt)

    # 5. high-volume flag
    hv_mask = flag_high_volume(opt, window=window, z_thresh=z_thresh)
    #opt_hv = opt[hv_mask].copy()
    opt_hv = opt
    # 6. implied volatility
    opt_hv = add_bs_implied_vol(opt_hv, r=r, q=q)
    print(opt_hv['ticker'].drop_duplicates())
    print(opt_hv)

    print(opt_hv[opt_hv['ticker']=='O:AAPL250718C00190000'].set_index('date')['iv'])
    opt_hv[opt_hv['ticker']=='O:AAPL250718C00190000'].set_index('date')['iv'].plot()
    plt.show()
    
    return opt_hv
def parse_ticker(opt):
    opt['root'] = opt['ticker'].str.slice(2, -15)
    opt['expiry'] = pd.to_datetime(opt['ticker'].str.slice(-15, -9), format='%y%m%d', errors='coerce')
    opt['cp'] = opt['ticker'].str.slice(-9, -8)
    opt['strike'] = opt['ticker'].str.slice(-8, step=1).astype(float) / 1000.0
    return opt
if __name__ == "__main__":
    # build_combined_time_series(
    #    ["~/data/polygon_data_2024", "~/data/polygon_data_2025"],
    #    hdf_path="~/data/contract_timeseries.h5"
    # )
    # fast_filter_contract_data(
    #     hdf_input="contract_timeseries.h5",
    #     hdf_output="filtered_contract_timeseries.h5",
    #     spot_path="./yf_data1.csv"
    # )


    # append_latest_contract_file(
    #     data_dir="~/data/polygon_data_2025",
    #     hdf_path="~/data/contract_timeseries.h5",
    #     file_name_provided="2025-07-03.csv.gz"
    # )

    # update_filtered_contract_for_date(
    #     date="2025-07-03",
    #     hdf_input="~/data/contract_timeseries.h5",
    #     hdf_output="~/data/filtered_contract_timeseries.h5",
    #     spot_path="~/data/yf_data1.csv"
    # )

    # voldf = pd.read_csv("~/data/vol_timeseries.csv", index_col=0, parse_dates=True)
    # voldf = voldf.iloc[-40:]
    # tickers = voldf.sum(axis=0).sort_values().iloc[-500:]
    # tickers = tickers.index.tolist()
    # drop_columns_contract_data(
    #     tickers=tickers,
    #     hdf_input="~/data/filtered_contract_timeseries.h5",
    #     hdf_output="~/data/filtered_contract_timeseries_500.h5"
    #     )
    
    # choose your span
    start = "2025-01-01"
    end   = "2025-12-31"

    third_fridays = pd.date_range(
        start=start,
        end=end,
        freq="WOM-3FRI",      # Week-Of-Month, 3rd Friday
    )
    print(third_fridays)
    df = read_data()
    px = pd.read_csv("~/data/yf_data1.csv", index_col="Date", parse_dates=True)
    symbol = "AAPL"  
    df = df[df["ticker"].str.startswith(f"O:{symbol}")]
    #df = parse_ticker(df)
    df = df[df["expiry"].isin(third_fridays)]

    
    iv_screen = screen_high_vol(
        df, px,
        min_obs=30,        # loosen longevity filter if you wish
        window=15,         # 15-day universe window
        z_thresh=2.5,      # stricter high-volume definition
    )