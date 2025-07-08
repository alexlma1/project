import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import os
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Iterable, Tuple
from tqdm.contrib.concurrent import process_map 

def load_spot_prices(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    return df.set_index("Date")

def bs_price(S, K, T, r, sigma, call=True):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol(price, S, K, T, r=0.01, call=True):
    try:
        return brentq(lambda sigma: bs_price(S, K, T, r, sigma, call) - price, 1e-5, 5)
    except:
        return np.nan

def get_atm_options(store, date, symbol, spot_price, dte_window=(20, 70), n_strikes=3, moneyness_range=(0.5, 1.2)):
    date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
    root_symbol = f"O:{symbol}"
    try:
        options = store.select(
            "/full_dataset",
            where=f'root_symbol == "{root_symbol}" and date == "{date_str}"'
        )
    except Exception:
        return pd.DataFrame()
    if options.empty:
        return options
    options["dte"] = (options["expiration_date"] - date).dt.days
    
    filtered = options[(options["dte"] >= dte_window[0]) & (options["dte"] <= dte_window[1])].copy()
    filtered["strike_price"] *= 10
    
    filtered["moneyness_ratio"] = filtered["strike_price"] / spot_price
    filtered = filtered[(filtered["moneyness_ratio"] >= moneyness_range[0]) & (filtered["moneyness_ratio"] <= moneyness_range[1])]
    if filtered.empty:
        return filtered

    filtered['moneyness_closeness'] = np.abs(filtered['moneyness_ratio'] - 1)
    atm = filtered.nsmallest(n_strikes, 'moneyness_closeness')
    return atm

def compute_daily_iv(atm_df, spot_price, date):
    T = atm_df["dte"] / 365
    K = atm_df["strike_price"]
    option_price = atm_df["close"]
    is_call = atm_df["option_type"] == "C"

    ivs = []
    for i in range(len(atm_df)):
        iv = implied_vol(option_price.iloc[i], spot_price, K.iloc[i], T.iloc[i], r=0.05, call=is_call.iloc[i])
        if not np.isnan(iv):
            ivs.append(iv)
    return np.mean(ivs) if ivs else np.nan

def calculate_all_iv(spot_df, hdf_path="contract_timeseries_combined.h5"):
    store = pd.HDFStore(hdf_path)
    results = []

    for symbol in tqdm(spot_df.columns, desc="\U0001F50D Processing symbols"):
        for date, spot in tqdm(spot_df[symbol].dropna().items(), leave=False, desc=f"\U0001F4C6 {symbol}"):
            atm_options = get_atm_options(store, date, symbol, spot)
            #print(atm_options)
            
            if atm_options.empty:
                continue
            avg_iv = compute_daily_iv(atm_options, spot, date)
            if not np.isnan(avg_iv):
                results.append({"date": date, "symbol": symbol, "atm_iv": avg_iv})

    store.close()

    if not results:
        print("\u26A0\uFE0F No IV results calculated. Exiting.")
        return pd.DataFrame(), pd.DataFrame()

    iv_df = pd.DataFrame(results)#.sort_values(["symbol", "date"])
    iv_wide = iv_df.pivot(index="date", columns="symbol", values="atm_iv").sort_index()
    print(iv_wide)
    return iv_wide


def update_iv_for_date(date, hdf_path="~/data/filtered_contract_timeseries.h5", 
                       spot_path="~/data/yf_data1.csv", iv_output_path="~/data/iv_df.csv"):
    """
    Update the IV dataframe with one day's IV values.
    """
    date = pd.to_datetime(date)
    hdf_path = os.path.expanduser(hdf_path)
    spot_path = os.path.expanduser(spot_path)
    iv_output_path = os.path.expanduser(iv_output_path)

    # Load spot prices
    spot_df = load_spot_prices(spot_path)
    if date not in spot_df.index:
        print(f"⚠️ Spot data not available for {date.date()}")
        return

    spot_row = spot_df.loc[date].dropna()
    if spot_row.empty:
        print(f"⚠️ All spot prices are NaN on {date.date()}")
        return

    # Open HDF5 and get IVs
    results = []
    with pd.HDFStore(hdf_path) as store:
        for symbol, spot in spot_row.items():
            atm_df = get_atm_options(store, date, symbol, spot, n_strikes=n_strikes)
            if atm_df.empty:
                continue
            avg_iv = compute_daily_iv(atm_df, spot, date)
            if not np.isnan(avg_iv):
                results.append({"date": date, "symbol": symbol, "atm_iv": avg_iv})

    if not results:
        print(f"⚠️ No IVs computed for {date.date()}")
        return

    new_df = pd.DataFrame(results)
    new_wide = new_df.pivot(index="date", columns="symbol", values="atm_iv")

    # Append to existing CSV
    if Path(iv_output_path).exists():
        old_iv = pd.read_csv(iv_output_path, index_col=0, parse_dates=True)
        combined = pd.concat([old_iv, new_wide])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    else:
        combined = new_wide

    combined.to_csv(iv_output_path)
    print(f"✅ IV updated for {date.date()} in {iv_output_path}")



# def load_filtered_data_by_date(store, key="/full_dataset"):
#     """Cache filtered HDF data by date to speed up downstream access."""
#     dates = pd.to_datetime(store.select_column(key, "date")).dropna().unique()
#     cache = {}
#     for date in tqdm(dates, desc="🗃️ Caching HDF5 by date"):
#         try:
#             df = store.select(key, where=f'date == "{date.strftime("%Y-%m-%d")}"')
#             if not df.empty:
#                 cache[date] = df
#         except:
#             continue
#     return cache

# def get_atm_options_from_cache(cache, date, symbol, spot_price, dte_window=(25, 70), n_strikes=3, moneyness_range=(0.5, 1.2)):
#     root_symbol = f"O:{symbol}"
#     if date not in cache:
#         return pd.DataFrame()

#     options = cache[date]
#     options = options[options["root_symbol"] == root_symbol].copy()
#     if options.empty:
#         return options

#     options["dte"] = (options["expiration_date"] - date).dt.days
#     options = options[(options["dte"] >= dte_window[0]) & (options["dte"] <= dte_window[1])]
#     options["moneyness_ratio"] = options["strike_price"] / spot_price
#     options = options[(options["moneyness_ratio"] >= moneyness_range[0]) & (options["moneyness_ratio"] <= moneyness_range[1])]

#     if options.empty:
#         return options

#     options["moneyness_closeness"] = np.abs(options["moneyness_ratio"] - 1.0)
#     return options.sort_values("moneyness_closeness").groupby("option_type").head(n_strikes)

# def compute_daily_iv(atm_df, spot_price, date):
#     T = atm_df["dte"] / 365
#     K = atm_df["strike_price"]
#     option_price = atm_df["close"]
#     is_call = atm_df["option_type"] == "C"

#     ivs = [
#         implied_vol(option_price.iloc[i], spot_price, K.iloc[i], T.iloc[i], r=0.05, call=is_call.iloc[i])
#         for i in range(len(atm_df))
#     ]
#     ivs = [iv for iv in ivs if not np.isnan(iv)]
#     return np.mean(ivs) if ivs else np.nan

# def calculate_all_iv(spot_df, hdf_path="contract_timeseries.h5"):
#     store = pd.HDFStore(hdf_path)
#     cache = load_filtered_data_by_date(store)
#     store.close()

#     results = []
#     for symbol in tqdm(spot_df.columns, desc="🔍 Processing symbols"):
#         for date, spot in tqdm(spot_df[symbol].dropna().items(), leave=False, desc=f"📆 {symbol}"):
#             atm_df = get_atm_options_from_cache(cache, date, symbol, spot)
#             if atm_df.empty:
#                 continue
#             avg_iv = compute_daily_iv(atm_df, spot, date)
#             if not np.isnan(avg_iv):
#                 results.append({"date": date, "symbol": symbol, "atm_iv": avg_iv})

#     if not results:
#         print("⚠️ No IV results calculated.")
#         return pd.DataFrame(), pd.DataFrame()

#     iv_df = pd.DataFrame(results)
#     iv_wide = iv_df.pivot(index="date", columns="symbol", values="atm_iv").sort_index()
#     return iv_df, iv_wide

def list_available_dates(hdf_path):
    try:
        with pd.HDFStore(hdf_path, mode='r') as store:
            df = store.select_column("/filtered_dataset", "date")
            return pd.to_datetime(df).dropna().sort_values().unique()
    except Exception as e:
        return str(e)

def read_if_df(path):
    """
    Read the implied volatility DataFrame from a CSV file.
    If the file does not exist, return an empty DataFrame.
    """
    path = os.path.expanduser(path)
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        print(f"⚠️ No IV data found at {path}. Returning empty DataFrame.")
        return pd.DataFrame()


def build_volume_timeseries(
    hdf_path: str | Path,
    spot_df: pd.DataFrame,
    spot_df_cols: Iterable[str] | None = None,
    *,
    vol_df_path: str | Path | None = None,
    n_strikes: int = 200,
    moneyness_range: Tuple[float, float] = (0.2, 2.0),
    plot: bool = True,
    show_progress: bool = True,
):
    """
    Summarise option volume into a time-series DataFrame.

    Parameters
    ----------
    hdf_path
        Location of the HDF5 file queried by ``get_atm_options``.
    spot_df
        Index = dates, columns = spot prices.  These dates/symbols drive the loops.
    spot_df_cols
        Optional subset of columns to process. Default: all columns in *spot_df*.
    vol_df_path
        If given, write the output DataFrame to this CSV path.
    n_strikes, moneyness_range
        Passed straight through to ``get_atm_options``.
    plot
        If *True*, draw a quick line-plot of the finished DataFrame.
    show_progress
        Toggle the nested ``tqdm`` progress bars.

    Returns
    -------
    pd.DataFrame
        Option-volume time-series with the same shape as *spot_df*.
    """
    hdf_path = Path(hdf_path).expanduser()
    if vol_df_path is not None:
        vol_df_path = Path(vol_df_path).expanduser()

    # Column list to iterate
    symbols = (list(spot_df_cols)
               if spot_df_cols is not None
               else list(spot_df.columns))

    # Pre-allocate output
    vol_df = pd.DataFrame(index=spot_df.index, columns=symbols, dtype="float64")

    # tqdm settings: always show elapsed + ETA
    tkwargs = dict(
        dynamic_ncols=True,
        bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[elapsed {elapsed} · ETA {remaining}, {rate_fmt}]")
    )

    with pd.HDFStore(hdf_path, mode="r") as store:

        outer_iter = tqdm(symbols, desc="Symbols", **tkwargs) if show_progress else symbols

        for symbol in outer_iter:
            # Skip if the symbol column is missing entirely
            if symbol not in spot_df.columns:
                continue

            date_spot_iter = spot_df[symbol].dropna().items()
            inner_iter = tqdm(date_spot_iter,
                              desc=f"📆 {symbol}",
                              leave=False,
                              position=1,
                              **tkwargs) if show_progress else date_spot_iter

            for date, spot in inner_iter:
                atm_opts = get_atm_options(
                    store, date, symbol, spot,
                    n_strikes=n_strikes,
                    moneyness_range=moneyness_range,
                )
                vol_sum = atm_opts["volume"].sum()
                vol_df.at[date, symbol] = vol_sum if vol_sum else None

            if show_progress:
                inner_iter.close()        # clear the inner bar (leave=False)

    # ── Post-processing ────────────────────────────────────────────────────────
    if vol_df_path is not None:
        vol_df.to_csv(vol_df_path)

    if plot:
        ax = vol_df.plot(figsize=(12, 5), title="Option Volume Time-Series")
        ax.set_ylabel("Volume")
        plt.tight_layout()
        plt.show()

    return vol_df

def update_volume_for_date(
    date: str,
    hdf_path: str | Path = "~/data/contract_timeseries.h5",
    spot_path: str | Path = "~/data/yf_data1.csv",
    vol_output_path: str | Path = "~/data/vol_timeseries.csv",
    n_strikes: int = 200,
    moneyness_range: Tuple[float, float] = (0.2, 2.0),
    plot: bool = True,
    show_progress: bool = True,
    ) -> pd.DataFrame:
    """
    Update the option volume DataFrame with one day's volume values.
    Parameters
    ----------
    """
    date = pd.to_datetime(date)
    hdf_path = Path(hdf_path).expanduser()
    spot_path = Path(spot_path).expanduser()
    vol_output_path = Path(vol_output_path).expanduser()
    with pd.HDFStore(hdf_path, mode="r") as store:
        spot_df = load_spot_prices(spot_path)
        if date not in spot_df.index:
            print(f" Spot data not available for {date.date()}")
            return pd.DataFrame()

        spot_row = spot_df.loc[date].dropna()
        if spot_row.empty:
            print(f" All spot prices are NaN on {date.date()}")
            return pd.DataFrame()

        results = []
        for symbol, spot in spot_row.items():
            atm_df = get_atm_options(store, date, symbol, spot,
                                     n_strikes=n_strikes,
                                     moneyness_range=moneyness_range)
            if atm_df.empty:
                continue
            vol_sum = atm_df["volume"].sum()
            results.append({symbol: vol_sum if vol_sum else np.nan})

    old_results = pd.read_csv(vol_output_path, index_col=0, parse_dates=True) if vol_output_path.exists() else pd.DataFrame()
    # add new date to old results
    row = {k: v for d in results for k, v in d.items()}
    new_df = pd.DataFrame([row], index=[date])
    old_results = old_results.reindex(
        index   = old_results.index.union(new_df.index),
        columns = old_results.columns.union(new_df.columns)
    )
    old_results.update(new_df)
    old_results.to_csv(vol_output_path)
    print(old_results)
    
# ── worker --------------------------------------------------------------------
def _symbol_volume_job(args: Tuple[str, str, pd.Series, int, Tuple[float, float]]  # noqa: E501
                       ) -> Tuple[str, Dict[pd.Timestamp, float | None]]:
    """Run inside its own process; *always* re-open the HDFStore here."""
    symbol, hdf_path, spot_series, n_strikes, m_range = args
    from pandas import HDFStore                                    # lazy import

    out: Dict[pd.Timestamp, float | None] = {}
    with HDFStore(hdf_path, mode="r") as store:
        for date, spot in spot_series.items():                     # already dropna'ed
            atm = get_atm_options(store, date, symbol, spot,
                                  n_strikes=n_strikes,
                                  moneyness_range=m_range)
            vol = atm["volume"].sum()
            out[date] = vol if vol else np.nan
    return symbol, out


# ── public helper -------------------------------------------------------------
def build_volume_timeseries_parallel(
    hdf_path: str | Path,
    spot_df: pd.DataFrame,
    *,
    spot_df_cols: Iterable[str] | None = None,
    vol_df_path: str | Path | None = None,
    n_strikes: int = 200,
    moneyness_range: Tuple[float, float] = (0.2, 2),
    workers: int | None = None,        # default = os.cpu_count()
    chunksize: int = 1,                # (tqdm will auto-chunk if set to None)
    plot: bool = True,
) -> pd.DataFrame:
    """
    Same job as the serial version, but each symbol is processed in its own process.
    Progress bars show elapsed / ETA automatically.
    """
    hdf_path = Path(hdf_path).expanduser()
    if vol_df_path is not None:
        vol_df_path = Path(vol_df_path).expanduser()

    symbols = (list(spot_df_cols)
               if spot_df_cols is not None
               else list(spot_df.columns))

    # Build the argument tuples once so they are pickle-friendly
    jobs = [
        (sym, str(hdf_path), spot_df[sym].dropna(), n_strikes, moneyness_range)
        for sym in symbols
        if sym in spot_df.columns
    ]

    # --- run in parallel with nice tqdm bars ---------------------------------
    # Each worker returns (symbol, {date: volume})
    results = process_map(
        _symbol_volume_job,
        jobs,
        max_workers=workers,
        chunksize=chunksize,
        desc="Symbols",
        unit="sym",                    # shows “ 12/50 [elapsed …]” etc.
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                   "[elapsed {elapsed} · ETA {remaining}, {rate_fmt}]"
    )

    # --- assemble the wide DataFrame -----------------------------------------
    vol_df = pd.DataFrame(index=spot_df.index,
                          columns=[sym for sym, _ in results],
                          dtype="float64")

    for sym, mapping in results:
        vol_df.loc[mapping.keys(), sym] = list(mapping.values())
    print(vol_df)
    if vol_df_path is not None:
        vol_df.to_csv(vol_df_path)

    if plot:
        ax = vol_df.plot(figsize=(12, 5), title="Option-volume time-series")
        ax.set_ylabel("Volume")
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.show()

    return vol_df

if __name__ == "__main__":
    spot_df = load_spot_prices("~/data/yf_data1.csv").loc["2024-01-03":]
    spot_df_cols = spot_df.columns
    # #spot_df = spot_df[["AA"]]
    # #ivdf = pd.read_csv("iv_df.csv")
    # #ivdf['date'] = pd.to_datetime(spot_df.index)
    # #ivdf = ivdf.set_index("date")
    # #print(ivdf)
    # #ivdf.to_csv("iv_df.csv")
    # # update_iv_for_date("2025-07-01", hdf_path="~/data/filtered_contract_timeseries.h5",
    # #                    spot_path="~/data/yf_data1.csv", iv_output_path="~/data/iv_df.csv")
    
    # df=read_if_df("~/data/iv_df.csv")  # Read existing IV data if available
    # print(df["MU"])
    # exit()
    # iv_df = calculate_all_iv(spot_df, hdf_path="~/data/filtered_contract_timeseries.h5")
    # print(iv_df)
    # iv_df.to_csv("~/data/iv_df.csv")
    
    vol_df_path = '~/data/vol_timeseries.csv'
    hdf_path = '~/data/contract_timeseries.h5'
    # store = pd.HDFStore(hdf_path)
    # results = []
    # vol_df = pd.DataFrame(index=spot_df.index, columns=spot_df_cols)
    # for symbol in spot_df_cols:
    #     print(symbol)

    #     for date, spot in tqdm(spot_df[symbol].dropna().items(), leave=False, desc=f"{symbol}"):
    #         atm_options = get_atm_options(store, date, symbol, spot,n_strikes=200, moneyness_range=(0.2,2))
    #         volume = atm_options['volume'].sum()
    #         if volume:
    #             vol_df.loc[date,symbol] = volume
    #         else:
    #             vol_df.loc[date,symbol] = None
    
    # vol_df.to_csv(vol_df_path)
    # vol_df.plot()
    # plt.show()
    # volumes = build_volume_timeseries_parallel(
    #     hdf_path="~/data/contract_timeseries.h5",
    #     spot_df=spot_df,
    #     spot_df_cols=spot_df_cols,          # or omit for all columns
    #     vol_df_path="~/data/vol_timeseries.csv",
    #     n_strikes=200,
    #     moneyness_range=(0.2, 2),
    #     workers=10,                          # or None = use all CPUs
    #     chunksize=1,                        # tune if symbols are many & tiny
    #     plot=True,
    # )
    # update_volume_for_date(
    #     "2025-07-03",
    #     hdf_path="~/data/filtered_contract_timeseries.h5",
    #     spot_path="~/data/yf_data1.csv",
    #     vol_output_path="~/data/vol_timeseries.csv"
    # )
