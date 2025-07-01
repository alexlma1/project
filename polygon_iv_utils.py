import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from tqdm import tqdm

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

def get_atm_options(store, date, symbol, spot_price, dte_window=(25, 70), n_strikes=3, moneyness_range=(0.5, 1.2)):
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

if __name__ == "__main__":
    spot_df = load_spot_prices("yf_data1.csv").loc["2024-01-03":]
    spot_df_cols = spot_df.columns
    spot_df = spot_df[["AA"]]
    #ivdf = pd.read_csv("iv_df.csv")
    #ivdf['date'] = pd.to_datetime(spot_df.index)
    #ivdf = ivdf.set_index("date")
    #print(ivdf)
    #ivdf.to_csv("iv_df.csv")
    exit()
    iv_df = calculate_all_iv(spot_df, hdf_path="filtered_contract_timeseries.h5")
    print(iv_df)
    iv_df.to_csv("iv_df.csv")
    