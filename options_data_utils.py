import dotenv, os
#from uvatradier import Tradier, Account, Quotes, EquityOrder, OptionsData, OptionsOrder, Stream
from tradierapi import Tradier, Account, OptionsData,Quotes

import pandas as pd

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt

def chunked(iterable, size):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def get_options_symbols_list_parallel_chunks(options_data, symbols, chunk_size=100, max_workers=4, verbose=True):
    """
    Parallelize option symbol lookup using get_options_symbols_list in chunks.

    Args:
        options_data (OptionsData): Instance of OptionsData class.
        symbols (list): List of symbols to query.
        chunk_size (int): Number of symbols per request.
        max_workers (int): Number of parallel threads.
        verbose (bool): If True, show progress bar.

    Returns:
        pd.DataFrame: Combined DataFrame of option symbols.
    """

    chunks = list(chunked(symbols, chunk_size))
    results = []

    def fetch(symbol_chunk):
        try:
            return options_data.get_options_symbols_list(symbol_chunk)
        except Exception as e:
            if verbose:
                print(f"Chunk error: {e}")
            return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch, c): c for c in chunks}
        iterable = tqdm(as_completed(futures), total=len(futures), desc="Fetching option chunks") if verbose else as_completed(futures)
        for future in iterable:
            df = future.result()
            if not df.empty:
                results.append(df)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def save_symbols_to_hdf(
    quotes,
    symbols,
    filename='historical_data.h5',
    interval='daily',
    start_date=None,
    end_date=None,
    overwrite_file=False,
    overwrite_symbols=False,
    verbose=False
):
    """
    Fetch historical quotes and save them to an HDF5 file, safely appendable.

    Args:
        quotes (Quotes): An instance of your Quotes class.
        symbols (list): List of stock or option symbols.
        filename (str): HDF5 file to write to.
        interval (str): 'daily', 'weekly', or 'monthly'.
        start_date (str): Optional start date 'YYYY-MM-DD'.
        end_date (str): Optional end date 'YYYY-MM-DD'.
        overwrite_file (bool): If True, clear the entire HDF file before writing.
        overwrite_symbols (bool): If True, overwrite individual symbols if already saved.
        verbose (bool): If True, print extra info.
    """
    mode = 'w' if overwrite_file else 'a'

    with pd.HDFStore(filename, mode=mode, complevel=9, complib='zlib') as store:
        existing_keys = set([key.strip('/') for key in store.keys()])

        for symbol in tqdm(symbols, desc="Fetching & Saving Symbols", ncols=90):
            symbol_key = symbol.upper()

            if symbol_key in existing_keys and not overwrite_symbols:
                if verbose:
                    print(f"⏩ Skipping {symbol_key} (already exists)")
                continue

            try:
                df = quotes.get_historical_quotes(
                    symbol=symbol_key,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    verbose=False
                )
                if not df.empty:
                    store.put(symbol_key, df, format='table', data_columns=True)
                    if verbose:
                        print(f"✓ Saved {symbol_key} ({len(df)} rows)")
                else:
                    store.put(symbol_key, pd.DataFrame(), format='table', data_columns=True)
                    if verbose:
                        print(f"⚠ No data for {symbol_key}")

            except Exception as e:
                print(f"✗ Error saving {symbol_key}: {e}")

    print(f"\n✅ Done. HDF5 file updated: {filename}")


if __name__ == "__main__":
    import os
    import dotenv
    from tradierapi import Account, OptionsData

    dotenv.load_dotenv()
    tradier_account = os.getenv("tradier_account")
    tradier_token = os.getenv("tradier_token")

    account = Account(account_number=tradier_account, auth_token=tradier_token, live_trade=True)
    options_data = OptionsData(account, tradier_token, live_trade=True)

    stock_list = pd.read_csv("finviz_data.csv", index_col=0).index.tolist()
    #stock_list = stock_list[:100]  # Limit to first 100 for testing
    #df = get_options_symbols_list_parallel_chunks(options_data, stock_list,  max_workers=6)
    #df = options_data.get_options_symbols_list(stock_list)
    #df.to_csv("options_symbols.csv", index=False)

    # get options price history

    optionsymbols = pd.read_csv("options_symbols.csv")
    stock_data = pd.read_csv("yf_data1.csv", index_col=0).loc["2025-01-01":]
    stock_data_mean = pd.DataFrame(stock_data.mean(axis=0), columns=['price'])
    print(stock_data_mean)
    print(optionsymbols)
    optionsymbols = optionsymbols.set_index('root_symbol')
    optionsymbols = pd.merge(optionsymbols, stock_data_mean, left_index=True, right_index=True, how='left')
    optionsymbols = optionsymbols.reset_index()
    optionsymbols['price_strike'] = optionsymbols['strike_price'] / optionsymbols['price']
    today = pd.Timestamp.now()
    optionsymbols['dte'] = (pd.to_datetime(optionsymbols['expiration_date']) - today).dt.days
    #optionsymbols = optionsymbols[optionsymbols['dte'] < 40]
    #optionsymbols = optionsymbols[optionsymbols['dte'] > 0]
    target_dates = ['2025-07-18']#, '2025-08-15', '2025-09-19']

    optionsymbols = optionsymbols[optionsymbols['expiration_date'].isin(target_dates)]
    

    optionsymbols = optionsymbols[optionsymbols['price_strike'] < 1.2]
    optionsymbols = optionsymbols[optionsymbols['price_strike'] > 0.8]
    optionsymbols = optionsymbols[optionsymbols['option_type'] == 'C']
    print(optionsymbols)

    import random

    symbols_to_retrieve = optionsymbols['symbol'].unique().tolist()
    quotes = Quotes(tradier_account, tradier_token, live_trade=True)
    
    symbol_test = random.choice(symbols_to_retrieve)
    symbol_test = "BKNG250711C05780000"

    print(f"Fetching historical quotes")
    print(len(symbols_to_retrieve), "symbols to retrieve")
    #df = quotes.get_historical_quotes(symbol_test, interval='daily', start_date='2024-01-01', end_date='2025-01-01')
    save_symbols_to_hdf(
        quotes,
        symbols_to_retrieve,
        filename='historical_data.h5',
        interval='daily',
        start_date='2024-01-01',
        end_date='2025-06-27',
        overwrite_file=False,
        overwrite_symbols=False,
        verbose=False
    )
    #df['close'].plot(title=f"Historical Prices for {symbol_test}")
    #plt.show()