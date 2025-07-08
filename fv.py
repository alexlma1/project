import pandas as pd
import requests, time, os
from bs4 import BeautifulSoup
from tqdm import tqdm  # Import tqdm
import yfinance as yf
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

def download_fv_data(hdf_file="finviz_cache.h5", reset_cache=False):
    sections = ["111", "141", "161", "121", "131"]
    rows = [str(i) for i in range(0, 2000, 20)]
    url = "https://finviz.com/screener.ashx?v="

    if reset_cache and os.path.exists(hdf_file):
        os.remove(hdf_file)

    df_sections = []

    for section in sections:
        key = f"/section_{section}"

        # Try to load cached section from disk
        if os.path.exists(hdf_file):
            with pd.HDFStore(hdf_file, mode="r") as store:
                if key in store and not reset_cache:
                    print(f"Loading cached section {section}...")
                    df = store[key]
                    df_sections.append(df)
                    continue

        print(f"Downloading section {section}...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
        }
        session = requests.Session()
        data = []

        for i in tqdm(rows, desc=f"Section {section}"):
            page_url = f"{url}{section}&r={i}&f=cap_midover,sh_opt_option"
            try:
                r = session.get(page_url, headers=headers, timeout=10)
                soup = BeautifulSoup(r.content, "html.parser")
                table = soup.find("table", class_="styled-table-new is-rounded is-tabular-nums w-full screener_table")
                if not table:
                    print(f"No table found on page {page_url}. Skipping...")
                    continue

                table_rows = table.find_all("tr")
                columns = [col.get_text(strip=True) for col in table_rows[0].find_all("th")]

                for row in table_rows[1:]:
                    cols = row.find_all("td")
                    if len(cols) == len(columns):
                        row_data = {columns[i]: cols[i].get_text(strip=True) for i in range(len(cols))}
                        data.append(row_data)

                time.sleep(0.1)
            except Exception as e:
                print(f"Error fetching {page_url}: {e}")
                continue
        
        df = pd.DataFrame(data)
        if not df.empty and "Ticker" in df.columns:
            df.set_index("Ticker", inplace=True)
            df = df[~df.index.duplicated(keep='first')]

            # Save this section immediately and close the file
            with pd.HDFStore(hdf_file, mode="a") as store:
                store.put(key, df, format="table")
                print(f"Saved section {section} to disk with {len(df)} rows")

            df_sections.append(df)
        else:
            print(f"No data saved for section {section}")
        print(df)

    if df_sections:
        df_combined = pd.concat(df_sections, axis=1)
        unique_columns = df_combined.columns.unique()
        df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
        #Fix market cap data
        
        market_caps = df_combined['Market Cap'].values.tolist()
        for i in range(len(market_caps)):
            if type(market_caps[i]) is not str:
                continue
            if 'B' in market_caps[i]:
                market_caps[i] = float(market_caps[i].replace('B', '').replace(',', '')) * 1e9
            elif 'M' in market_caps[i]:
                market_caps[i] = float(market_caps[i].replace('M', '').replace(',', '')) * 1e6
        df_combined['Market Cap'] = market_caps

        df_combined.to_csv("finviz_data.csv")
        print("Combined data shape:", df_combined.shape)
        return df_combined
    else:
        print("No data collected.")
        return None


def get_large_cap_yf_data(csv_file='finviz_data.csv',
                           output_file='yf_data.csv',
                           min_market_cap=2e9,
                           years_lookback=5):
    today = dt.date.today()
    start = today - dt.timedelta(days=365 * years_lookback)
    end = today
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')

    # Load and filter stocks by market cap
    list_of_stocks = pd.read_csv(csv_file, index_col=0)
    list_of_stocks = list_of_stocks[list_of_stocks['Market Cap'] > min_market_cap]
    tickers = list(list_of_stocks.index)

    if not os.path.exists(output_file):
        print('Downloading data from Yahoo Finance...')
        yf_data = yf.download(tickers, start=start_str, end=end_str, group_by='ticker', auto_adjust=False)
        yf_close = yf_data.xs('Close', level='Price', axis=1)
        yf_close.to_csv(output_file)
    else:
        print('Reading data from file...')
        yf_close = pd.read_csv(output_file, index_col=0)
        yf_close.index = pd.to_datetime(yf_close.index)

    return yf_close

if __name__ == "__main__":
    

    get_large_cap_yf_data(csv_file='~/data/finviz_data.csv',
                           output_file='~/data/yf_data1.csv',
                           min_market_cap=1e9,
                           years_lookback=5)
    #download_fv_data(reset_cache=False)