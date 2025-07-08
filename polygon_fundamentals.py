from polygon import RESTClient

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from polygon import RESTClient

def get_flattened_fundamentals(ticker: str, api_key: str = "QIPCCY3i8zC8TtHo0raWZMsK7NAMm8ic") -> pd.DataFrame:
    """
    Fetch and flatten quarterly financials for a given ticker from Polygon.io.
    
    Each row in the output DataFrame represents a quarter, and each column is
    a financial metric from the balance sheet, income statement, etc.
    All metrics are retained even if they are None (set as NaN).
    
    Args:
        ticker (str): Stock ticker symbol.
        api_key (str): Polygon.io API key.
    
    Returns:
        pd.DataFrame: Flattened quarterly financials.
    """
    client = RESTClient(api_key)
    print(f"Fetching financials for {ticker}...")

    try:
        financials = list(client.vx.list_stock_financials(
            ticker=ticker,
            filing_date_gt="2015-01-01",
        ))
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data for {ticker}: {e}")

    if not financials or not isinstance(financials, list):
        print(f"⚠️ No financials data found for {ticker}")
        return pd.DataFrame()

    df = pd.DataFrame(financials)
    all_rows = []

    for _, row in df.iterrows():
        row_data = {
            "filing_date": row.get("filing_date"),
            "end_date": row.get("end_date"),
            "fiscal_period": row.get("fiscal_period"),
            "fiscal_year": row.get("fiscal_year"),
        }

        financial_sections = row.get("financials", {})
        for section_name, section_data in financial_sections.items():
            if not isinstance(section_data, dict):
                continue

            for col_name, item in section_data.items():
                key = f"{section_name}.{col_name}"

                if isinstance(item, dict):
                    row_data[key] = item.get("value", float("nan"))
                elif item is None:
                    row_data[key] = float("nan")
                else:
                    row_data[key] = float("nan")

        all_rows.append(row_data)

    flat_df = pd.DataFrame(all_rows).sort_values("filing_date", ascending=False)
    return flat_df

def save_fundamentals_to_hdf(tickers, hdf_path, api_key):
    """
    Fetch and store fundamentals for multiple tickers into a single HDF5 file.

    Args:
        tickers (list of str): List of stock tickers.
        hdf_path (str): Path to output HDF5 file.
        api_key (str): Polygon.io API key.
    """
    i=0
    n = len(tickers)
    with pd.HDFStore(hdf_path, mode='w') as store:
        for ticker in tickers:
            print(f"📥 Processing {ticker}... {i}/{n}")
            i+=1
            df = get_flattened_fundamentals(ticker, api_key=api_key)
            store.put(f"/{ticker}", df, format="table")
            print(f"✅ Saved {ticker} to {hdf_path}")
        

hdf_path = "~/data/quarterly_fundamentals.h5"
polygon_api_key = "QIPCCY3i8zC8TtHo0raWZMsK7NAMm8ic"

#tickers = pd.read_csv("~/data/finviz_data.csv")["Ticker"]
#save_fundamentals_to_hdf(tickers, hdf_path, api_key=polygon_api_key)

with pd.HDFStore(hdf_path, mode='r') as store:
    df = store.get("/CDE")
df = df.sort_values("end_date")
print(df)
for i in df.columns:
    print(i)
df = df.set_index("end_date")
df = df[df['fiscal_period'].isin(['Q1', 'Q2', 'Q3', 'Q4'])]
df = df.filter(regex=r"^cash_flow_statement\.")
(df.pct_change()+1.0).cumprod().plot(legend=False)
plt.show()