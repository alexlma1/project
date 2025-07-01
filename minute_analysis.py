import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

file = "~/Downloads/2025-06-30.csv"
today = datetime.now()
df = pd.read_csv(file)
print(df.sort_values('transactions'))
df['root_symbol'] = df['ticker'].str.slice(2,-15)
df['exp'] = df['ticker'].str.slice(-15,-9)
df['opttype'] = df['ticker'].str.slice(-9,-8)
df['strike'] = df['ticker'].str.slice(-8,step=1).astype(float)
df['strike'] = df['strike'].astype(int) / 1000.0
df['dte'] = (pd.to_datetime(df['exp'], format='%y%m%d') - today).dt.days
df = df[df['exp']=='250718']
print(df)
df_grouped = df[['ticker','transactions']].groupby('ticker').sum()
print(df_grouped.sort_values('transactions', ascending=False))
series = df[df['ticker']=='O:AAPL250718C00210000']
series['window_start'] = pd.to_datetime(series['window_start'],unit='ns')
print(series)
series['close'].plot()
plt.show()