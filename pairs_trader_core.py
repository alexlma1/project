import numpy as np
import pandas as pd
import datetime as dt
import random
from statsmodels.tsa.api import VECM
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from numpy.linalg import lstsq
import itertools
import matplotlib.dates as mdates
from numpy.lib.stride_tricks import sliding_window_view
import numba


'''
Core trading logic and backtesting functionality.

PairsTrader class and all its methods:
    __init__
    fetch_data
    run_rolling_analysis
    backtest
    score_strategy
    plot_spread_series
    plot_weights
    run_strategy
    plot_positions
'''

def read_yf_data(file_name="yf_data.csv"):
    yf_data = pd.read_csv(file_name)
    yf_data = yf_data.dropna(axis=1, how='any')
    yf_data = yf_data.reset_index()
    yf_data = yf_data.drop('CHRD', axis=1, errors='ignore')
    yf_data = yf_data.drop('index', axis=1, errors='ignore')
    yf_data = yf_data.set_index('Date')
    yf_data.index = pd.to_datetime(yf_data.index)
    return yf_data


def load_iv_data(filepath="iv_df.csv"):
    df = pd.read_csv(filepath, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df
    


class PairsTrader:
    def __init__(self, stock1, stock2, start_date, end_date, lookback=130, yf_data=None, use_options=False, iv_data=None):
        self.stock1 = stock1
        self.stock2 = stock2
        self.start_date = start_date
        self.end_date = end_date
        self.lookback = lookback
        self.yf_data = yf_data
        self.use_options = use_options
        self.iv_data = iv_data

        self.data = None
        self.z_scores = None
        self.weights_list = []
        self.spread_series = None
        self.positions = None
        self.returns = None
        self.trade_entries = None
        self.trade_durations = []
        self.trade_log = []

    @staticmethod
    def black_scholes_price(S, K, T, r, sigma, option_type='call'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def fetch_data(self):
        self.data = self.yf_data[[self.stock1, self.stock2]].loc[self.start_date:self.end_date].dropna()
        self.smoothed_data = self.data.rolling(window=10).mean()

    def run_rolling_analysis(self, weight_zscore_window=160, weight_zscore_thresh=0.3, sma_window=80, z_score_window=200):
        z_scores = []
        spreads = []
        weights_all = []
        dates = []
        weight_history = []

        for i in range(self.lookback, len(self.data)):
            window_data = self.data.iloc[i - self.lookback:i]
            try:
                model = VECM(window_data.values, k_ar_diff=1, coint_rank=1, deterministic='n')
                results = model.fit()
                weights = results.beta.reshape(-1)
                weights = weights / np.sum(np.abs(weights))
            except:
                continue

            weight_history.append(weights)
            smoothed_weights = np.mean(weight_history[-sma_window:], axis=0) if len(weight_history) >= sma_window else weights
            spread = np.dot(window_data, smoothed_weights)
            z = (spread[-350:] - spread[-z_score_window:].mean()) / spread[-z_score_window:].std()

            weights_all.append(smoothed_weights.copy())
            spreads.append(spread[-1])
            z_scores.append(z[-1])
            dates.append(window_data.index[-1])

        self.z_scores = pd.Series(z_scores, index=dates)
        self.spread_series = pd.Series(spreads, index=dates)
        self.weights_list = weights_all
    def backtest(self, entry_threshold=2, exit_threshold=0.3, max_duration=30, take_profit=0.15):
        positions, returns, trade_entries, trading_spreads = [], [], [], []
        prev_position = 0
        fixed_weights = None
        trade_start_idx = None
        cumulative_return = 0.0
        current_trade = {}

        for i in range(1, len(self.z_scores)):
            z = self.z_scores.iloc[i]
            dynamic_weights = np.array(self.weights_list[i])
            idx = self.data.index[self.lookback + i]
            date = self.data.index[self.lookback + i]
            price_today = self.data.iloc[self.lookback + i]
            price_yesterday = self.data.iloc[self.lookback + i - 1]
            pct_returns = (price_today.values - price_yesterday.values) / price_yesterday.values

            position = prev_position
            entry = False
            exit_trade = False

            if prev_position == 0:
                cumulative_return = 0.0
                if z > entry_threshold:
                    position = -1
                    entry = True
                    fixed_weights = dynamic_weights.copy()
                    trade_start_idx = i
                elif z < -entry_threshold:
                    position = 1
                    entry = True
                    fixed_weights = dynamic_weights.copy()
                    trade_start_idx = i

                if entry:
                    w1, w2 = fixed_weights[0], fixed_weights[1]
                    abs_w1, abs_w2 = abs(w1), abs(w2)
                    opt1_type = 'call' if w1 > 0 else 'put'
                    opt2_type = 'call' if w2 > 0 else 'put'

                    current_trade = {
                        'Entry Date': idx,
                        'Entry Z-Score': z,
                        'Entry Position': position,
                        'Stock 1': self.stock1,
                        'Stock 2': self.stock2,
                        'Stock 1 Entry Price': price_today[self.stock1],
                        'Stock 2 Entry Price': price_today[self.stock2],
                        'Weight Stock 1': w1,
                        'Weight Stock 2': w2,
                        'Option Type 1': opt1_type,
                        'Option Type 2': opt2_type,
                        'Entry Option Size 1': abs_w1,
                        'Entry Option Size 2': abs_w2
                    }

                    if self.use_options:
                        S1, S2 = price_today[self.stock1], price_today[self.stock2]
                        K1, K2 = S1, S2
                        T = 30 / 252
                        r = 0.05
                        iv1 = self.iv_data[self.stock1].loc[date]
                        iv2 = self.iv_data[self.stock2].loc[date]
                        opt_price_1 = self.black_scholes_price(S1, K1, T, r, iv1, opt1_type)
                        opt_price_2 = self.black_scholes_price(S2, K2, T, r, iv2, opt2_type)
                        entry_cost = opt_price_1 * abs_w1 + opt_price_2 * abs_w2
                        current_trade.update({
                            'Option 1 Entry Price': opt_price_1,
                            'Option 2 Entry Price': opt_price_2,
                            'IV Stock 1 Entry': iv1,
                            'IV Stock 2 Entry': iv2,
                            'Entry Option Cost': entry_cost
                        })

            else:
                used_weights = fixed_weights
                duration = i - trade_start_idx + 1
                step_return = 0.0

                if self.use_options:
                    w1, w2 = used_weights[0], used_weights[1]
                    abs_w1, abs_w2 = abs(w1), abs(w2)
                    opt1_type = 'call' if w1 > 0 else 'put'
                    opt2_type = 'call' if w2 > 0 else 'put'

                    S1, S2 = price_today[self.stock1], price_today[self.stock2]
                    K1, K2 = S1, S2
                    T = 30 / 252
                    r = 0.05
                    iv1 = self.iv_data[self.stock1].loc[date]
                    iv2 = self.iv_data[self.stock2].loc[date]
                    opt_price_1 = self.black_scholes_price(S1, K1, T, r, iv1, opt1_type)
                    opt_price_2 = self.black_scholes_price(S2, K2, T, r, iv2, opt2_type)
                    entry_cost = current_trade.get('Entry Option Cost', 1e-8)
                    exit_value = opt_price_1 * abs_w1 + opt_price_2 * abs_w2
                    return_ratio = (exit_value - entry_cost) / abs(entry_cost)
                    cumulative_return = return_ratio
                    
                else:
                    spread_pct_return = np.dot(used_weights, pct_returns)
                    step_return = spread_pct_return * prev_position

                cumulative_return += step_return

                if (
                    (prev_position == 1 and z >= exit_threshold)
                    or (prev_position == -1 and z <= -exit_threshold)
                    or duration >= max_duration
                    or cumulative_return >= take_profit
                ):
                    position = 0
                    exit_trade = True

            used_weights = fixed_weights if prev_position != 0 else dynamic_weights
            spread_pct_return = np.dot(used_weights, pct_returns)
            realized_return = spread_pct_return * prev_position if prev_position != 0 else 0.0

            positions.append(position)
            returns.append(realized_return)
            trade_entries.append(1 if entry else 0)
            trading_spreads.append(np.dot(price_today.values, used_weights))

            if exit_trade and trade_start_idx is not None:
                end_date = self.data.index[self.lookback + i]
                end_price = self.data.iloc[self.lookback + i]
                w1, w2 = used_weights[0], used_weights[1]
                abs_w1, abs_w2 = abs(w1), abs(w2)
                opt1_type = 'call' if w1 > 0 else 'put'
                opt2_type = 'call' if w2 > 0 else 'put'

                current_trade.update({
                    'Exit Date': end_date,
                    'Exit Z-Score': z,
                    'Stock 1 Exit Price': end_price[self.stock1],
                    'Stock 2 Exit Price': end_price[self.stock2],
                    'Exit Weight Stock 1': w1,
                    'Exit Weight Stock 2': w2,
                    'Duration': i - trade_start_idx + 1
                })

                if self.use_options:
                    S1e, S2e = end_price[self.stock1], end_price[self.stock2]
                    K1, K2 = S1e, S2e
                    T = 30 / 252
                    r = 0.05
                    try:
                        iv1e = self.iv_data[self.stock1].loc[end_date]
                        iv2e = self.iv_data[self.stock2].loc[end_date]
                        opt_price_1e = self.black_scholes_price(S1e, K1, T, r, iv1e, opt1_type)
                        opt_price_2e = self.black_scholes_price(S2e, K2, T, r, iv2e, opt2_type)
                        exit_value = opt_price_1e * abs_w1 + opt_price_2e * abs_w2
                        entry_cost = current_trade.get('Entry Option Cost', 1e-8)
                        return_ratio = (exit_value - entry_cost) / abs(entry_cost)
                        cumulative_return = return_ratio
                        current_trade.update({
                            'Option 1 Exit Price': opt_price_1e,
                            'Option 2 Exit Price': opt_price_2e,
                            'IV Stock 1 Exit': iv1e,
                            'IV Stock 2 Exit': iv2e,
                            'Exit Option Value': exit_value
                        })
                    except Exception as e:
                        print(f"Error calculating option exit price: {e}")
                        cumulative_return = 0.0

                current_trade['Realized Return'] = cumulative_return
                self.trade_log.append(current_trade)
                current_trade = {}
                trade_start_idx = None
                cumulative_return = 0.0

            prev_position = position

        self.positions = pd.Series(positions, index=self.z_scores.index[1:])
        self.returns = pd.Series(returns, index=self.z_scores.index[1:])
        self.trade_entries = pd.Series(trade_entries, index=self.z_scores.index[1:])
        self.trading_spread_series = pd.Series(trading_spreads, index=self.z_scores.index[1:])


    def score_strategy(self):
        returns = self.returns
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        volatility = self.spread_series.std()
        frequency = np.mean((self.z_scores.abs() > 1.4).astype(int))
        correlation = self.data[self.stock1].pct_change().corr(self.data[self.stock2].pct_change())
        total_return = returns.sum()
        n_trades = self.trade_entries.sum()
        in_position_returns = returns[self.positions != 0]
        avg_return_per_trade = in_position_returns.sum() / n_trades if n_trades > 0 else 0.0
        avg_trade_duration = np.mean(self.trade_durations) if self.trade_durations else 0

        self.score = {
            'Sharpe': sharpe,
            'Volatility': volatility,
            'Signal Frequency': frequency,
            'Total Return': total_return,
            'Average Return Per Trade': avg_return_per_trade,
            'Number of Trades': int(n_trades),
            'Average Trade Duration': avg_trade_duration,
            'Correlation': correlation
        }
        return self.score

    def run_strategy(self):
        self.fetch_data()
        self.run_rolling_analysis()
        self.backtest()
        self.score_strategy()
        return self.score

    def plot_spread_series(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.spread_series, label='Spread', color='blue')
        plt.axhline(self.spread_series.mean(), color='red', linestyle='--', label='Mean Spread')
        plt.axhline(self.spread_series.mean() + self.spread_series.std(), color='green', linestyle='--', label='Upper Band')
        plt.axhline(self.spread_series.mean() - self.spread_series.std(), color='orange', linestyle='--', label='Lower Band')
        plt.title(f'Spread Series for {self.stock1} and {self.stock2}')
        plt.xlabel('Date')
        plt.ylabel('Spread')
        plt.legend()
        plt.show()
        plt.close()

    def plot_weights(self):
        weights_df = pd.DataFrame(self.weights_list, index=self.z_scores.index, columns=[self.stock1, self.stock2])
        plt.figure(figsize=(14, 7))
        plt.plot(weights_df)
        plt.title(f'Weights for {self.stock1} and {self.stock2}')
        plt.xlabel('Date')
        plt.ylabel('Weights')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close()

    def plot_positions(self):
        if self.positions is None or self.spread_series is None:
            print("No positions or spread data to plot.")
            return

        fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        # === 1. Spread and Position ===
        axs[0].plot(self.spread_series, label='Spread', color='blue')
        axs[0].axhline(self.spread_series.mean(), color='black', linestyle='--', label='Mean')
        axs[0].axhline(self.spread_series.mean() + self.spread_series.std(), color='green', linestyle='--', label='+1σ')
        axs[0].axhline(self.spread_series.mean() - self.spread_series.std(), color='red', linestyle='--', label='-1σ')
        


        # Mark long/short entries
        long_entries = self.positions[self.positions == 1].index
        short_entries = self.positions[self.positions == -1].index
        flat_positions = self.positions[self.positions == 0].index

        axs[0].scatter(long_entries, self.spread_series.loc[long_entries], marker='^', color='green', label='Long Entry', zorder=5)
        axs[0].scatter(short_entries, self.spread_series.loc[short_entries], marker='v', color='red', label='Short Entry', zorder=5)

        axs[0].set_title(f"Spread with Trading Signals: {self.stock1} / {self.stock2}")
        axs[0].set_ylabel("Spread")
        axs[0].legend()
        axs[0].grid(True)
        axs[1].grid(True)
        axs[2].grid(True)
        axs[1].plot(self.z_scores, label='Z-Score', color='orange' )
        # plot (stock1 - stock2), the difference in prices

        axs[2].plot(self.data[self.stock1] - self.data[self.stock2], label='Price Difference', color='purple')
        

        # === 2. Prices of Stock 1 and Stock 2 ===
        axs[3].plot(self.data[self.stock1], label=self.stock1, color='black')
        axs[3].plot(self.data[self.stock2], label=self.stock2, color='gray')

        for trade in getattr(self, "trade_log", []):
            entry_date = trade.get('Entry Date')
            exit_date = trade.get('Exit Date')
            pos = trade.get('Entry Position')
            if entry_date and exit_date:
                color = 'green' if pos == 1 else 'red'
                axs[3].axvspan(entry_date, exit_date, color=color, alpha=0.15)
                axs[2].axvspan(entry_date, exit_date, color=color, alpha=0.15)
                axs[1].axvspan(entry_date, exit_date, color=color, alpha=0.15)

        axs[3].set_title("Stock Prices with Trade Durations")
        axs[3].set_ylabel("Price")
        axs[3].legend()
        axs[3].grid(True)

        # Format x-axis
        axs[3].xaxis.set_major_locator(mdates.MonthLocator())
        axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        plt.tight_layout()
        plt.show()

