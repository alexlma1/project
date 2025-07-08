from __future__ import annotations
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
from scipy.stats import norm

from pathlib import Path
from typing import Iterable, Dict, List
from tqdm.auto import tqdm


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

def read_yf_data(file_name="~/data/yf_data.csv"):
    yf_data = pd.read_csv(file_name)
    yf_data = yf_data.dropna(axis=1, how='any')
    yf_data = yf_data.reset_index()
    yf_data = yf_data.drop('CHRD', axis=1, errors='ignore')
    yf_data = yf_data.drop('index', axis=1, errors='ignore')
    yf_data = yf_data.set_index('Date')
    yf_data.index = pd.to_datetime(yf_data.index)
    return yf_data


def load_iv_data(filepath="~/data/iv_df.csv"):
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
        self.option_duration = 45

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
                    if position == 1:
                        opt1_type = 'call'
                        opt2_type = 'put'
                    elif position == -1:
                        opt1_type = 'put'
                        opt2_type = 'call'

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
                        T = self.option_duration / 365
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
                    if position == 1:
                        opt1_type = 'call'
                        opt2_type = 'put'
                    elif position == -1:
                        opt1_type = 'put'
                        opt2_type = 'call'

                    S1, S2 = price_today[self.stock1], price_today[self.stock2]
                    K1, K2 = current_trade.get('Stock 1 Entry Price', S1), current_trade.get('Stock 2 Entry Price', S2)

                    entry_date = current_trade.get('Entry Date')
                    if entry_date is not None:
                        duration_days = (date - entry_date).days
                        remaining_days = max(self.option_duration - duration_days, 1)
                        T = remaining_days / 365
                    else:
                        T = self.option_duration / 365  # fallback

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
                if position == 1:
                    opt1_type = 'call'
                    opt2_type = 'put'
                elif position == -1:
                    opt1_type = 'put'
                    opt2_type = 'call'

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
                    K1, K2 = current_trade.get('Stock 1 Entry Price', S1e), current_trade.get('Stock 2 Entry Price', S2e)
                    duration_days = self.option_duration - (end_date - current_trade['Entry Date']).days
                    T = max(duration_days / 365, 1 / 365)  # Ensure non-zero
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
        # Log final trade if still open at the end of backtest
        if prev_position != 0 and current_trade:
            final_date = self.data.index[self.lookback + len(self.z_scores) - 1]
            final_price = self.data.iloc[self.lookback + len(self.z_scores) - 1]
            w1, w2 = fixed_weights[0], fixed_weights[1]
            abs_w1, abs_w2 = abs(w1), abs(w2)
            if position == 1:
                    opt1_type = 'call'
                    opt2_type = 'put'
            elif position == -1:
                opt1_type = 'put'
                opt2_type = 'call'

            current_trade.update({
                'Exit Date': final_date,
                'Exit Z-Score': self.z_scores.iloc[-1],
                'Stock 1 Exit Price': final_price[self.stock1],
                'Stock 2 Exit Price': final_price[self.stock2],
                'Exit Weight Stock 1': w1,
                'Exit Weight Stock 2': w2,
                'Duration': len(self.z_scores) - trade_start_idx
            })

            if self.use_options:
                S1e, S2e = final_price[self.stock1], final_price[self.stock2]
                K1, K2 = current_trade.get('Stock 1 Entry Price', S1e), current_trade.get('Stock 2 Entry Price', S2e)
                duration_days = self.option_duration - (final_date - current_trade['Entry Date']).days
                T = max(duration_days / 365, 1 / 365)
                r = 0.05
                try:
                    iv1e = self.iv_data[self.stock1].loc[final_date]
                    iv2e = self.iv_data[self.stock2].loc[final_date]
                    opt_price_1e = self.black_scholes_price(S1e, K1, T, r, iv1e, opt1_type)
                    opt_price_2e = self.black_scholes_price(S2e, K2, T, r, iv2e, opt2_type)
                    exit_value = opt_price_1e * abs_w1 + opt_price_2e * abs_w2
                    entry_cost = current_trade.get('Entry Option Cost', 1e-8)
                    return_ratio = (exit_value - entry_cost) / abs(entry_cost)
                    current_trade.update({
                        'Option 1 Exit Price': opt_price_1e,
                        'Option 2 Exit Price': opt_price_2e,
                        'IV Stock 1 Exit': iv1e,
                        'IV Stock 2 Exit': iv2e,
                        'Exit Option Value': exit_value,
                        'Realized Return': return_ratio
                    })
                except Exception as e:
                    print(f"Error calculating final option valuation: {e}")
                    current_trade['Realized Return'] = None
            else:
                current_trade['Realized Return'] = cumulative_return

            self.trade_log.append(current_trade)

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


# ── main class ───────────────────────────────────────────────────────────────
class BasketTrader:
    """
    Generic pairs/basket trading back-tester that supports an arbitrary
    number *n* ≥ 2 of stocks.

    Parameters
    ----------
    tickers      : list[str]
        Ordered list of symbols in the basket.
    start_date   : date-like
    end_date     : date-like
    lookback     : int, default 130
        Size of the rolling window (in trading days) fed to VECM.
    yf_data      : pd.DataFrame, required
        Pre-downloaded price data (close prices) for *all* tickers.
        Index must be datetime.
    use_options  : bool, default False
        If True, the back-tester prices at-the-money options rather than the
        underlying.  (The option block has been generalised but you can
        safely leave it switched off while you verify the cash logic.)
    iv_data      : pd.DataFrame, optional
        Implied-vol surface: same index as `yf_data`, same columns as
        `tickers`; contains *annualised* σ.
    """
    # ─────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        tickers      : List[str],
        start_date   ,
        end_date     ,
        lookback: int = 130,
        yf_data      : pd.DataFrame | None = None,
        *,
        use_options  : bool = False,
        iv_data      : pd.DataFrame | None = None
    ) -> None:
        if len(tickers) < 2:
            raise ValueError("BasketTrader needs at least two tickers")

        self.tickers        = list(tickers)
        self.start_date     = start_date
        self.end_date       = end_date
        self.lookback       = lookback
        self.use_options    = use_options
        self.option_duration: int  = 45      # days to expiry for ATM options
        self.yf_data        = yf_data
        self.iv_data        = iv_data

        # runtime containers
        self.data            : pd.DataFrame | None = None
        self.smoothed_data   : pd.DataFrame | None = None
        self.z_scores        : pd.Series  | None = None
        self.spread_series   : pd.Series  | None = None
        self.weights_list    : List[np.ndarray] = []   # len == len(z_scores)
        self.positions       : pd.Series  | None = None
        self.returns         : pd.Series  | None = None
        self.trade_entries   : pd.Series  | None = None
        self.trading_spread_series: pd.Series | None = None
        self.option_prices: pd.DataFrame | None = None

        self.trade_log       : List[dict] = []
        self.trade_durations : list[int]  = []
        self.score           : Dict[str, float] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # static helper
    @staticmethod
    def black_scholes_price(
        S: float, K: float, T: float, r: float, sigma: float, *,
        option_type: str = "call"
    ) -> float:
        """European Black-Scholes price."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. DATA
    def fetch_data(self) -> None:
        """
        Cut the master price DataFrame to the desired date range and
        drop rows with *any* missing prices.
        """
        if self.yf_data is None:
            raise ValueError("yf_data must be supplied (already downloaded).")

        self.data = (
            self.yf_data[self.tickers]
            .loc[self.start_date : self.end_date]
            .dropna()
        )
        # a 10-day SMA used only for plotting clarity
        self.smoothed_data = self.data.rolling(window=10).mean()

    # ─────────────────────────────────────────────────────────────────────────
    # 2. ROLLING COINTEGRATION ANALYSIS
    def run_rolling_analysis(
        self,
        *,
        weight_sma_window: int = 80,
        z_score_window  : int = 200,
        weight_z_thresh : float = 0.3  # (kept for future use – not enforced)
    ) -> None:
        """
        • For every day `t >= lookback`, fit a rank-1 VECM on the previous
          `lookback` days of *close* prices.

        • Extract β (cointegration vector) → basket weights.

        • Optionally smooth those weights with an SMA (length
          `weight_sma_window`) to reduce churn.

        • Create spread = βᵀ·P    and rolling z-score of that spread
          over `z_score_window` days.
        """
        if self.data is None:
            self.fetch_data()

        z_scores: list[float]        = []
        spreads : list[float]        = []
        weights_all: list[np.ndarray] = []
        dates   : list[pd.Timestamp] = []
        weight_hist: list[np.ndarray] = []

        for i in range(self.lookback, len(self.data)):
            win_prices = self.data.iloc[i - self.lookback : i]
            try:
                model   = VECM(win_prices.values,
                               k_ar_diff=1,
                               coint_rank=1,
                               deterministic="n")
                res     = model.fit()
                beta    = res.beta.reshape(-1)          # shape (n,)
                beta    = beta / np.sum(np.abs(beta))   # scale: Σ|w| = 1
            except Exception:
                continue

            weight_hist.append(beta)
            smooth_beta = (
                np.mean(weight_hist[-weight_sma_window:], axis=0)
                if len(weight_hist) >= weight_sma_window
                else beta
            )

            # last row of the window (prices at time i-1)
            spread = np.dot(win_prices.iloc[-1].values, smooth_beta)

            # collect rolling z-score
            roll = pd.Series(
                np.dot(win_prices.values, smooth_beta),
                index=win_prices.index
            )
            z = (
                (roll.iloc[-1] - roll.iloc[-z_score_window:].mean())
                / roll.iloc[-z_score_window:].std(ddof=0)
            )

            z_scores.append(z)
            spreads.append(spread)
            weights_all.append(smooth_beta.copy())
            dates.append(win_prices.index[-1])

        self.z_scores      = pd.Series(z_scores, index=dates, name="z")
        self.spread_series = pd.Series(spreads , index=dates, name="spread")
        self.weights_list  = weights_all

    # ─────────────────────────────────────────────────────────────────────────
    # 3. BACK-TEST
    def backtest(
        self,
        *,
        entry_threshold : float =  2.0,
        exit_threshold  : float =  0.3,
        max_duration    : int   = 30,
        take_profit     : float =  2.0
    ) -> None:
        """
        Simple mean-reversion rules identical to the 2-asset version:

        * Open *basket* position (= ±1) when |z| > entry_threshold.
        * Close when z crosses exit_threshold back toward zero **or**
          when a stop/time condition is hit.

        • Cash PnL: w·ΔP / P * position

        • If `use_options=True`, the same idea is applied but on ATM
          Black-Scholes prices.  The option block has been generalised for
          *n* stocks, but you are free to leave it switched off.
        """
        if self.z_scores is None:
            self.run_rolling_analysis()
        if self.use_options:
            idx = self.data.index[self.lookback + 1:]  # same length as returns
            self.option_prices = pd.DataFrame(index=idx,
                                              columns=self.tickers,
                                              dtype="float64")
        positions, returns, entries, trade_spreads = [], [], [], []
        prev_position  = 0
        fixed_weights  = None
        trade_start_ix = None
        cumulative_ret = 0.0
        current_trade  : dict = {}

        # tqdm loop – progress bar helps on long back-tests
        for i in tqdm(range(1, len(self.z_scores)), desc="Back-test", dynamic_ncols=True):
            z         = self.z_scores.iloc[i]
            d_index   = self.data.index[self.lookback + i]
            price_now = self.data.iloc[self.lookback + i]
            price_prev= self.data.iloc[self.lookback + i - 1]

            pct_ret_vec = (price_now.values - price_prev.values) / price_prev.values
            dyn_weights = np.array(self.weights_list[i])

            position   = prev_position
            entry_flag = False
            exit_flag  = False

            # ─── Entry logic ───
            if prev_position == 0:
                cumulative_ret = 0.0
                if z >  entry_threshold:
                    position      = -1
                    entry_flag    = True
                    fixed_weights = dyn_weights.copy()
                    trade_start_ix= i
                elif z < -entry_threshold:
                    position      =  1
                    entry_flag    = True
                    fixed_weights = dyn_weights.copy()
                    trade_start_ix= i

                if entry_flag:
                    current_trade = {
                        "Entry Date"      : d_index,
                        "Entry Z"         : float(z),
                        "Position"        : int(position),
                        "Weights"         : dict(zip(self.tickers, fixed_weights)),
                        "Entry Prices"    : price_now[self.tickers].to_dict(),
                    }

                    if self.use_options:
                        opt_recs = []
                        T  = self.option_duration / 365
                        r  = 0.05
                        for tk, w in zip(self.tickers, fixed_weights):
                            opt_type = ("call" if (position * np.sign(w) > 0) else "put")
                            S  = price_now[tk]
                            iv = self.iv_data[tk].loc[d_index]
                            opt_price = self.black_scholes_price(S, S, T, r, iv, option_type=opt_type)
                            opt_recs.append(
                                dict(ticker=tk,
                                     weight = float(w),
                                     opt_type = opt_type,
                                     iv_entry = float(iv),
                                     opt_entry_price = float(opt_price))
                            )
                        current_trade["Option Legs"] = opt_recs
                        current_trade["Entry Option Cost"] = float(
                            sum(abs(rec["weight"]) * rec["opt_entry_price"] for rec in opt_recs)
                        )

            # ─── Update open position ───
            else:
                w = fixed_weights
                return_step = np.dot(w, pct_ret_vec) * prev_position

                if self.use_options:
                    # Mark-to-market option basket
                    opt_val = 0.0
                    T_rem   = max(self.option_duration - (d_index - current_trade["Entry Date"]).days, 1) / 365
                    r       = 0.05
                    new_legs= []
                    for rec in current_trade["Option Legs"]:
                        tk   = rec["ticker"]
                        S    = price_now[tk]
                        K    = current_trade['Entry Prices'][tk]
                        iv   = self.iv_data[tk].loc[d_index]
                        opt_p= self.black_scholes_price(S, K, T_rem, r, iv, option_type=rec["opt_type"])
                        new_legs.append({**rec, "iv_now": float(iv), "opt_now_price": float(opt_p)})
                        opt_val += abs(rec["weight"]) * opt_p
                        self.option_prices.at[d_index, tk] = opt_p

                    entry_cost   = current_trade["Entry Option Cost"]
                    cumulative_ret = (opt_val - entry_cost) / abs(entry_cost)
                    current_trade["Option Legs MTM"] = new_legs
                    
                else:
                    cumulative_ret += return_step

                duration = i - trade_start_ix + 1
                close_cond = (
                    (prev_position ==  1 and z >=  exit_threshold) or
                    (prev_position == -1 and z <= -exit_threshold) or
                    duration >= max_duration or
                    cumulative_ret >= take_profit
                )
                if close_cond:
                    position  = 0
                    exit_flag = True

            # ─── Record daily stats ───
            used_w      = fixed_weights if prev_position else dyn_weights
            spread_now  = np.dot(price_now.values, used_w)
            day_ret     = np.dot(used_w, pct_ret_vec) * prev_position if prev_position else 0.0

            positions.append(position)
            returns  .append(day_ret)
            entries  .append(1 if entry_flag else 0)
            trade_spreads.append(spread_now)

            # ─── Handle exit ───
            if exit_flag and trade_start_ix is not None:
                exit_prices = price_now[self.tickers].to_dict()
                current_trade.update({
                    "Exit Date"   : d_index,
                    "Exit Z"      : float(z),
                    "Exit Prices" : exit_prices,
                    "Duration"    : duration,
                    "Realized Return": float(cumulative_ret)
                })
                self.trade_log.append(current_trade)
                self.trade_durations.append(duration)
                current_trade  = {}
                trade_start_ix = None
                cumulative_ret = 0.0

            prev_position = position

        if prev_position != 0 and current_trade:
            # Log the last open trade if it was not closed
            final_date = self.data.index[self.lookback + len(self.z_scores) - 1]
            final_price = self.data.iloc[self.lookback + len(self.z_scores) - 1]
            exit_prices = final_price[self.tickers].to_dict()
            current_trade.update({
                "Exit Date"   : final_date,
                "Exit Z"      : self.z_scores.iloc[-1],
                "Exit Prices" : exit_prices,
                "Duration"    : len(self.z_scores) - trade_start_ix,
                "Realized Return": float(cumulative_ret)
            })
            self.trade_log.append(current_trade)
            self.trade_durations.append(len(self.z_scores) - trade_start_ix)
            
                



        # ── Convert daily series
        idx = self.z_scores.index[1:]  # align: skip z[0] because we diffed pct_ret_vec from t-1
        self.positions             = pd.Series(positions, index=idx, name="position")
        self.returns               = pd.Series(returns  , index=idx, name="daily_ret")
        self.trade_entries         = pd.Series(entries  , index=idx, name="entry_flag")
        self.trading_spread_series = pd.Series(trade_spreads, index=idx, name="trading_spread")
        # ──────────────────────────────────────────────────────────────────
    def flatten_trade_log(self, *, prefix_options: bool = True, cache: bool = True):
        """
        Squash `self.trade_log` into a pandas DataFrame (one row per trade).

        • Nested dicts → col names like  'Weights_AAPL',  'Exit_Prices_MSFT'
        • Option legs  → 'opt_AAPL_opt_type', 'opt_AAPL_iv_entry', …
        """
        import pandas as pd

        def _flat(d: dict, root: str = ""):
            for k, v in d.items():
                k = f"{root}{k.replace(' ', '_')}"
                if isinstance(v, dict):
                    yield from _flat(v, f"{k}_")                 # recurse
                elif isinstance(v, list) and v and isinstance(v[0], dict):
                    for leg in v:
                        tk = leg.get("ticker", "")
                        for kk, vv in leg.items():
                            if kk == "ticker":
                                continue
                            col = f"{'opt_' if prefix_options else ''}{tk}_{kk}"
                            yield col, vv
                else:
                    yield k, v

        df = pd.DataFrame([dict(_flat(tr)) for tr in self.trade_log])\
               .sort_values("Entry_Date", ignore_index=True)

        if cache:
            self.trade_log_df = df
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 4. SCORE
    def score_strategy(self) -> Dict[str, float]:
        """Compute Sharpe, volatility, frequency, etc."""
        returns = self.returns
        sharpe  = returns.mean() / returns.std(ddof=0) * np.sqrt(252) if returns.std(ddof=0) > 0 else 0
        score = dict(
            Sharpe             = float(sharpe),
            Volatility_spread  = float(self.spread_series.std(ddof=0)),
            Signal_frequency   = float(np.mean((self.z_scores.abs() > 1.4).astype(int))),
            Total_return       = float(returns.sum()),
            Trades             = int(self.trade_entries.sum()),
            Avg_ret_per_trade  = float(returns[self.positions != 0].sum() / max(1, self.trade_entries.sum())),
            Avg_trade_duration = float(np.mean(self.trade_durations) if self.trade_durations else 0.0),
        )
        # simple average pairwise correlation as an additional diagnostic
        corrs = self.data.pct_change().corr().values
        upper = corrs[np.triu_indices_from(corrs, k=1)]
        score["Avg_pairwise_corr"] = float(np.mean(upper))
        self.score = score
        return score

    # ─────────────────────────────────────────────────────────────────────────
    # 5. HIGH-LEVEL WRAPPER
    def run_strategy(self) -> Dict[str, float]:
        self.fetch_data()
        self.run_rolling_analysis()
        self.backtest()
        return self.score_strategy()

    # ─────────────────────────────────────────────────────────────────────────
    # 6. PLOTS (spread / weights / positions)
    def plot_spread(self) -> None:
        if self.spread_series is None:
            print("Run the strategy first.")
            return
        mu, sigma = self.spread_series.mean(), self.spread_series.std(ddof=0)
        plt.figure(figsize=(12, 6))
        plt.plot(self.spread_series, label="Spread")
        plt.axhline(mu,         linestyle="--", color="black", label="Mean")
        plt.axhline(mu + sigma, linestyle="--", color="green", label="+1σ")
        plt.axhline(mu - sigma, linestyle="--", color="red"  , label="-1σ")
        plt.title("Basket Spread")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    def plot_weights(self) -> None:
        if not self.weights_list:
            print("Weights not available yet.")
            return
        w_df = pd.DataFrame(self.weights_list, index=self.z_scores.index,
                            columns=self.tickers)
        w_df.plot(figsize=(12, 6), title="Rolling Basket Weights")
        plt.grid(True); plt.tight_layout(); plt.show()

    def plot_positions(self) -> None:
        if self.positions is None:
            print("Run back-test first.")
            return
        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        # 1) spread with entries
        ax[0].plot(self.spread_series, label="Spread")
        longs  = self.positions[self.positions ==  1].index
        shorts = self.positions[self.positions == -1].index
        ax[0].scatter(longs , self.spread_series.loc[longs ], color="green", marker="^", label="Long entry")
        ax[0].scatter(shorts, self.spread_series.loc[shorts], color="red"  , marker="v", label="Short entry")
        ax[0].legend(); ax[0].grid(True); ax[0].set_title("Spread & Signals")
        # 2) cumulative PnL
        self.returns.cumsum().plot(ax=ax[1])
        ax[1].set_title("Cumulative Basket PnL"); ax[1].grid(True)
        ax[1].xaxis.set_major_locator(mdates.MonthLocator())
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.tight_layout(); plt.show()
