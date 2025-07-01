import numpy as np
import pandas as pd
import datetime as dt
import random
from statsmodels.tsa.api import VECM
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from numpy.linalg import lstsq
import itertools
import matplotlib.dates as mdates
from numpy.lib.stride_tricks import sliding_window_view
import numba
from pairs_trader_core import PairsTrader, read_yf_data
'''
Pair selection, statistical testing, and utilities.

'''

def run_single_pair_parallel(args):
    stock1, stock2, start_date, end_date = args
    yf_data = read_yf_data()
    trader = PairsTrader(stock1, stock2, start_date, end_date, yf_data=yf_data)
    score = trader.run_strategy()
    return {
        'Stock 1': stock1,
        'Stock 2': stock2,
        'Sharpe': score['Sharpe'],
        'Volatility': score['Volatility'],
        'Signal Frequency': score['Signal Frequency'],
        'Total Return': score['Total Return'],
        'Average Return Per Trade': score['Average Return Per Trade'],
        'Number of Trades': score['Number of Trades'],
        'Average Trade Duration': score['Average Trade Duration'],
        'Correlation': score['Correlation']
    }

def monte_carlo_pairs_test_parallel(stock_universe, n_trials=10, seed=42, max_workers=8):
    random.seed(seed)
    pairs = [random.sample(stock_universe, 2) for _ in range(n_trials)]
    end_date = dt.datetime.now().strftime("%Y-%m-%d")
    start_date = (dt.datetime.now() - dt.timedelta(days=2*365)).strftime("%Y-%m-%d")
    args_list = [(s1, s2, start_date, end_date) for s1, s2 in pairs]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_pair_parallel, args) for args in args_list]
        for fut in futures:
            try:
                results.append(fut.result())
            except Exception as e:
                print("Error in parallel worker:", e)
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='Sharpe', ascending=False).reset_index(drop=True)
    return df_results
@numba.njit(fastmath=True, cache=True)
def rolling_fast_coint(s1_log, s2_log, window, lags):
    n = len(s1_log)
    n_rolls = n - window + 1
    pvals = np.ones(n_rolls)
    crit_vals = np.array([-2.66, -1.95, -1.61])
    p_vals = np.array([0.01, 0.05, 0.10])
    for i in range(n_rolls):
        x = s1_log[i:i+window]
        y = s2_log[i:i+window]
        # OLS regression: y = beta*x + intercept
        X = np.empty((window, 2), dtype=np.float64)
        X[:, 0] = x
        X[:, 1] = 1.0
        coef = fast_ols(X, y)
        resid = y - np.dot(X, coef)
        dy = resid[1:] - resid[:-1]
        y_lag = resid[:-1]
        if len(y_lag) <= np.max(lags) + 1:
            pvals[i] = 1.0
            continue
        best_stat = 0.0
        for lag in lags:
            if len(y_lag) <= lag + 1:
                continue
            # Build lagged Xadf matrix
            n_obs = len(y_lag) - lag
            Xadf = np.empty((n_obs, 1 + lag), dtype=np.float64)
            Xadf[:, 0] = y_lag[lag:]
            for j in range(lag):
                Xadf[:, 1 + j] = dy[lag - j - 1: -j - 1]
            Yadf = dy[lag:]
            coef_adf = fast_ols(Xadf, Yadf)
            resid_adf = Yadf - np.dot(Xadf, coef_adf)
            stat = coef_adf[0] / np.std(resid_adf)
            if (lag == lags[0]) or (stat < best_stat):
                best_stat = stat
        # Convert stat to p-value (approximate, fast)
        if best_stat < crit_vals[0]:
            pval = 0.005
        elif best_stat > crit_vals[-1]:
            pval = 0.5
        else:
            # Linear interpolation
            idx = np.searchsorted(crit_vals, best_stat)
            if idx == 0:
                pval = p_vals[0]
            elif idx >= len(crit_vals):
                pval = p_vals[-1]
            else:
                x0, x1 = crit_vals[idx-1], crit_vals[idx]
                y0, y1 = p_vals[idx-1], p_vals[idx]
                pval = y0 + (best_stat - x0) * (y1 - y0) / (x1 - x0)
        pvals[i] = pval
    return pvals


def stable_pair_worker_fast(pair_list, price_data, min_corr, max_pvalue,
                            window, min_corr_ratio, min_coint_ratio, lags):
    stable_pairs = []
    log_prices = np.log(price_data.to_numpy())
    stock_names = price_data.columns.to_list()
    stock_idx = {name: i for i, name in enumerate(stock_names)}
    for s1, s2 in pair_list:
        i1, i2 = stock_idx[s1], stock_idx[s2]
        p1, p2 = log_prices[:, i1], log_prices[:, i2]
        mask = (~np.isnan(p1)) & (~np.isnan(p2))
        if np.sum(mask) < window * 2:
            continue
        p1 = p1[mask]
        p2 = p2[mask]
        returns1 = np.diff(p1)
        returns2 = np.diff(p2)
        rolling_corrs = rolling_corr_np(returns1, returns2, window)
        high_corr_ratio = np.mean(rolling_corrs > min_corr)
        # Call Numba rolling coint
        pvals = rolling_fast_coint(p1, p2, window, np.array(lags))
        low_pval_ratio = np.mean(pvals < max_pvalue)
        if high_corr_ratio >= min_corr_ratio and low_pval_ratio >= min_coint_ratio:
            stable_pairs.append((s1, s2))
    return stable_pairs

def find_stable_pairs_parallel_fast(price_data, 
                                    min_corr=0.7, 
                                    max_pvalue=0.05, 
                                    window=120, 
                                    min_corr_ratio=0.7, 
                                    min_coint_ratio=0.7, 
                                    max_workers=8,
                                    lags=[1,2,3,4]):
    """
    Parallelized stable pair finder using fast_coint (no statsmodels).
    """
    stocks = price_data.columns.to_list()
    all_pairs = list(itertools.combinations(stocks, 2))
    n_pairs = len(all_pairs)
    print(f"Checking {n_pairs} pairs across {max_workers} workers...")

    chunk_size = (n_pairs + max_workers - 1) // max_workers
    chunks = [all_pairs[i:i+chunk_size] for i in range(0, n_pairs, chunk_size)]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(
            stable_pair_worker_fast, chunk, price_data, min_corr, max_pvalue,
            window, min_corr_ratio, min_coint_ratio, lags
        ) for chunk in chunks]

        for i, fut in enumerate(futures):
            try:
                res = fut.result()
                print(f"Worker {i+1} found {len(res)} stable pairs.")
                results.extend(res)
            except Exception as e:
                print(f"Worker {i+1} failed: {e}")

    print(f"Total stable pairs found: {len(results)}")
    results = pd.DataFrame(results, columns=['Stock 1', 'Stock 2'])
    return results

def fast_adf(y, lags=[1, 2, 3, 4]):
    y = np.asarray(y)
    y = y[~np.isnan(y)]
    best_stat = None
    for lag in lags:
        if len(y) <= lag + 1:
            continue
        dy = np.diff(y)
        y_lagged = y[:-1]
        X = np.column_stack([y_lagged[lag:]] +
                            [dy[lag - i - 1:-i - 1] for i in range(lag)])
        y_target = dy[lag:]
        try:
            coef, _, _, _ = lstsq(X, y_target, rcond=None)
            stat = coef[0] / np.std(y_target - X @ coef)
            if best_stat is None or stat < best_stat:
                best_stat = stat
        except Exception:
            continue

    if best_stat is None:
        return 0.0, 1.0

    crit_vals = [-2.66, -1.95, -1.61]
    p_vals = [0.01, 0.05, 0.10]

    if best_stat < crit_vals[0]:
        pval = 0.005
    elif best_stat > crit_vals[-1]:
        pval = 0.5
    else:
        for i in range(len(crit_vals) - 1):
            if crit_vals[i] <= best_stat <= crit_vals[i + 1]:
                x0, x1 = crit_vals[i], crit_vals[i + 1]
                y0, y1 = p_vals[i], p_vals[i + 1]
                pval = y0 + (best_stat - x0) * (y1 - y0) / (x1 - x0)
                break

    return best_stat, pval

def fast_coint(y0, y1, lags=[1, 2, 3, 4]):
    y0 = np.asarray(y0)
    y1 = np.asarray(y1)
    X = np.vstack([y1, np.ones(len(y1))]).T
    beta, _, _, _ = lstsq(X, y0, rcond=None)
    resid = y0 - X @ beta
    stat, pval = fast_adf(resid, lags=lags)
    return stat, pval

def find_stable_pairs(price_data, 
                      min_corr=0.4, 
                      max_pvalue=0.05, 
                      window=120, 
                      min_corr_ratio=0.5, 
                      min_coint_ratio=0.5,
                      lags=[1,2,3,4]):
    """
    Non-parallel stable pair finder using fast_coint().
    Returns a list of (stock1, stock2) pairs.
    """
    stocks = price_data.columns.to_list()
    all_pairs = list(itertools.combinations(stocks, 2))
    stable_pairs = []

    print(f"Checking {len(all_pairs)} pairs...")

    for idx, (s1, s2) in enumerate(all_pairs):
        try:
            s1_prices = price_data[s1].dropna()
            s2_prices = price_data[s2].dropna()
            df = pd.concat([s1_prices, s2_prices], axis=1).dropna()

            if len(df) < window * 2:
                continue

            s1_log = np.log(df[s1])
            s2_log = np.log(df[s2])
            returns1 = s1_log.diff().dropna()
            returns2 = s2_log.diff().dropna()

            # Rolling correlation
            rolling_corr = returns1.rolling(window).corr(returns2)
            high_corr_ratio = np.mean(rolling_corr > min_corr)

            # Rolling cointegration p-values
            rolling_pvals = []
            for k in range(len(df) - window):
                s1w = s1_log.iloc[k:k+window]
                s2w = s2_log.iloc[k:k+window]
                _, pval = fast_coint(s1w, s2w, lags=lags)
                rolling_pvals.append(pval)

            low_pval_ratio = np.mean(np.array(rolling_pvals) < max_pvalue)

            if high_corr_ratio >= min_corr_ratio and low_pval_ratio >= min_coint_ratio:
                stable_pairs.append((s1, s2))
                print(f"{s1}-{s2} OK: Corr={high_corr_ratio:.2f}, Coint={low_pval_ratio:.2f}")

            if idx % 100 == 0:
                print(f"Checked {idx}/{len(all_pairs)} pairs...")

        except Exception as e:
            print(f"Failed {s1}-{s2}: {e}")
            continue

    print(f"Finished. Found {len(stable_pairs)} stable pairs.")
    return stable_pairs

def find_stable_pairs_by_crosscorr(price_data, 
                                   lags=range(-5, 6), 
                                   min_corr=0.3, 
                                   min_obs=120, 
                                   min_corr_ratio=0.7, 
                                   window=120):
    """
    Finds stable lead-lag stock pairs using rolling cross-correlation.

    Parameters:
        price_data (pd.DataFrame): Log price data (or raw price, will be logged inside).
        lags (iterable): Lags to test for lead-lag relationships (positive = stock1 leads).
        min_corr (float): Minimum correlation threshold to consider "high".
        min_obs (int): Minimum number of data points required to evaluate a pair.
        min_corr_ratio (float): Required ratio of windows exceeding min_corr.
        window (int): Rolling window size for correlation.

    Returns:
        pd.DataFrame: List of stable pairs with lag info and correlation stats.
    """
    from itertools import combinations
    stocks = price_data.columns.tolist()
    all_pairs = list(combinations(stocks, 2))
    stable_pairs = []

    print(f"Checking {len(all_pairs)} pairs using rolling cross-correlation...")

    for idx, (s1, s2) in enumerate(all_pairs):
        try:
            s1_log = np.log(price_data[s1]).dropna()
            s2_log = np.log(price_data[s2]).dropna()
            df = pd.concat([s1_log, s2_log], axis=1).dropna()
            if len(df) < min_obs:
                continue

            ret1 = df[s1].diff().dropna()
            ret2 = df[s2].diff().dropna()

            best_ratio = 0
            best_lag = None
            best_avg_corr = 0

            for lag in lags:
                if lag > 0:
                    shifted_ret2 = ret2.shift(lag)
                    common = ret1.index.intersection(shifted_ret2.index)
                    series1 = ret1.loc[common]
                    series2 = shifted_ret2.loc[common]
                elif lag < 0:
                    shifted_ret1 = ret1.shift(-lag)
                    common = shifted_ret1.index.intersection(ret2.index)
                    series1 = shifted_ret1.loc[common]
                    series2 = ret2.loc[common]
                else:
                    series1 = ret1
                    series2 = ret2

                aligned = pd.concat([series1, series2], axis=1).dropna()
                if len(aligned) < min_obs:
                    continue

                rolling_corr = aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])
                high_corr_ratio = (rolling_corr.abs() > min_corr).mean()
                avg_corr = rolling_corr.mean()

                if high_corr_ratio > best_ratio:
                    best_ratio = high_corr_ratio
                    best_lag = lag
                    best_avg_corr = avg_corr

            if best_ratio >= min_corr_ratio:
                stable_pairs.append({
                    'Stock 1': s1,
                    'Stock 2': s2,
                    'Best Lag': best_lag,
                    'Stable Corr Ratio': best_ratio,
                    'Average Corr': best_avg_corr
                })

        except Exception as e:
            print(f"Failed {s1}-{s2}: {e}")

        if idx % 50 == 0 or idx == len(all_pairs) - 1:
            print(f"Checked {idx+1}/{len(all_pairs)} pairs...")
    
    df_stable = pd.DataFrame(stable_pairs)
    print(f"Finished. Found {len(df_stable)} stable pairs.")
    return df_stable
def score_single_pair_by_crosscorr(args):
    s1, s2, returns, lags, min_corr, min_obs, min_corr_ratio, window = args
    try:
        ret1 = returns[s1].dropna()
        ret2 = returns[s2].dropna()
        common_idx = ret1.index.intersection(ret2.index)

        if len(common_idx) < min_obs:
            return None

        #calculate an adf stat


        ret1 = ret1.loc[common_idx]
        ret2 = ret2.loc[common_idx]

        best_ratio = 0
        best_lag = None
        best_avg_corr = 0
        cross_corrs = []

        for lag in lags:
            if lag > 0:
                shifted_ret2 = ret2.shift(lag)
                common = ret1.index.intersection(shifted_ret2.index)
                series1 = ret1.loc[common]
                series2 = shifted_ret2.loc[common]
            elif lag < 0:
                shifted_ret1 = ret1.shift(-lag)
                common = shifted_ret1.index.intersection(ret2.index)
                series1 = shifted_ret1.loc[common]
                series2 = ret2.loc[common]
            else:
                series1 = ret1
                series2 = ret2

            aligned = pd.concat([series1, series2], axis=1).dropna()
            if len(aligned) < min_obs:
                continue

            rolling_corr = aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])
            high_corr_ratio = (rolling_corr.abs() > min_corr).mean()
            avg_corr = rolling_corr.mean()
            cross_corrs.append(avg_corr)
            if high_corr_ratio > best_ratio:
                best_ratio = high_corr_ratio
                best_lag = lag
                best_avg_corr = avg_corr

        if best_ratio >= min_corr_ratio:
            return {
                'Stock 1': s1,
                'Stock 2': s2,
                'Best Lag': best_lag,
                'Stable Corr Ratio': best_ratio,
                'Average Corr': best_avg_corr,
                'Cross Correlations': np.mean(cross_corrs)
            }
    except Exception:
        return None

def find_stable_pairs_by_crosscorr_parallel(price_data,
                                            lags=range(-5, 6),
                                            min_corr=0.3,
                                            min_obs=120,
                                            min_corr_ratio=0.7,
                                            window=120,
                                            max_workers=8):
    log_prices = np.log(price_data)
    returns = log_prices.diff().dropna()
    stocks = price_data.columns.tolist()
    all_pairs = list(itertools.combinations(stocks, 2))

    print(f"Checking {len(all_pairs)} pairs using parallel cross-correlation...")

    args_list = [
        (s1, s2, returns, lags, min_corr, min_obs, min_corr_ratio, window)
        for s1, s2 in all_pairs
    ]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(score_single_pair_by_crosscorr, args_list)
        for idx, result in enumerate(futures):
            if idx % 50 == 0 or idx == len(all_pairs) - 1:
                print(f"Checked {idx+1}/{len(all_pairs)} pairs...")
            if result:
                results.append(result)

    df_stable = pd.DataFrame(results)
    print(f"Finished. Found {len(df_stable)} stable pairs.")
    return df_stable

def rolling_corr_np(a, b, window):
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) < window:
        return np.full(len(a), np.nan)
    shape = (len(a) - window + 1, window)
    strides = (a.strides[0], a.strides[0])
    a_w = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    b_w = np.lib.stride_tricks.as_strided(b, shape=shape, strides=strides)
    a_mean = a_w.mean(axis=1)
    b_mean = b_w.mean(axis=1)
    cov = ((a_w - a_mean[:, None]) * (b_w - b_mean[:, None])).mean(axis=1)
    a_std = a_w.std(axis=1)
    b_std = b_w.std(axis=1)
    corr = cov / (a_std * b_std)
    return np.concatenate([np.full(window-1, np.nan), corr])

def rolling_corr(a, b, window):
    # Fast rolling correlation using numpy
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) != len(b):
        raise ValueError("Inputs must be same length")
    n = len(a)
    if n < window:
        return np.full(n, np.nan)
    a_ma = np.convolve(a, np.ones(window)/window, mode='valid')
    b_ma = np.convolve(b, np.ones(window)/window, mode='valid')
    a2_ma = np.convolve(a**2, np.ones(window)/window, mode='valid')
    b2_ma = np.convolve(b**2, np.ones(window)/window, mode='valid')
    ab_ma = np.convolve(a*b, np.ones(window)/window, mode='valid')
    cov = ab_ma - a_ma*b_ma
    a_std = np.sqrt(a2_ma - a_ma**2)
    b_std = np.sqrt(b2_ma - b_ma**2)
    corr = cov / (a_std * b_std)
    # Pad with NaN at front to match pandas rolling
    return np.concatenate([np.full(window-1, np.nan), corr])

def ols_coef(X, y):
    # Closed-form OLS: (X^T X)^-1 X^T y
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)

def faster_adf(y, lags=[1, 2, 3, 4]):
    y = np.asarray(y)
    y = y[~np.isnan(y)]
    best_stat = None
    for lag in lags:
        n = len(y)
        if n <= lag + 1:
            continue
        dy = np.diff(y)
        y_lagged = y[:-1]
        # Efficiently create lagged X using strides
        # X: [y_lagged_lagged, dY_lag-1, ..., dY_0]
        X = np.column_stack([
            y_lagged[lag:],
            *(dy[lag - i - 1: -i - 1] for i in range(lag))
        ])
        y_target = dy[lag:]
        try:
            coef = ols_coef(X, y_target)
            resid = y_target - X @ coef
            stat = coef[0] / np.std(resid, ddof=len(coef))
            if best_stat is None or stat < best_stat:
                best_stat = stat
        except Exception:
            continue

    if best_stat is None:
        return 0.0, 1.0

    crit_vals = [-2.66, -1.95, -1.61]
    p_vals = [0.01, 0.05, 0.10]
    if best_stat < crit_vals[0]:
        pval = 0.005
    elif best_stat > crit_vals[-1]:
        pval = 0.5
    else:
        for i in range(len(crit_vals) - 1):
            if crit_vals[i] <= best_stat <= crit_vals[i + 1]:
                x0, x1 = crit_vals[i], crit_vals[i + 1]
                y0, y1 = p_vals[i], p_vals[i + 1]
                pval = y0 + (best_stat - x0) * (y1 - y0) / (x1 - x0)
                break

    return best_stat, pval

def faster_coint(y0, y1, lags=[1, 2, 3, 4]):
    y0 = np.asarray(y0)
    y1 = np.asarray(y1)
    X = np.vstack([y1, np.ones(len(y1))]).T
    # Use closed-form OLS instead of lstsq
    beta = ols_coef(X, y0)
    resid = y0 - X @ beta
    stat, pval = faster_adf(resid, lags=lags)
    return stat, pval
@numba.njit(inline='always')
def fast_ols(X, y):
    # Solve (X^T X) beta = X^T y for beta
    XtX = np.dot(X.T, X)
    Xty = np.dot(X.T, y)
    return np.linalg.solve(XtX, Xty)
def fast_adf(y, lags=[1, 2, 3, 4]):
    y = np.asarray(y)
    y = y[~np.isnan(y)]
    best_stat = None
    for lag in lags:
        if len(y) <= lag + 1:
            continue
        dy = np.diff(y)
        y_lagged = y[:-1]
        X = np.column_stack([y_lagged[lag:]] +
                            [dy[lag - i - 1:-i - 1] for i in range(lag)])
        y_target = dy[lag:]
        try:
            coef, _, _, _ = lstsq(X, y_target, rcond=None)
            stat = coef[0] / np.std(y_target - X @ coef)
            if best_stat is None or stat < best_stat:
                best_stat = stat
        except Exception:
            continue

    if best_stat is None:
        return 0.0, 1.0

    crit_vals = [-2.66, -1.95, -1.61]
    p_vals = [0.01, 0.05, 0.10]

    if best_stat < crit_vals[0]:
        pval = 0.005
    elif best_stat > crit_vals[-1]:
        pval = 0.5
    else:
        for i in range(len(crit_vals) - 1):
            if crit_vals[i] <= best_stat <= crit_vals[i + 1]:
                x0, x1 = crit_vals[i], crit_vals[i + 1]
                y0, y1 = p_vals[i], p_vals[i + 1]
                pval = y0 + (best_stat - x0) * (y1 - y0) / (x1 - x0)
                break

    return best_stat, pval

def fast_coint(y0, y1, lags=[1, 2, 3, 4]):
    y0 = np.asarray(y0)
    y1 = np.asarray(y1)
    X = np.vstack([y1, np.ones(len(y1))]).T
    beta, _, _, _ = lstsq(X, y0, rcond=None)
    resid = y0 - X @ beta
    stat, pval = fast_adf(resid, lags=lags)
    return stat, pval
