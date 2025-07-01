"""Core backtesting engine for option pairs."""
from typing import List, Tuple
import pandas as pd

from data.data_loader import DataLoader
from pairs.pair_selector import PairSelector
from strategies.strategy_config import StrategyConfig

class OptionBacktester:
    """Run backtests on pairs trading strategies using options data."""

    def __init__(self, data_loader: DataLoader, pair_selector: PairSelector):
        self.data_loader = data_loader
        self.pair_selector = pair_selector

    def run_backtest(self, symbols: List[str], config: StrategyConfig) -> pd.DataFrame:
        """Run a simple backtest over the provided symbols.

        Parameters
        ----------
        symbols : list of str
            Symbols to consider for pairs.
        config : StrategyConfig
            Strategy parameters.

        Returns
        -------
        pd.DataFrame
            Summary results.
        """
        price_data = {s: self.data_loader.load_price_data(s) for s in symbols}
        price_df = pd.concat(price_data, axis=1)
        pairs = self.pair_selector.select_pairs(price_df)
        # Placeholder: just return pairs and config
        return pd.DataFrame({"pair": pairs, "entry": config.zscore_entry})
