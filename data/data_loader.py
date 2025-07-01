"""Data loading utilities for options and underlying data."""
from pathlib import Path
import pandas as pd

class DataLoader:
    """Load and cache options and underlying price data."""

    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_options_data(self, symbol: str) -> pd.DataFrame:
        """Load options data for the given symbol.

        Parameters
        ----------
        symbol: str
            Ticker symbol for the options chain.

        Returns
        -------
        pd.DataFrame
            Normalized options data.
        """
        # Placeholder implementation
        path = self.cache_dir / f"{symbol}_options.csv"
        if path.exists():
            return pd.read_csv(path)
        raise FileNotFoundError(f"Options data for {symbol} not found")

    def load_price_data(self, symbol: str) -> pd.DataFrame:
        """Load underlying price data for the given symbol."""
        path = self.cache_dir / f"{symbol}_prices.csv"
        if path.exists():
            return pd.read_csv(path, index_col=0, parse_dates=True)
        raise FileNotFoundError(f"Price data for {symbol} not found")
