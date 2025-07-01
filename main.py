"""Example runner script for the options pairs framework."""
from data.data_loader import DataLoader
from pairs.pair_selector import PairSelector
from strategies.strategy_config import StrategyConfig
from engine.option_backtester import OptionBacktester


def main():
    loader = DataLoader()
    selector = PairSelector()
    backtester = OptionBacktester(loader, selector)
    config = StrategyConfig()

    # Example list of symbols
    symbols = ["AAPL", "MSFT"]
    try:
        results = backtester.run_backtest(symbols, config)
        print(results)
    except FileNotFoundError as exc:
        print(f"Data missing: {exc}")


if __name__ == "__main__":
    main()
