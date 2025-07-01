"""Configuration objects for pairs trading strategies."""
from dataclasses import dataclass

@dataclass
class StrategyConfig:
    """Parameters controlling a pairs trading strategy."""

    zscore_entry: float = 2.0
    zscore_exit: float = 0.0
    holding_period: int = 5
    profit_target: float = 0.05
