"""Utility functions for the framework."""
from pathlib import Path


def ensure_dir(path: str) -> Path:
    """Create a directory if it doesn't exist and return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
