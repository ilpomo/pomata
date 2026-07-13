"""
Canonical column-name constants for the test frames, so the OHLCV names live in one place instead of as string
literals scattered across the suite (a typo becomes impossible and a rename is one edit).

The price / volume names keep the canonical OHLCV reading order in every grouping (a quant reads ``open, high, low,
close, volume``); only ``import`` statements reorder them, since ruff sorts those by name. The ``returns`` /
``benchmark`` pair names the two aligned series the benchmark-relative metrics consume.
"""

OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
VOLUME = "volume"
COLUMN_X = "x"
GROUP_KEY = "g"
RETURNS = "returns"
BENCHMARK = "benchmark"

__all__ = ("BENCHMARK", "CLOSE", "COLUMN_X", "GROUP_KEY", "HIGH", "LOW", "OPEN", "RETURNS", "VOLUME")
