"""
Canonical column-name constants for the test frames, so the shared names live in one place instead of as string
literals scattered across the bespoke tests and the harness self-tests (a typo becomes impossible and a rename is
one edit).

The price names keep the canonical reading order in every grouping (a quant reads ``high, low, close``); only
``import`` statements reorder them, since ruff sorts those by name. The ``returns`` / ``benchmark`` pair names the
two aligned series the benchmark-relative metrics consume.
"""

HIGH = "high"
LOW = "low"
CLOSE = "close"
COLUMN_X = "x"
RETURNS = "returns"
BENCHMARK = "benchmark"

__all__ = ("BENCHMARK", "CLOSE", "COLUMN_X", "HIGH", "LOW", "RETURNS")
