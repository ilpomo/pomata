"""
Reproduce the realized indicator-vs-oracle agreement that calibrates the 1e-10 precision band.

Run from the repo root::

    uv run python scripts/calibrate_tolerances.py

For a representative set of well-conditioned indicators, this draws many random price-like series across several seeds,
runs each indicator against its independent reference oracle, and reports the worst realized *relative* residual on the
defined outputs -- the measurement behind CORRECTNESS.md's claim that the agreement sits comfortably inside the 1e-10
guarantee. It prints a per-indicator table and the overall worst case, and exits non-zero if any well-conditioned
residual breaches 1e-10. The oracles and indicators are the same ones the test suite uses, so the figure is recomputable
from a clean clone rather than asserted.
"""

import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import polars as pl

# The oracles live under tests/; make the repo root importable when run as a plain script (as precision_table.py does).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pomata.indicators import dema, ema, hma, rma, rsi, sma, tema, trima, wma  # noqa: E402
from tests.indicators.oracles import (  # noqa: E402
    dema_reference,
    ema_reference,
    hma_reference,
    rma_reference,
    rsi_reference,
    sma_reference,
    tema_reference,
    trima_reference,
    wma_reference,
)

GUARANTEE = 1e-10
SEEDS = (1, 7, 424242)
DRAWS_PER_SEED = 64
OUTPUT_FLOOR = 1e-6  # skip near-zero outputs: the relative band is ill-conditioned there (documented in CORRECTNESS.md)

Indicator = Callable[[pl.Expr, int], pl.Expr]
Oracle = Callable[[Sequence[float], int], list[float | None]]

# A representative set of well-conditioned, single-input (close, window) indicators -- the recursive and windowed means
# the 1e-10 band is claimed for. Each pairs the shipped factory with its independent reference oracle.
CURATED: tuple[tuple[str, Indicator, Oracle, int], ...] = (
    ("sma", sma, sma_reference, 20),
    ("ema", ema, ema_reference, 20),
    ("rma", rma, rma_reference, 14),
    ("wma", wma, wma_reference, 20),
    ("hma", hma, hma_reference, 16),
    ("dema", dema, dema_reference, 20),
    ("tema", tema, tema_reference, 20),
    ("trima", trima, trima_reference, 20),
    ("rsi", rsi, rsi_reference, 14),
)


def _price_series(rng: np.random.Generator, length: int) -> list[float]:
    """A positive, well-conditioned price-like random walk -- the domain where the 1e-10 band is claimed to hold."""
    return (100.0 + np.cumsum(rng.normal(0.0, 1.0, length))).tolist()


def _worst_relative(indicator: Indicator, oracle: Oracle, window: int) -> float:
    """The worst relative residual between the indicator and its oracle over the fuzzed, well-conditioned domain."""
    worst = 0.0
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for _ in range(DRAWS_PER_SEED):
            series = _price_series(rng, int(rng.integers(window + 5, 200)))
            shipped = pl.DataFrame({"close": series}).select(indicator(pl.col("close"), window).alias("y"))["y"].to_list()
            reference = oracle(series, window)
            for got, ref in zip(shipped, reference, strict=True):
                if got is None or ref is None or abs(ref) < OUTPUT_FLOOR:
                    continue
                worst = max(worst, abs(got - ref) / abs(ref))
    return worst


def main() -> None:
    print(f"Realized indicator-vs-oracle agreement ({len(SEEDS)} seeds x {DRAWS_PER_SEED} draws), guarantee {GUARANTEE:.0e}:\n")
    print("| indicator | worst relative residual |")
    print("| --- | :-: |")
    overall = 0.0
    for name, indicator, oracle, window in CURATED:
        worst = _worst_relative(indicator, oracle, window)
        overall = max(overall, worst)
        print(f"| `{name}({window})` | `{worst:.1e}` |")
    verdict = "INSIDE" if overall < GUARANTEE else "OUTSIDE"
    print(f"\nworst across the representative set: {overall:.2e}  ({verdict} the {GUARANTEE:.0e} guarantee)")
    sys.exit(0 if overall < GUARANTEE else 1)


if __name__ == "__main__":
    main()
