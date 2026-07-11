"""
Performance benchmarks and complexity-scaling checks for every PnL function (non-gating).

Kept out of the default run: the module skips unless ``POMATA_BENCHMARKS`` is set, and a dedicated, non-gating CI job
runs it. The pytest-benchmark timings give performance visibility (and a baseline for ``--benchmark-compare``); the
scaling check guards against an accidental super-linear regression — an O(n) vectorized accounting kernel slipping to
O(n^2) — which the absolute timings alone would not reveal. Every public function is covered, so a regression is caught
wherever it lands.
"""

import os
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from tests.support import fastest_eval

from pomata import pnl
from pomata.pnl import (
    cost_borrow,
    cost_fixed,
    cost_funding,
    cost_notional,
    cost_per_share,
    cost_proportional,
    cost_slippage,
    cumulative_pnl,
    dividend,
    equity_curve,
    pnl_gross,
    pnl_gross_inverse,
    pnl_net,
    returns_gross,
    returns_log,
    returns_net,
    returns_simple,
    turnover,
)

if not os.environ.get("POMATA_BENCHMARKS"):
    pytest.skip("set POMATA_BENCHMARKS=1 to run the benchmark tier", allow_module_level=True)


def _frame(size: int) -> pl.DataFrame:
    """
    A deterministic frame of ``size`` rows for timing, carrying both PnL flows: signed positions and positive prices
    (cash flow), signed weights and per-bar returns (return flow), and the intermediate series the net forms consume.
    """
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "quantity": rng.normal(100.0, 10.0, size),
            "price": 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.005, size)),
            "expr": 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.005, size)),
            "weight": rng.uniform(-1.0, 1.0, size),
            "returns": rng.normal(0.0005, 0.01, size),
            "asset_returns": rng.normal(0.0005, 0.01, size),
            "returns_gross": rng.normal(0.0005, 0.01, size),
            "rate": rng.uniform(0.0, 0.001, size),
            "dividend_per_share": np.abs(rng.normal(0.1, 0.05, size)),
            "pnl_gross": rng.normal(0.0, 1.0, size),
            "cost": np.abs(rng.normal(0.01, 0.005, size)),
        }
    )


CASES: dict[str, Callable[[], pl.Expr]] = {
    "cost_borrow": lambda: cost_borrow(pl.col("quantity"), pl.col("price"), rate=0.01),
    "cost_fixed": lambda: cost_fixed(pl.col("quantity"), fee=1.0),
    "cost_funding": lambda: cost_funding(pl.col("quantity"), pl.col("price"), pl.col("rate")),
    "cost_notional": lambda: cost_notional(pl.col("quantity"), pl.col("price"), rate=0.01),
    "cost_per_share": lambda: cost_per_share(pl.col("quantity"), fee=1.0),
    "cost_proportional": lambda: cost_proportional(pl.col("weight"), rate=0.01),
    "cost_slippage": lambda: cost_slippage(pl.col("weight"), half_spread=0.0005),
    "cumulative_pnl": lambda: cumulative_pnl(pl.col("returns")),
    "dividend": lambda: dividend(pl.col("quantity"), pl.col("dividend_per_share")),
    "equity_curve": lambda: equity_curve(pl.col("returns")),
    "pnl_gross": lambda: pnl_gross(pl.col("quantity"), pl.col("price")),
    "pnl_gross_inverse": lambda: pnl_gross_inverse(pl.col("quantity"), pl.col("price")),
    "pnl_net": lambda: pnl_net(pl.col("pnl_gross"), pl.col("cost")),
    "returns_gross": lambda: returns_gross(pl.col("weight"), pl.col("asset_returns")),
    "returns_log": lambda: returns_log(pl.col("expr")),
    "returns_net": lambda: returns_net(pl.col("returns_gross"), pl.col("cost")),
    "returns_simple": lambda: returns_simple(pl.col("expr")),
    "turnover": lambda: turnover(pl.col("weight")),
}


def test_cases_cover_public_surface() -> None:
    """
    Verifies that ``CASES`` covers exactly the public PnL surface, so a newly added function cannot slip the benchmark
    net (its complexity-scaling regression would otherwise go unmeasured).
    """
    assert set(CASES) == set(pnl.__all__)


@pytest.mark.benchmark
@pytest.mark.parametrize("name", sorted(CASES), ids=sorted(CASES))
def test_throughput(benchmark: BenchmarkFixture, name: str) -> None:
    """
    Records the evaluation time of each PnL function over a fixed-size frame.
    """
    frame = _frame(100_000)
    expr = CASES[name]().alias("y")
    benchmark(lambda: frame.select(expr))


@pytest.mark.benchmark
@pytest.mark.parametrize("name", sorted(CASES), ids=sorted(CASES))
def test_scales_sub_quadratically(name: str) -> None:
    """
    Verifies that a 10x increase in rows costs less than 25x the time (plus a small additive floor), guarding
    against an O(n^2) regression.

    The bound is multiplicative with a small additive floor: the cheapest functions evaluate in well under a
    millisecond, where the time is dominated by fixed overhead rather than the row count, and the floor keeps that
    measurement noise from failing them. A genuine super-linear regression blows the absolute time up to seconds at a
    million rows, far past either term.
    """
    build = CASES[name]
    base = fastest_eval(_frame(100_000), build)
    large = fastest_eval(_frame(1_000_000), build)
    assert large < 25.0 * base + 0.02, f"{name}: {large:.4f}s vs {base:.4f}s base for 10x the rows (super-linear?)"
