"""
Performance benchmarks and complexity-scaling checks for every metric (non-gating).

Kept out of the default run: the module skips unless ``POMATA_BENCHMARKS`` is set, and a dedicated, non-gating CI job
runs it. The pytest-benchmark timings give performance visibility (and a baseline for ``--benchmark-compare``); the
scaling check guards against an accidental super-linear regression — an O(n) reduction or rolling window slipping to
O(n^2) — which the absolute timings alone would not reveal. Every public metric is covered, so a regression is caught
wherever it lands.
"""

import os
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from tests.support import fastest_eval

from pomata import metrics
from pomata.metrics import (
    adjusted_sharpe_ratio,
    alpha,
    alpha_rolling,
    beta,
    beta_rolling,
    burke_ratio,
    cagr,
    cagr_rolling,
    calmar_ratio,
    capture_downside_ratio,
    capture_ratio,
    capture_upside_ratio,
    common_sense_ratio,
    conditional_drawdown_at_risk,
    conditional_value_at_risk,
    downside_deviation,
    downside_deviation_rolling,
    drawdown,
    drawdown_rolling,
    gain_to_pain_ratio,
    information_ratio,
    information_ratio_rolling,
    kelly_criterion,
    kurtosis,
    kurtosis_rolling,
    max_drawdown,
    max_drawdown_duration,
    modigliani_risk_adjusted_performance,
    omega_ratio,
    omega_ratio_rolling,
    pain_index,
    pain_ratio,
    payoff_ratio,
    probabilistic_sharpe_ratio,
    profit_factor,
    recovery_ratio,
    risk_of_ruin,
    sharpe_ratio,
    sharpe_ratio_rolling,
    skewness,
    skewness_rolling,
    sortino_ratio,
    sortino_ratio_rolling,
    stability,
    sterling_ratio,
    tail_ratio,
    tail_ratio_rolling,
    total_return,
    total_return_rolling,
    treynor_ratio,
    treynor_ratio_rolling,
    ulcer_index,
    ulcer_performance_ratio,
    value_at_risk,
    value_at_risk_modified,
    value_at_risk_parametric,
    value_at_risk_rolling,
    volatility,
    volatility_rolling,
    win_rate,
)

if not os.environ.get("POMATA_BENCHMARKS"):
    pytest.skip("set POMATA_BENCHMARKS=1 to run the benchmark tier", allow_module_level=True)


def _frame(size: int) -> pl.DataFrame:
    """
    A deterministic frame of ``size`` rows for timing: a return series, a benchmark series, and a positive equity curve.
    """
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.01, size)
    benchmark = rng.normal(0.0004, 0.009, size)
    return pl.DataFrame(
        {
            "returns": returns,
            "benchmark": benchmark,
            "equity_curve": 100.0 * np.cumprod(1.0 + returns),
        }
    )


CASES: dict[str, Callable[[], pl.Expr]] = {
    "adjusted_sharpe_ratio": lambda: adjusted_sharpe_ratio(pl.col("returns"), periods_per_year=252),
    "alpha": lambda: alpha(pl.col("returns"), pl.col("benchmark"), periods_per_year=252),
    "alpha_rolling": lambda: alpha_rolling(pl.col("returns"), pl.col("benchmark"), 20, periods_per_year=252),
    "beta": lambda: beta(pl.col("returns"), pl.col("benchmark")),
    "beta_rolling": lambda: beta_rolling(pl.col("returns"), pl.col("benchmark"), 20),
    "burke_ratio": lambda: burke_ratio(pl.col("equity_curve"), periods_per_year=252),
    "cagr": lambda: cagr(pl.col("equity_curve"), periods_per_year=252),
    "cagr_rolling": lambda: cagr_rolling(pl.col("equity_curve"), 20, periods_per_year=252),
    "calmar_ratio": lambda: calmar_ratio(pl.col("equity_curve"), periods_per_year=252),
    "capture_downside_ratio": lambda: capture_downside_ratio(
        pl.col("returns"), pl.col("benchmark"), periods_per_year=252
    ),
    "capture_ratio": lambda: capture_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252),
    "capture_upside_ratio": lambda: capture_upside_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252),
    "common_sense_ratio": lambda: common_sense_ratio(pl.col("returns")),
    "conditional_drawdown_at_risk": lambda: conditional_drawdown_at_risk(pl.col("equity_curve")),
    "conditional_value_at_risk": lambda: conditional_value_at_risk(pl.col("returns")),
    "downside_deviation": lambda: downside_deviation(pl.col("returns"), periods_per_year=252),
    "downside_deviation_rolling": lambda: downside_deviation_rolling(pl.col("returns"), 20, periods_per_year=252),
    "drawdown": lambda: drawdown(pl.col("equity_curve")),
    "drawdown_rolling": lambda: drawdown_rolling(pl.col("equity_curve"), 20),
    "gain_to_pain_ratio": lambda: gain_to_pain_ratio(pl.col("returns")),
    "information_ratio": lambda: information_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252),
    "information_ratio_rolling": lambda: information_ratio_rolling(
        pl.col("returns"), pl.col("benchmark"), 20, periods_per_year=252
    ),
    "kelly_criterion": lambda: kelly_criterion(pl.col("returns")),
    "kurtosis": lambda: kurtosis(pl.col("returns")),
    "kurtosis_rolling": lambda: kurtosis_rolling(pl.col("returns"), 20),
    "max_drawdown": lambda: max_drawdown(pl.col("equity_curve")),
    "max_drawdown_duration": lambda: max_drawdown_duration(pl.col("equity_curve")),
    "modigliani_risk_adjusted_performance": lambda: modigliani_risk_adjusted_performance(
        pl.col("returns"), pl.col("benchmark"), periods_per_year=252
    ),
    "omega_ratio": lambda: omega_ratio(pl.col("returns")),
    "omega_ratio_rolling": lambda: omega_ratio_rolling(pl.col("returns"), 20),
    "pain_index": lambda: pain_index(pl.col("equity_curve")),
    "pain_ratio": lambda: pain_ratio(pl.col("equity_curve"), periods_per_year=252),
    "payoff_ratio": lambda: payoff_ratio(pl.col("returns")),
    "probabilistic_sharpe_ratio": lambda: probabilistic_sharpe_ratio(pl.col("returns"), periods_per_year=252),
    "profit_factor": lambda: profit_factor(pl.col("returns")),
    "recovery_ratio": lambda: recovery_ratio(pl.col("equity_curve")),
    "risk_of_ruin": lambda: risk_of_ruin(pl.col("returns")),
    "sharpe_ratio": lambda: sharpe_ratio(pl.col("returns"), periods_per_year=252),
    "sharpe_ratio_rolling": lambda: sharpe_ratio_rolling(pl.col("returns"), 20, periods_per_year=252),
    "skewness": lambda: skewness(pl.col("returns")),
    "skewness_rolling": lambda: skewness_rolling(pl.col("returns"), 20),
    "sortino_ratio": lambda: sortino_ratio(pl.col("returns"), periods_per_year=252),
    "sortino_ratio_rolling": lambda: sortino_ratio_rolling(pl.col("returns"), 20, periods_per_year=252),
    "stability": lambda: stability(pl.col("returns")),
    "sterling_ratio": lambda: sterling_ratio(pl.col("equity_curve"), periods_per_year=252),
    "tail_ratio": lambda: tail_ratio(pl.col("returns")),
    "tail_ratio_rolling": lambda: tail_ratio_rolling(pl.col("returns"), 20),
    "total_return": lambda: total_return(pl.col("equity_curve")),
    "total_return_rolling": lambda: total_return_rolling(pl.col("equity_curve"), 20),
    "treynor_ratio": lambda: treynor_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252),
    "treynor_ratio_rolling": lambda: treynor_ratio_rolling(
        pl.col("returns"), pl.col("benchmark"), 20, periods_per_year=252
    ),
    "ulcer_index": lambda: ulcer_index(pl.col("equity_curve")),
    "ulcer_performance_ratio": lambda: ulcer_performance_ratio(pl.col("equity_curve"), periods_per_year=252),
    "value_at_risk": lambda: value_at_risk(pl.col("returns")),
    "value_at_risk_modified": lambda: value_at_risk_modified(pl.col("returns")),
    "value_at_risk_parametric": lambda: value_at_risk_parametric(pl.col("returns")),
    "value_at_risk_rolling": lambda: value_at_risk_rolling(pl.col("returns"), 20),
    "volatility": lambda: volatility(pl.col("returns"), periods_per_year=252),
    "volatility_rolling": lambda: volatility_rolling(pl.col("returns"), 20, periods_per_year=252),
    "win_rate": lambda: win_rate(pl.col("returns")),
}


@pytest.mark.benchmark
def test_cases_cover_public_surface() -> None:
    """
    Verifies that ``CASES`` covers exactly the public metric surface, so a newly added metric cannot slip the benchmark
    net (its complexity-scaling regression would otherwise go unmeasured).
    """
    assert set(CASES) == set(metrics.__all__)


@pytest.mark.benchmark
@pytest.mark.parametrize("name", sorted(CASES), ids=sorted(CASES))
def test_throughput(benchmark: BenchmarkFixture, name: str) -> None:
    """
    Records the evaluation time of each metric over a fixed-size frame.
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

    The bound is multiplicative with a small additive floor: the cheapest metrics reduce in well under a millisecond,
    where the time is dominated by fixed overhead rather than the row count, and the floor keeps that measurement noise
    from failing them. A genuine super-linear regression blows the absolute time up to seconds at a million rows, far
    past either term.
    """
    build = CASES[name]
    base = fastest_eval(_frame(100_000), build)
    large = fastest_eval(_frame(1_000_000), build)
    assert large < 25.0 * base + 0.02, f"{name}: {large:.4f}s vs {base:.4f}s base for 10x the rows (super-linear?)"
