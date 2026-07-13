"""Spec for ``pomata.metrics.treynor_ratio`` — reducing, annualized excess return per unit of beta, scale-exempt."""

import math
from collections.abc import Sequence

import polars as pl
from tests.metrics.oracles import treynor_ratio_reference
from tests.support import complete_benchmark, well_spread
from tests_new.support.spec import ScaleExempt, Shape, Spec, SpecPin

from pomata.metrics import beta, treynor_ratio

_BETA_FLOOR = 5e-2


def _beta_nondegenerate(returns: Sequence[float | None], benchmark: Sequence[float | None]) -> bool:
    """Whether the regression slope is bounded away from zero: treynor divides the excess mean by it."""
    pairs = [
        (value_returns, value_benchmark)
        for value_returns, value_benchmark in zip(returns, benchmark, strict=True)
        if value_returns is not None
        and value_benchmark is not None
        and not math.isnan(value_returns)
        and not math.isnan(value_benchmark)
    ]
    if not well_spread([value_returns for value_returns, _ in pairs]):
        return False
    if len(pairs) < 2:
        return True
    complete_returns = [value for value, _ in pairs]
    complete_bench = [value for _, value in pairs]
    mean_returns = sum(complete_returns) / len(pairs)
    mean_benchmark = sum(complete_bench) / len(pairs)
    variance = sum((value - mean_benchmark) ** 2 for value in complete_bench) / len(pairs)
    if variance == 0.0:
        return False
    covariance = sum(
        (value_returns - mean_returns) * (value_benchmark - mean_benchmark)
        for value_returns, value_benchmark in zip(complete_returns, complete_bench, strict=True)
    ) / len(pairs)
    return abs(covariance / variance) >= _BETA_FLOOR


def _treynor_conditioning(frame: pl.DataFrame) -> bool:
    """A well-spread benchmark and a beta bounded away from zero — the two regimes treynor's quotient needs."""
    returns = frame["returns"].to_list()
    benchmark = frame["benchmark"].to_list()
    return well_spread(complete_benchmark(returns, benchmark)) and _beta_nondegenerate(returns, benchmark)


def _treynor_component() -> pl.Expr:
    """Treynor recomposed from the public ``beta`` factory at the spec's default params (rf 0.0, 252 periods)."""
    rf_period = math.pow(1.0 + 0.0, 1.0 / 252) - 1.0
    return (pl.col("returns") - rf_period).mean() * 252 / beta(pl.col("returns"), pl.col("benchmark"))


TREYNOR_RATIO = Spec(
    factory=treynor_ratio,
    inputs=("returns", "benchmark"),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    shape=Shape.REDUCING,
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    oracle=treynor_ratio_reference,
    conditioning=_treynor_conditioning,
    # An annualized excess return divided by a regression slope — neither scale-homogeneous nor scale-invariant
    # (tests/metrics/test_treynor_ratio.py module docstring).
    scale=ScaleExempt(
        reason="an annualized excess return divided by a regression slope — neither scale-homogeneous nor "
        "scale-invariant"
    ),
    golden_input={
        "returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018),
        "benchmark": (0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015),
    },
    golden_output=(0.3083,),
    golden_params={"risk_free_rate": 0.02},
    component_expr=_treynor_component,
    pins=(
        SpecPin(
            label="single_pair",
            inputs={"returns": (0.05,), "benchmark": (0.04,)},
            expected=(None,),
            reason="a single complete pair yields null — the regression slope needs two observations "
            "(tests/metrics/test_treynor_ratio.py::TestTreynorRatioEdge::test_single_pair)",
        ),
        SpecPin(
            label="zero_beta_is_inf",
            inputs={"returns": (3.0, 3.0, 1.0, 1.0), "benchmark": (1.0, -1.0, 1.0, -1.0)},
            expected=(math.inf,),
            reason="a zero beta (uncorrelated returns) with a positive excess return gives +inf, reported not clipped "
            "(tests/metrics/test_treynor_ratio.py::TestTreynorRatioEdge::test_zero_beta_is_inf)",
        ),
        SpecPin(
            label="constant_benchmark_is_nan_0_1",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (0.1, 0.1, 0.1)},
            expected=(math.nan,),
            reason="a constant benchmark makes the embedded beta NaN, so the excess-over-beta ratio is NaN "
            "(tests/metrics/test_treynor_ratio.py::TestTreynorRatioEdge::test_constant_benchmark_is_nan)",
        ),
        SpecPin(
            label="constant_benchmark_is_nan_1_3",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)},
            expected=(math.nan,),
            reason="the same guard at a repeating-decimal constant magnitude "
            "(tests/metrics/test_treynor_ratio.py::TestTreynorRatioEdge::test_constant_benchmark_is_nan)",
        ),
        SpecPin(
            label="constant_benchmark_is_nan_0_123456789",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (0.123456789, 0.123456789, 0.123456789)},
            expected=(math.nan,),
            reason="the same guard at a third magnitude, firing regardless of the constant "
            "(tests/metrics/test_treynor_ratio.py::TestTreynorRatioEdge::test_constant_benchmark_is_nan)",
        ),
    ),
)
