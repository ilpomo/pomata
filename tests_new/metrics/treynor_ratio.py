"""Spec for ``pomata.metrics.treynor_ratio`` — reducing, annualized excess return per unit of beta, scale-exempt."""

import math
from collections.abc import Sequence

import polars as pl
from tests_new.metrics.oracles import treynor_ratio_reference
from tests_new.support.spec import ScaleExempt, Shape, Spec, SpecPin

from pomata.metrics import beta, treynor_ratio

# Spec-local beta floor: the old 5e-2 cut sat ~6 orders of magnitude above the measured impl-vs-oracle crossing
# (|beta| ~3e-8..1e-7), with agreement at 1e-13..1e-16 relative deviation everywhere down to 1e-5, so 1e-5 keeps a
# ~2-3 order margin above the crossing while re-admitting the needlessly excluded band. The old filter's two
# well-spread clauses (returns leg, benchmark leg) were the same mal-applied grade-1 guards removed from alpha /
# beta / sharpe_ratio — the vanishing-slope quotient below is the only regime treynor's ratio genuinely needs.
_BETA_FLOOR = 1e-5


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
    """A beta bounded away from zero — the one regime treynor's quotient genuinely needs."""
    return _beta_nondegenerate(frame["returns"].to_list(), frame["benchmark"].to_list())


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
            reason="a zero beta (uncorrelated returns) with a positive excess return gives +inf, reported not "
            "clipped — the exact core of the vanishing-slope regime the conditioning filter excludes from the "
            "property tiers (tests/metrics/test_treynor_ratio.py::TestTreynorRatioEdge::test_zero_beta_is_inf)",
            covers_conditioning=True,
        ),
        SpecPin(
            label="small_beta_matches_reference",
            inputs={"returns": (3.001, 2.999, 1.001, 0.999), "benchmark": (1.0, -1.0, 1.0, -1.0)},
            expected=(504000.0000000275,),
            reason="an exact-by-construction beta of 0.001 — 50x inside the band the old 5e-2 floor excluded: the "
            "quotient still matches the oracle to ~5.6e-14 relative deviation, the measured fact that justified "
            "narrowing the floor to 1e-5",
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
