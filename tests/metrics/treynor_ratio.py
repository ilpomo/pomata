"""Spec for ``pomata.metrics.treynor_ratio`` — reducing, annualized excess return per unit of beta, degree-1 at rf=0."""

import math
from collections.abc import Sequence

import polars as pl
from tests.metrics.oracles import treynor_ratio_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import beta, treynor_ratio

# Spec-local beta floor. Measured: impl and oracle agree at 1e-13..1e-16 relative deviation everywhere down to
# |beta| = 1e-5, and real disagreement starts only at |beta| ~3e-8..1e-7, so a floor of 1e-5 keeps a ~2-3 order
# margin above the crossing. The vanishing-slope quotient below is the only regime treynor's ratio genuinely needs.
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
    # Degree-1 homogeneous in a joint returns/benchmark rescale at risk_free_rate=0 (the spec's params): the linear
    # annualization mean(r) * P is degree-1 and the slope cov/var is degree-0, so the quotient rides the scale — the
    # same default-scoped axis convention as downside_deviation's "at threshold=0". A non-zero rate breaks it.
    scale=(ScaleAxis(roles=("returns", "benchmark"), degree=1),),
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
            reason="a single complete pair yields null — the regression slope needs two observations",
        ),
        SpecPin(
            label="zero_beta_is_inf",
            inputs={"returns": (3.0, 3.0, 1.0, 1.0), "benchmark": (1.0, -1.0, 1.0, -1.0)},
            expected=(math.inf,),
            reason="a zero beta (uncorrelated returns) with a positive excess return gives +inf, reported not "
            "clipped — the exact core of the vanishing-slope regime the conditioning filter excludes from the "
            "property tiers",
            covers_conditioning=True,
        ),
        SpecPin(
            label="small_beta_matches_reference",
            inputs={"returns": (3.001, 2.999, 1.001, 0.999), "benchmark": (1.0, -1.0, 1.0, -1.0)},
            expected=(504000.0000000275,),
            reason="an exact-by-construction beta of 0.001, well inside the small-slope band the 1e-5 conditioning "
            "floor admits: the quotient matches the oracle to ~5.6e-14 relative deviation, holding the fact that a "
            "tiny-but-clean beta stays well-conditioned",
        ),
        SpecPin(
            label="constant_benchmark_is_nan_0_1",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (0.1, 0.1, 0.1)},
            expected=(math.nan,),
            reason="a constant benchmark makes the embedded beta NaN, so the excess-over-beta ratio is NaN",
        ),
        SpecPin(
            label="constant_benchmark_is_nan_1_3",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)},
            expected=(math.nan,),
            reason="the same guard at a repeating-decimal constant magnitude",
        ),
        SpecPin(
            label="constant_benchmark_is_nan_0_123456789",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (0.123456789, 0.123456789, 0.123456789)},
            expected=(math.nan,),
            reason="the same guard at a third magnitude, firing regardless of the constant",
        ),
    ),
)
