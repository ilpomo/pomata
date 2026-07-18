"""
Declaration for ``pomata.metrics.alpha`` — reducing, annualized Jensen's alpha over a benchmark baseline, scale-
exempt.
"""

import math

import polars as pl

from pomata.metrics import alpha, beta
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_alpha
from tests.support.declaration import Golden, Pin, ScaleExempt


def _alpha_component() -> pl.Expr:
    """Jensen's alpha recomposed from the public ``beta`` factory at the spec's default params (rf 0.0, 252 periods)."""
    rf_period = math.pow(1.0 + 0.0, 1.0 / 252) - 1.0
    excess = (pl.col("returns") - rf_period) - beta(pl.col("returns"), pl.col("benchmark")) * (
        pl.col("benchmark") - rf_period
    )
    return (1.0 + excess.mean()) ** 252 - 1.0


ALPHA = suite_metrics(
    factory=alpha,
    inputs=("returns", "benchmark"),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.GEOMETRIC,
    degenerate=Degenerate.ZERO_DISPERSION_IS_NAN,
    oracle=reference_alpha,
    recomposition=_alpha_component,
    scaling=ScaleExempt(
        reason="annualizes a return beyond a benchmark-explained baseline — neither scale-invariant nor homogeneous"
    ),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -2.0}, r"risk_free_rate must be >= -1"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    golden=Golden(
        inputs={
            "returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018),
            "benchmark": (0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015),
        },
        output=(-0.0903,),
        params={"risk_free_rate": 0.02},
    ),
    pins=(
        Pin(
            label="null_misalignment_drops_pair",
            inputs={"returns": (0.01, None, 0.03, -0.01, 0.02), "benchmark": (0.008, -0.01, None, -0.005, 0.018)},
            expected=(-0.47369237088902216,),
            reason="an observation with a null in either leg is dropped, matching the reference over the "
            "retained pairs",
        ),
        Pin(
            label="nan_poisons",
            inputs={"returns": (0.01, math.nan, 0.03, -0.01), "benchmark": (0.008, -0.01, 0.025, -0.005)},
            expected=(math.nan,),
            reason="a NaN in either leg of a retained pair poisons the result to NaN",
        ),
        Pin(
            label="single_pair",
            inputs={"returns": (0.05,), "benchmark": (0.04,)},
            expected=(None,),
            reason="a single complete pair yields null — the regression slope needs two observations",
        ),
        Pin(
            label="constant_benchmark_0_1",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (0.1, 0.1, 0.1)},
            expected=(math.nan,),
            reason="a constant benchmark makes the embedded beta NaN, which propagates to alpha — the "
            "exact-zero core of the near-constant regime; no conditioning filter is declared: the "
            "embedded cov/var slope matches the oracle within one ULP even at ULP-adjacent benchmark "
            "spreads (measured down to a 1e-15 spread on base 0.1)",
        ),
        Pin(
            label="constant_benchmark_one_third",
            inputs={
                "returns": (0.01, -0.02, 0.03),
                "benchmark": (0.3333333333333333, 0.3333333333333333, 0.3333333333333333),
            },
            expected=(math.nan,),
            reason="the same guard at a constant not exactly representable in float",
        ),
        Pin(
            label="constant_benchmark_many_digits",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (0.123456789, 0.123456789, 0.123456789)},
            expected=(math.nan,),
            reason="the same guard at a third, many-digit constant magnitude",
        ),
    ),
)
