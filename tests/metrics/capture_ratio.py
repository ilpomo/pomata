"""Declaration for ``pomata.metrics.capture_ratio`` — reducing, up-capture over down-capture, scale-exempt."""

import math

import polars as pl

from pomata.metrics import capture_downside_ratio, capture_ratio, capture_upside_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_capture_ratio
from tests.support.declaration import Golden, Pin, ScaleExempt

CAPTURE_RATIO = suite_metrics(
    factory=capture_ratio,
    inputs=("returns", "benchmark"),
    params={"periods_per_year": 252},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.GEOMETRIC,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_capture_ratio,
    recomposition=lambda: (
        capture_upside_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252)
        / capture_downside_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252)
    ),
    scaling=ScaleExempt(reason="a ratio of two capture ratios — neither scale-homogeneous nor scale-invariant"),
    raises=(({"periods_per_year": 0}, r"periods_per_year must be >= 1"),),
    golden=Golden(
        inputs={
            "returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018),
            "benchmark": (0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015),
        },
        output=(1.3479,),
    ),
    pins=(
        Pin(
            label="missing_regime_is_null",
            inputs={"returns": (0.01, 0.02, 0.03), "benchmark": (0.01, 0.02, 0.03)},
            expected=(None,),
            reason="with no down-market period the downside capture is undefined, so the ratio is null ",
        ),
        Pin(
            label="return_below_negative_one_is_nan_returns_leg",
            inputs={"returns": (0.02, -1.5, 0.01), "benchmark": (0.01, 0.02, -0.03)},
            expected=(math.nan,),
            reason="a selected gross return <= -1 (returns leg) is out of the geometric-growth domain, a loud NaN",
        ),
        Pin(
            label="return_below_negative_one_is_nan_benchmark_leg",
            inputs={"returns": (0.02, -0.03, 0.01), "benchmark": (0.01, -1.2, 0.03)},
            expected=(math.nan,),
            reason="the same domain-boundary fact carried by the benchmark leg ",
        ),
    ),
)
