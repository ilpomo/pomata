"""Declaration for ``pomata.metrics.capture_ratio`` — reducing, up-capture over down-capture, scale-exempt."""

import math

import polars as pl

from pomata.metrics import capture_downside_ratio, capture_ratio, capture_upside_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_capture_ratio
from tests.support.declaration import Example, Golden, Pin, ScaleExempt

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
    reference='Morningstar. "Upside/Downside Capture Ratio" (methodology).',
    see_also=(
        ("capture_upside_ratio", "The numerator."),
        ("capture_downside_ratio", "The denominator."),
        ("beta", "The symmetric benchmark sensitivity whose up/down asymmetry this score summarizes."),
    ),
    bullets=(
        ("Null", "an observation is used only where both legs are present; a ``null`` in either drops that pair."),
        ("NaN", "a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``."),
        (
            "Domain",
            "the compounded (geometric) leg growth is defined only while every selected gross return "
            "``1 + r`` stays positive; a selected return at or below ``-1`` wipes that leg out of "
            "domain, so the result is a loud ``NaN`` — never a plausible wrong number.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the capture ratio (one value in ``select``, one per group "
    "under ``.over``). ``null`` when either capture ratio is undefined (no complete pairs, or "
    "a missing up- or down-market regime).",
    raises_prose="ValueError: If ``periods_per_year < 1``.",
    examples=(
        Example(
            inputs={
                "returns": (0.02, -0.01, 0.03, -0.02, 0.015, 0.005),
                "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004),
            },
            params={"periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={
                "returns": (0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.01, 0.025, -0.015, 0.008, -0.005, 0.012),
                "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004, 0.012, 0.02, -0.01, 0.006, -0.004, 0.01),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"),
            params={"periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={
                "returns": (None, 0.02, 0.03, float("nan"), 0.015, 0.005),
                "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004),
            },
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            params={"periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.02, -1.5, 0.01), "benchmark": (0.01, 0.02, -0.03)},
            intro="**Domain** — a selected gross return at or below ``-1`` on the returns leg is out of the "
            "geometric-growth domain, so the result is a loud ``NaN``:",
            params={"periods_per_year": 252},
        ),
    ),
)
