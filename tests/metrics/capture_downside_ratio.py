"""Declaration for ``pomata.metrics.capture_downside_ratio`` — reducing, down-market capture, scale-exempt."""

import math

from pomata.metrics import capture_downside_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_capture_downside_ratio
from tests.support.declaration import Example, Golden, Pin, ScaleExempt

CAPTURE_DOWNSIDE_RATIO = suite_metrics(
    factory=capture_downside_ratio,
    inputs=("returns", "benchmark"),
    params={"periods_per_year": 252},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.GEOMETRIC,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_capture_downside_ratio,
    scaling=ScaleExempt(reason="a ratio of two annualized geometric returns — neither scale-invariant nor homogeneous"),
    raises=(({"periods_per_year": 0}, r"periods_per_year must be >= 1"),),
    golden=Golden(
        inputs={
            "returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018),
            "benchmark": (0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015),
        },
        output=(1.0224,),
    ),
    pins=(
        Pin(
            label="no_down_market_is_null",
            inputs={"returns": (0.01, 0.02, 0.03), "benchmark": (0.01, 0.02, 0.03)},
            expected=(None,),
            reason="with no negative-benchmark period the ratio is undefined, so the result is null ",
        ),
        Pin(
            label="return_leg_wiped_out_is_nan",
            inputs={"returns": (0.02, -1.5, 0.01), "benchmark": (-0.01, -0.02, -0.03)},
            expected=(math.nan,),
            reason="a selected portfolio return <= -1 wipes that leg out of the geometric-growth domain, a loud NaN",
        ),
        Pin(
            label="benchmark_leg_wiped_out_is_nan",
            inputs={"returns": (0.02, -0.03, 0.01), "benchmark": (-0.01, -1.2, -0.03)},
            expected=(math.nan,),
            reason="a selected benchmark value <= -1 wipes that leg out of the geometric-growth domain, a loud NaN",
        ),
    ),
    reference='Morningstar. "Upside/Downside Capture Ratio" (methodology).',
    see_also=(
        ("capture_upside_ratio", "The up-market counterpart."),
        ("capture_ratio", "Their ratio, an overall asymmetry measure."),
        ("beta", "The symmetric benchmark sensitivity this asymmetric down-market measure refines."),
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
    returns_body="A single ``Float64`` value: the downside capture ratio (one value in ``select``, one per "
    "group under ``.over``). ``null`` when there are no complete pairs or no down-market "
    "periods.",
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
            inputs={"returns": (0.02, -1.5, 0.01), "benchmark": (-0.01, -0.02, -0.03)},
            intro="**Domain** — a selected portfolio return at or below ``-1`` wipes that leg out of the "
            "geometric-growth domain, so the result is a loud ``NaN``:",
            params={"periods_per_year": 252},
        ),
    ),
)
