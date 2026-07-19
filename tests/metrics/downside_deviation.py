"""
Declaration for ``pomata.metrics.downside_deviation`` — reducing, the annualized RMS of shortfall, degree-1
homogeneous.
"""

import math

from pomata.metrics import downside_deviation
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_downside_deviation
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

DOWNSIDE_DEVIATION = suite_metrics(
    factory=downside_deviation,
    inputs=("returns",),
    params={"periods_per_year": 252},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.SQRT_TIME,
    degenerate=Degenerate.EXACT_ZERO,
    oracle=reference_downside_deviation,
    scaling=(ScaleAxis(roles=("returns",), degree=1),),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"threshold": math.nan}, r"threshold must be a finite number"),
        ({"threshold": math.inf}, r"threshold must be a finite number"),
        ({"threshold": -math.inf}, r"threshold must be a finite number"),
    ),
    golden=Golden(inputs={"returns": (0.02, -0.04, 0.01, -0.06, 0.03)}, output=(0.5119,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (-0.02,)},
            expected=(0.3174901573277509,),
            reason="a one-element downside series annualizes its shortfall RMS ",
        ),
        Pin(
            label="no_downside_is_zero",
            inputs={"returns": (0.01, 0.02, 0.0, 0.03)},
            expected=(0.0,),
            reason="returns all at or above the threshold have zero downside, so the deviation is 0 ",
        ),
        Pin(
            label="threshold_nonzero",
            inputs={"returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018)},
            expected=(0.24936118382779626,),
            reason="a non-zero threshold shifts the shortfall ",
            params_override={"threshold": 0.01},
        ),
    ),
    reference='Sortino, F. A. & Price, L. N. (1994). "Performance Measurement in a Downside Risk '
    'Framework." *The Journal of Investing*, 3(3), 59-64.',
    doi="https://doi.org/10.3905/joi.3.3.59",
    wikipedia="https://en.wikipedia.org/wiki/Downside_risk",
    see_also=(
        ("sortino_ratio", "The risk-adjusted return that divides excess return by this."),
        ("volatility", "The symmetric (two-sided) dispersion."),
        ("downside_deviation_rolling", "The rolling (windowed) form."),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` return is skipped (excluded from the mean); an all-null (or empty) series yields ``null``.",
        ),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Degenerate denominator",
            "when every return is at or above the threshold the shortfall is all zero, so the result is exactly ``0``.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the annualized downside deviation (one value in ``select``, "
    "one per group under ``.over``). ``null`` when there are no returns.",
    raises_prose="ValueError: If ``periods_per_year < 1``, or if ``threshold`` is not finite.",
    args_prose={
        "threshold": "The **per-period** return level separating gains from losses / the minimum acceptable "
        "return (default ``0.0``); an annual target must be de-annualized by the caller before it "
        "is passed. Must be finite.",
    },
    examples=(
        Example(inputs={"returns": (0.02, -0.04, 0.01, -0.06, 0.03)}, params={"periods_per_year": 252}, round_to=4),
        Example(
            inputs={"returns": (0.02, -0.04, 0.01, -0.06, 0.03, 0.01, -0.02, 0.04, -0.03, 0.02)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("A", "A", "A", "A", "A", "B", "B", "B", "B", "B"),
            params={"periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.02, None, -0.04, 0.01, float("nan"), -0.06, 0.03)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            params={"periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.01, 0.02, 0.0, 0.03)},
            intro="**Degenerate denominator** — returns all at or above the threshold have zero downside, "
            "so the deviation is ``0``:",
            params={"periods_per_year": 252},
        ),
    ),
)
