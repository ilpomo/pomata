"""Declaration for ``pomata.metrics.tail_ratio`` — reducing, the right-tail quantile over the left-tail magnitude."""

import math

from pomata.metrics import tail_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_tail_ratio
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

TAIL_RATIO = suite_metrics(
    factory=tail_ratio,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_tail_ratio,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    golden=Golden(inputs={"returns": (0.02, -0.04, 0.01, -0.06, 0.03)}, output=(0.5,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(1.0,),
            reason="a one-element series has equal tails, so the ratio is 1.0",
        ),
        Pin(
            label="constant_is_one",
            inputs={"returns": (0.01, 0.01, 0.01)},
            expected=(1.0,),
            reason="a constant series has equal 5th/95th percentiles, so the ratio is 1.0 ",
        ),
        Pin(
            label="zero_left_tail_is_inf",
            inputs={"returns": (0.0, 0.0, 0.0, 0.0, 0.02)},
            expected=(math.inf,),
            reason="a zero 5th-percentile against a non-zero 95th gives +inf ",
        ),
        Pin(
            label="all_zero_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series gives 0/0 at both tails, so the ratio is NaN ",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Tail_risk",
    see_also=(
        ("tail_ratio_rolling", "The rolling (windowed) form."),
        ("common_sense_ratio", "Scales the profit factor by this tail ratio."),
        ("skewness", "The moment-based companion measure of distributional asymmetry."),
    ),
    bullets=(
        ("Null", "a ``null`` return is skipped; an all-null (or empty) series yields ``null``."),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Degenerate denominator",
            "a constant series has equal 5th and 95th percentiles, so the ratio is ``1.0`` (an "
            "all-``0`` series is the ``0 / 0`` exception, yielding ``NaN``); when the 5th-percentile "
            "return is exactly ``0`` against a non-zero 95th the ratio is ``+inf`` — reported, not "
            "clipped, following IEEE division.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the tail ratio (one value in ``select``, one per group under "
    "``.over``). ``null`` when there are no returns.",
    examples=(
        Example(inputs={"returns": (0.02, -0.04, 0.01, -0.06, 0.03)}, round_to=4),
        Example(
            inputs={"returns": (0.02, -0.04, 0.01, -0.06, 0.03, 0.05, -0.02, 0.04, -0.03, 0.02)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("AAPL",) * 5 + ("NVDA",) * 5,
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.02, None, -0.04, 0.01, float("nan"), -0.06, 0.03)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.0, 0.0, 0.0, 0.0, 0.02)},
            intro="**Degenerate denominator** — a zero 5th-percentile against a non-zero 95th gives ``+inf``:",
        ),
        Example(
            inputs={"returns": (0.0, 0.0, 0.0)},
            intro="**Degenerate denominator** — an all-zero series gives ``0/0`` at both tails, so the "
            "ratio is ``NaN``:",
        ),
    ),
)
