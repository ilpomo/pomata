"""
Declaration for ``pomata.metrics.common_sense_ratio`` — reducing, profit factor times tail ratio, scale-invariant.
"""

import math

import polars as pl

from pomata.metrics import common_sense_ratio, profit_factor, tail_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_common_sense_ratio
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

COMMON_SENSE_RATIO = suite_metrics(
    factory=common_sense_ratio,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_common_sense_ratio,
    recomposition=lambda: profit_factor(pl.col("returns")) * tail_ratio(pl.col("returns")),
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    golden=Golden(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, output=(2.1081,)),
    pins=(
        Pin(
            label="single_row_all_loss",
            inputs={"returns": (-0.02,)},
            expected=(0.0,),
            reason="a one-element loss has profit factor 0 and tail ratio 1, so the product is 0 ",
        ),
        Pin(
            label="no_losses_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(math.inf,),
            reason="an all-positive series has an infinite profit factor and a finite positive tail ratio, so +inf",
        ),
        Pin(
            label="all_zero_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series drives both the profit factor and the tail ratio to a 0/0, so the "
            "product is NaN — the degenerate-denominator NaN beside the +inf pin",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Tail_risk",
    see_also=(
        ("profit_factor", "The aggregate gain-to-loss factor."),
        ("tail_ratio", "The right-tail to left-tail factor."),
        ("omega_ratio", "The whole-distribution gain-to-loss ratio about a threshold."),
    ),
    bullets=(
        ("Null", "a ``null`` return is skipped; an all-null (or empty) series yields ``null``."),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "a one-element loss has a zero profit factor and a unit tail ratio, so the result is exactly ``0``.",
        ),
        (
            "Degenerate denominator",
            "it inherits the degeneracies of its two factors: ``+inf`` when there are no losses (the "
            "profit factor diverges) or a zero left tail (the tail ratio diverges), and ``NaN`` where "
            "a ``0 * inf`` arises; all reported, not clipped.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the common sense ratio (one value in ``select``, one per "
    "group under ``.over``). ``null`` when there are no returns.",
    examples=(
        Example(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, round_to=4),
        Example(
            inputs={
                "returns": (
                    0.03,
                    -0.01,
                    0.02,
                    -0.015,
                    0.01,
                    0.005,
                    -0.02,
                    0.02,
                    -0.005,
                    0.015,
                    -0.01,
                    0.025,
                    0.0,
                    -0.012,
                )
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("A", "A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B", "B"),
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.03, None, 0.02, -0.015, float("nan"), 0.005, -0.02)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            round_to=4,
        ),
        Example(
            inputs={"returns": (-0.02,)},
            intro="**Insufficient sample** — a single loss has a zero profit factor and a unit tail ratio, "
            "so the product is exactly ``0``:",
        ),
        Example(
            inputs={"returns": (0.01, 0.02, 0.03)},
            intro="**Degenerate denominator** — an all-positive series has an infinite profit factor and a "
            "finite positive tail ratio, so the product is ``+inf``:",
        ),
        Example(
            inputs={"returns": (0.0, 0.0, 0.0)},
            intro="**Degenerate denominator** — an all-zero series drives both the profit factor and the "
            "tail ratio to a ``0 / 0``, so the product is ``NaN``:",
        ),
    ),
)
