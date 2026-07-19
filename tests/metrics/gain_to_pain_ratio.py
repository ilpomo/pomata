"""Declaration for ``pomata.metrics.gain_to_pain_ratio`` — reducing, net return over total loss, scale-invariant."""

import math

from pomata.metrics import gain_to_pain_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_gain_to_pain_ratio
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

GAIN_TO_PAIN_RATIO = suite_metrics(
    factory=gain_to_pain_ratio,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_gain_to_pain_ratio,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    golden=Golden(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, output=(0.4444,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.02,)},
            expected=(math.inf,),
            reason="a one-element positive series has no loss, so the ratio is +inf ",
        ),
        Pin(
            label="no_losses_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(math.inf,),
            reason="an all-positive series has no loss, so the ratio is +inf ",
        ),
        Pin(
            label="all_negative_is_minus_one",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(-1.0,),
            reason="an all-negative series has net loss equal to its total loss, so the ratio is -1 ",
        ),
        Pin(
            label="all_zero_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series has zero total loss and zero net return, so the ratio is a 0/0, i.e. "
            "NaN — the degenerate-denominator NaN beside the +inf pin",
        ),
    ),
    reference="Schwager, J. D. (2012). *Hedge Fund Market Wizards*. Wiley.",
    see_also=(
        ("profit_factor", "The gross-gain to gross-loss counterpart."),
        ("omega_ratio", "The probability-weighted gain-to-loss ratio about a threshold."),
        ("ulcer_performance_ratio", "A drawdown-based return-to-pain ratio."),
    ),
    note_extension="\n\n"
    "It is computed on the return series as given, with no calendar resampling and no "
    "risk-free adjustment (the pure Schwager ratio).",
    bullets=(
        ("Null", "a ``null`` return is skipped; an all-null (or empty) series yields ``null``."),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "a single positive observation has no offsetting loss, so the result is ``+inf`` — reported, not clipped.",
        ),
        (
            "Degenerate denominator",
            "with no negative returns the total loss is zero, so the ratio is ``+inf`` (or ``NaN`` "
            "when the net return is also zero) — reported, not clipped.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the gain to pain ratio (one value in ``select``, one per "
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
            inputs={"returns": (0.02,)},
            intro="**Insufficient sample** — a single positive return has no offsetting loss, so the ratio "
            "is ``+inf``:",
        ),
        Example(
            inputs={"returns": (0.01, 0.02, 0.03)},
            intro="**Degenerate denominator** — an all-positive series has no loss, so the ratio is ``+inf``:",
        ),
        Example(
            inputs={"returns": (0.0, 0.0, 0.0)},
            intro="**Degenerate denominator** — an all-zero series has zero total loss and zero net return, "
            "so the ratio is a ``0 / 0``, i.e. ``NaN``:",
        ),
    ),
)
