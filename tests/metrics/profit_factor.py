"""Declaration for ``pomata.metrics.profit_factor`` — reducing, gross gains over gross losses, scale-invariant."""

import math

from pomata.metrics import profit_factor
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_profit_factor
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

PROFIT_FACTOR = suite_metrics(
    factory=profit_factor,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_profit_factor,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    golden=Golden(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, output=(1.4444,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(math.inf,),
            reason="a single gain has zero gross loss, so the factor is +inf",
        ),
        Pin(
            label="no_losses_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(math.inf,),
            reason="an all-positive series has no losses, so the ratio is +inf ",
        ),
        Pin(
            label="no_gains_is_zero",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(0.0,),
            reason="an all-negative series has no gains, so the ratio is 0 ",
        ),
        Pin(
            label="all_zero_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series has zero gains and losses, so the ratio is 0/0, i.e. NaN ",
        ),
    ),
    reference="Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies* (2nd ed.). Wiley.",
    see_also=(
        ("payoff_ratio", "The average-win to average-loss counterpart."),
        ("omega_ratio", "The same ratio generalized to an arbitrary threshold."),
        ("common_sense_ratio", "Scales this profit factor by the tail ratio."),
    ),
    note_extension="\n\n"
    "This is a **bar-level** statistic: each return observation is treated as one gain or "
    "loss. It is not a per-trade statistic -- true per-trade profit factor needs trade-level "
    "fill data, which is outside this toolkit's scope.",
    bullets=(
        ("Null", "a ``null`` return is skipped; an all-null (or empty) series yields ``null``."),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "a single gain has no offsetting loss, so the result is ``+inf`` — reported, not clipped.",
        ),
        (
            "Degenerate denominator",
            "with no negative returns the total loss is zero, so the ratio is ``+inf`` (or ``NaN`` "
            "when an all-``0`` series also has zero gross gain, a ``0 / 0``) — reported, not clipped.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the profit factor (one value in ``select``, one per group "
    "under ``.over``). ``null`` when there are no returns.",
    examples=(
        Example(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, round_to=4),
        Example(
            inputs={
                "returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02, 0.04, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03)
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("A", "A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B", "B"),
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.03, None, -0.01, 0.02, float("nan"), -0.015, 0.01)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.05,)},
            intro="**Insufficient sample** — a single gain has zero gross loss, so the factor is ``+inf``:",
        ),
        Example(
            inputs={"returns": (0.01, 0.02, 0.03)},
            intro="**Degenerate denominator** — an all-positive series has no losses, so the ratio is ``+inf``:",
        ),
        Example(
            inputs={"returns": (-0.01, -0.02, -0.03)},
            intro="**Degenerate denominator** — an all-negative series has no gains, so the ratio is ``0``:",
        ),
        Example(
            inputs={"returns": (0.0, 0.0, 0.0)},
            intro="**Degenerate denominator** — an all-zero series has zero gains and losses, so the ratio "
            "is ``0/0``, i.e. ``NaN``:",
        ),
    ),
)
