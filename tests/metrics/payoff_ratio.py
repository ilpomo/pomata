"""
Declaration for ``pomata.metrics.payoff_ratio`` — reducing, average win over average loss magnitude, scale-
invariant.
"""

from pomata.metrics import payoff_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_payoff_ratio
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

PAYOFF_RATIO = suite_metrics(
    factory=payoff_ratio,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_payoff_ratio,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    golden=Golden(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, output=(1.0833,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(None,),
            reason="a one-element series leaves one side empty, so the ratio is null ",
        ),
        Pin(
            label="no_losses_is_null",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(None,),
            reason="an all-positive series has no losing side, so the ratio is null ",
        ),
        Pin(
            label="no_gains_is_null",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(None,),
            reason="an all-negative series has no winning side, so the ratio is null ",
        ),
    ),
    reference="Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies* (2nd ed.). Wiley.",
    see_also=(
        ("win_rate", "The companion frequency (how often returns win)."),
        ("profit_factor", "The aggregate (total-gain to total-loss) counterpart."),
        ("kelly_criterion", "The growth-optimal fraction built from this and the win rate."),
    ),
    notes=(("Zero return", "A return of exactly ``0`` is neither a win nor a loss and is excluded from both means."),),
    note_extension="\n\n"
    "This is a **bar-level** statistic: each return observation is treated as one win or "
    "loss. It is not a per-trade statistic -- true per-trade payoff needs trade-level fill "
    "data, which is outside this toolkit's scope.",
    bullets=(
        ("Null", "a ``null`` return is skipped; an all-null (or empty) series yields ``null``."),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        ("Insufficient sample", "a one-element series leaves one side of the ratio empty, so the result is ``null``."),
        (
            "Degenerate denominator",
            "with no winning (or no losing) returns one side of the ratio is undefined, so the result is ``null``.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the payoff ratio (one value in ``select``, one per group "
    "under ``.over``). ``null`` when there are no winning returns or no losing returns (one "
    "side of the ratio is undefined).",
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
            intro="**Insufficient sample** — a one-element series leaves one side empty, so the ratio is ``null``:",
        ),
        Example(
            inputs={"returns": (0.01, 0.02, 0.03)},
            intro="**Degenerate denominator** — an all-positive series has no losing side, so the ratio is ``null``:",
        ),
    ),
)
