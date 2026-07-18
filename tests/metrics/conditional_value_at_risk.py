"""
Declaration for ``pomata.metrics.conditional_value_at_risk`` — reducing, the mean of the worst-tail returns,
degree-1.
"""

from pomata.metrics import conditional_value_at_risk
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_conditional_value_at_risk
from tests.support.declaration import Golden, Pin, ScaleAxis

CONDITIONAL_VALUE_AT_RISK = suite_metrics(
    factory=conditional_value_at_risk,
    inputs=("returns",),
    params={"confidence": 0.95},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    oracle=reference_conditional_value_at_risk,
    scaling=(ScaleAxis(roles=("returns",), degree=1),),
    raises=(
        ({"confidence": 0.0}, r"confidence must be in the open interval"),
        ({"confidence": 1.0}, r"confidence must be in the open interval"),
        ({"confidence": -0.1}, r"confidence must be in the open interval"),
        ({"confidence": 1.5}, r"confidence must be in the open interval"),
    ),
    golden=Golden(
        inputs={"returns": (0.03, -0.05, 0.02, -0.08, 0.01, -0.06, 0.04, -0.02)},
        output=(-0.07,),
        params={"confidence": 0.75},
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (-0.02,)},
            expected=(-0.02,),
            reason="for a single observation the whole shortfall slice is that element ",
        ),
        Pin(
            label="fractional_weight_golden",
            inputs={"returns": (-0.1, -0.06, 0.0, 0.05, 0.1)},
            expected=(-0.08666666666666666,),
            reason="the Rockafellar-Uryasev fractional boundary weight at confidence=0.7 averages the worst "
            "in full and the second-worst at weight 0.5",
            params_override={"confidence": 0.7},
        ),
    ),
    reference='Rockafellar, R. T. & Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk." '
    "*Journal of Risk*, 2(3), 21-41.",
    doi="https://doi.org/10.21314/JOR.2000.038",
    wikipedia="https://en.wikipedia.org/wiki/Expected_shortfall",
    see_also=(
        ("value_at_risk", "The tail cutoff quantile; this coherent average of the tail is always at least as deep."),
        ("conditional_drawdown_at_risk", "The same tail-averaging applied to the drawdown curve."),
        ("value_at_risk_parametric", "The parametric cutoff this estimate averages the tail beyond."),
    ),
    notes=(
        (
            "Historical, not parametric",
            "The shortfall is taken over the empirical return distribution, with no normality or "
            "other distributional assumption.",
        ),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` return is skipped (excluded from the count ``n`` and the tail average); an "
            "all-null (or empty) series yields ``null``.",
        ),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the expected shortfall (one value in ``select``, one per "
    "group under ``.over``). ``null`` when there are no returns.",
    raises_prose="ValueError: If ``confidence`` is not in the open interval ``(0, 1)``.",
    args_prose={
        "confidence": "The tail confidence level (canonically ``0.95``); the shortfall is averaged over the "
        "worst ``1 - confidence`` of returns. Must be in the open interval ``(0, 1)``.",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
    intro_missing="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
    "handling visible:",
)
