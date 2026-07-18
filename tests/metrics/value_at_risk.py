"""
Declaration for ``pomata.metrics.value_at_risk`` — reducing, the historical return quantile at a confidence,
degree-1.
"""

from pomata.metrics import value_at_risk
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_value_at_risk
from tests.support.declaration import Golden, Pin, ScaleAxis

VALUE_AT_RISK = suite_metrics(
    factory=value_at_risk,
    inputs=("returns",),
    params={"confidence": 0.95},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    oracle=reference_value_at_risk,
    scaling=(ScaleAxis(roles=("returns",), degree=1),),
    raises=(
        ({"confidence": 0.0}, r"confidence must be in the open interval"),
        ({"confidence": 1.0}, r"confidence must be in the open interval"),
        ({"confidence": -0.1}, r"confidence must be in the open interval"),
        ({"confidence": 1.5}, r"confidence must be in the open interval"),
    ),
    golden=Golden(inputs={"returns": (0.02, -0.04, 0.01, -0.06, 0.03)}, output=(-0.056,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (-0.02,)},
            expected=(-0.02,),
            reason="every quantile of a single value is that value",
        ),
    ),
    reference="J.P. Morgan / Reuters (1996). *RiskMetrics — Technical Document* (4th ed.).",
    wikipedia="https://en.wikipedia.org/wiki/Value_at_risk",
    see_also=(
        ("conditional_value_at_risk", "The mean loss beyond this threshold (expected shortfall)."),
        ("value_at_risk_parametric", "The Gaussian (parametric) estimate of the same quantile."),
        ("value_at_risk_modified", "The skewness/kurtosis-corrected (Cornish-Fisher) estimate."),
        ("value_at_risk_rolling", "The rolling (windowed) form."),
    ),
    notes=(
        (
            "Sign convention",
            "Returned as the signed return quantile (negative for a loss), not a positive loss "
            "magnitude; negate it if a positive figure is wanted.",
        ),
        (
            "Historical, not parametric",
            "The quantile is taken over the empirical return distribution, with no normality or other "
            "distributional assumption.",
        ),
    ),
    bullets=(
        ("Null", "a ``null`` return is skipped; an all-null (or empty) series yields ``null``."),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the historical value-at-risk (one value in ``select``, one "
    "per group under ``.over``). ``null`` when there are no returns.",
    raises_prose="ValueError: If ``confidence`` is not in the open interval ``(0, 1)``.",
    args_prose={
        "confidence": "The tail confidence level (canonically ``0.95``); the quantile taken is ``1 - "
        "confidence``. Must be in the open interval ``(0, 1)``.",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
    intro_missing="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
    "handling visible:",
)
