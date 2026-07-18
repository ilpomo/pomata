"""
Declaration for ``pomata.metrics.value_at_risk_rolling`` — the rolling historical return quantile, degree-1
homogeneous.
"""

from pomata.metrics import value_at_risk_rolling
from tests.metrics.enums import BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_value_at_risk_rolling
from tests.metrics.value_at_risk import VALUE_AT_RISK
from tests.support.declaration import Golden, Pin, ScaleAxis

VALUE_AT_RISK_ROLLING = suite_metrics(
    factory=value_at_risk_rolling,
    inputs=("returns",),
    params={"window": 4, "confidence": 0.95},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    rolling_of=VALUE_AT_RISK,
    window="window",
    warmup=3,
    oracle=reference_value_at_risk_rolling,
    scaling=(ScaleAxis(roles=("returns",), degree=1),),
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"confidence": 0.0}, r"confidence must be in the open interval"),
        ({"confidence": 1.0}, r"confidence must be in the open interval"),
        ({"confidence": -0.1}, r"confidence must be in the open interval"),
        ({"confidence": 1.5}, r"confidence must be in the open interval"),
    ),
    golden=Golden(
        inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)},
        output=(None, None, None, -0.0185, -0.0185, -0.0085, -0.0142),
    ),
    pins=(
        Pin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(None, None, None, None, -0.018),
            reason="when the window equals the series length only the last row is defined ",
            params_override={"window": 5},
        ),
        Pin(
            label="sign_convention_is_signed_quantile",
            inputs={"returns": (-0.05, -0.04, -0.03, -0.02, -0.01)},
            expected=(None, None, -0.049, -0.039, -0.028999999999999998),
            reason="an all-loss series yields a strictly negative rolling VaR (the signed return quantile) ",
            params_override={"window": 3},
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Value_at_risk",
    see_also=(
        ("value_at_risk", "The whole-series reducing form."),
        ("tail_ratio_rolling", "Another rolling tail-risk measure."),
        ("downside_deviation_rolling", "Another rolling downside-risk measure."),
    ),
    notes=(
        (
            "Sign convention",
            "Returned as the signed return quantile (negative for a loss), not a positive loss "
            "magnitude; negate it if a positive figure is wanted.",
        ),
    ),
    opener_override="Each window matches an independent reference oracle (the reducing :func:`value_at_risk` "
    "recomputed over the window).",
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The rolling value-at-risk for each row, the same length as the input. The first ``window "
    "- 1`` rows are ``null`` (warm-up): the window must hold ``window`` non-null values "
    "before a result is emitted.",
    raises_prose="ValueError: If ``window < 1``, or if ``confidence`` is not in the open interval ``(0, 1)``.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 1``.",
        "confidence": "The tail confidence level (canonically ``0.95``); the quantile taken is ``1 - "
        "confidence``. Must be in the open interval ``(0, 1)``.",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up on its own "
    "(the ``B`` group never borrows ``A``'s tail):",
    intro_missing="A leading ``null`` and a later ``NaN`` show the per-window masking, with the result "
    "recovering once both leave the window:",
)
