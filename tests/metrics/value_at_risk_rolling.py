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
)
