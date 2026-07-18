"""
Declaration for ``pomata.metrics.value_at_risk`` — reducing, the historical return quantile at a confidence,
degree-1.
"""

from pomata.metrics import value_at_risk
from tests_new.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests_new.metrics.harness import suite_metrics
from tests_new.metrics.oracles import reference_value_at_risk
from tests_new.support.declaration import Golden, Pin, ScaleAxis

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
)
