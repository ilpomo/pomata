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
)
