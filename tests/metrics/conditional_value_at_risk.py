"""Spec for ``pomata.metrics.conditional_value_at_risk`` — reducing, the mean of the worst-tail returns, degree-1
homogeneous.
"""

from tests.metrics.oracles import conditional_value_at_risk_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import conditional_value_at_risk

CONDITIONAL_VALUE_AT_RISK = Spec(
    factory=conditional_value_at_risk,
    inputs=("returns",),
    params={"confidence": 0.95},
    shape=Shape.REDUCING,
    raises=(
        ({"confidence": 0.0}, r"confidence must be in the open interval"),
        ({"confidence": 1.0}, r"confidence must be in the open interval"),
        ({"confidence": -0.1}, r"confidence must be in the open interval"),
        ({"confidence": 1.5}, r"confidence must be in the open interval"),
    ),
    oracle=conditional_value_at_risk_reference,
    # A tail mean of returns scales linearly
    scale=(ScaleAxis(roles=("returns",), degree=1),),
    golden_input={"returns": (0.03, -0.05, 0.02, -0.08, 0.01, -0.06, 0.04, -0.02)},
    golden_output=(-0.07,),
    golden_params={"confidence": 0.75},
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (-0.02,)},
            expected=(-0.02,),
            reason="for a single observation the whole shortfall slice is that element ",
        ),
        SpecPin(
            label="fractional_weight_golden",
            inputs={"returns": (-0.10, -0.06, 0.0, 0.05, 0.10)},
            expected=(-0.08666666666666666,),
            reason="the Rockafellar-Uryasev fractional boundary weight at confidence=0.7 averages the worst in full "
            "and the second-worst at weight 0.5 "
            "",
            params_override={"confidence": 0.7},
        ),
    ),
)
