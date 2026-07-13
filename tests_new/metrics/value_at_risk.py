"""Spec for ``pomata.metrics.value_at_risk`` — reducing, the historical return quantile at a confidence, degree-1
homogeneous.
"""

from tests.metrics.oracles import value_at_risk_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import value_at_risk

VALUE_AT_RISK = Spec(
    factory=value_at_risk,
    inputs=("returns",),
    params={"confidence": 0.95},
    shape=Shape.REDUCING,
    raises=(
        ({"confidence": 0.0}, r"confidence must be in the open interval"),
        ({"confidence": 1.0}, r"confidence must be in the open interval"),
        ({"confidence": -0.1}, r"confidence must be in the open interval"),
        ({"confidence": 1.5}, r"confidence must be in the open interval"),
    ),
    oracle=value_at_risk_reference,
    # A historical quantile scales linearly (test_value_at_risk.py::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("returns",), degree=1),),
    golden_input={"returns": (0.02, -0.04, 0.01, -0.06, 0.03)},
    golden_output=(-0.056,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (-0.02,)},
            expected=(-0.02,),
            reason="every quantile of a single value is that value (test_value_at_risk.py::test_single_row)",
        ),
    ),
)
