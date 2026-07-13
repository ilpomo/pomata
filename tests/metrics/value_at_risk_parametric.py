"""Spec for ``pomata.metrics.value_at_risk_parametric`` — reducing, mean plus z times std, degree-1 homogeneous."""

from tests.metrics.oracles import value_at_risk_parametric_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import value_at_risk_parametric

VALUE_AT_RISK_PARAMETRIC = Spec(
    factory=value_at_risk_parametric,
    inputs=("returns",),
    params={"confidence": 0.95},
    shape=Shape.REDUCING,
    raises=(
        ({"confidence": 0.0}, r"confidence must be in the open interval"),
        ({"confidence": 1.0}, r"confidence must be in the open interval"),
        ({"confidence": -0.1}, r"confidence must be in the open interval"),
        ({"confidence": 1.5}, r"confidence must be in the open interval"),
    ),
    oracle=value_at_risk_parametric_reference,
    # mean + z*std is exactly degree-1 homogeneous
    scale=(ScaleAxis(roles=("returns",), degree=1),),
    golden_input={"returns": (0.02, -0.04, 0.01, -0.06, 0.03)},
    golden_output=(-0.0732,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(None,),
            reason="one observation has no sample standard deviation (ddof=1 needs two), so the result is null ",
        ),
        SpecPin(
            label="constant_is_mean",
            inputs={"returns": (0.01, 0.01, 0.01)},
            expected=(0.01,),
            reason="a constant series has zero dispersion, so z*std=0 and the value-at-risk is the mean itself ",
        ),
    ),
)
