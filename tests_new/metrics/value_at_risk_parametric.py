"""
Declaration for ``pomata.metrics.value_at_risk_parametric`` — reducing, mean plus z times std, degree-1 homogeneous.
"""

from pomata.metrics import value_at_risk_parametric
from tests_new.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests_new.metrics.harness import suite_metrics
from tests_new.metrics.oracles import reference_value_at_risk_parametric
from tests_new.support.declaration import Golden, Pin, ScaleAxis

VALUE_AT_RISK_PARAMETRIC = suite_metrics(
    factory=value_at_risk_parametric,
    inputs=("returns",),
    params={"confidence": 0.95},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    oracle=reference_value_at_risk_parametric,
    scaling=(ScaleAxis(roles=("returns",), degree=1),),
    raises=(
        ({"confidence": 0.0}, r"confidence must be in the open interval"),
        ({"confidence": 1.0}, r"confidence must be in the open interval"),
        ({"confidence": -0.1}, r"confidence must be in the open interval"),
        ({"confidence": 1.5}, r"confidence must be in the open interval"),
    ),
    golden=Golden(inputs={"returns": (0.02, -0.04, 0.01, -0.06, 0.03)}, output=(-0.0732,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(None,),
            reason="one observation has no sample standard deviation (ddof=1 needs two), so the result is null",
        ),
        Pin(
            label="constant_is_mean",
            inputs={"returns": (0.01, 0.01, 0.01)},
            expected=(0.01,),
            reason="a constant series has zero dispersion, so z*std=0 and the value-at-risk is the mean itself",
        ),
    ),
)
