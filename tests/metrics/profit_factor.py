"""Spec for ``pomata.metrics.profit_factor`` — reducing, gross gains over gross losses, scale-invariant."""

import math

from tests.metrics.oracles import profit_factor_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import profit_factor

PROFIT_FACTOR = Spec(
    factory=profit_factor,
    inputs=("returns",),
    params={},
    shape=Shape.REDUCING,
    oracle=profit_factor_reference,
    # A ratio of two sums of the same input is scale-invariant (test_profit_factor.py::test_scale_invariance).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)},
    golden_output=(1.4444,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(math.inf,),
            reason="a single gain has zero gross loss, so the factor is +inf (test_profit_factor.py::test_single_row)",
        ),
        SpecPin(
            label="no_losses_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(math.inf,),
            reason="an all-positive series has no losses, so the ratio is +inf "
            "(test_profit_factor.py::test_no_losses_is_inf)",
        ),
        SpecPin(
            label="no_gains_is_zero",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(0.0,),
            reason="an all-negative series has no gains, so the ratio is 0 "
            "(test_profit_factor.py::test_no_gains_is_zero)",
        ),
        SpecPin(
            label="all_zero_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series has zero gains and losses, so the ratio is 0/0, i.e. NaN "
            "(test_profit_factor.py::test_all_zero_is_nan)",
        ),
    ),
)
