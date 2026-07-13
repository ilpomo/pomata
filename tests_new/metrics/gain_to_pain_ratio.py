"""Spec for ``pomata.metrics.gain_to_pain_ratio`` — reducing, net return over total loss, scale-invariant."""

import math

from tests.metrics.oracles import gain_to_pain_ratio_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import gain_to_pain_ratio

GAIN_TO_PAIN_RATIO = Spec(
    factory=gain_to_pain_ratio,
    inputs=("returns",),
    params={},
    shape=Shape.REDUCING,
    oracle=gain_to_pain_ratio_reference,
    # A ratio of sums, scale-invariant (test_gain_to_pain_ratio.py::test_scale_invariance).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)},
    golden_output=(0.4444,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (0.02,)},
            expected=(math.inf,),
            reason="a one-element positive series has no loss, so the ratio is +inf "
            "(test_gain_to_pain_ratio.py::test_single_row)",
        ),
        SpecPin(
            label="no_losses_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(math.inf,),
            reason="an all-positive series has no loss, so the ratio is +inf "
            "(test_gain_to_pain_ratio.py::test_no_losses_is_inf)",
        ),
        SpecPin(
            label="all_negative_is_minus_one",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(-1.0,),
            reason="an all-negative series has net loss equal to its total loss, so the ratio is -1 "
            "(test_gain_to_pain_ratio.py::test_all_negative_is_minus_one)",
        ),
    ),
)
