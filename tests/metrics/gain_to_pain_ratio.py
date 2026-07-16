"""Spec for ``pomata.metrics.gain_to_pain_ratio`` — reducing, net return over total loss, scale-invariant."""

import math

from tests.metrics.oracles import gain_to_pain_ratio_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import gain_to_pain_ratio

GAIN_TO_PAIN_RATIO = Spec(
    factory=gain_to_pain_ratio,
    inputs=("returns",),
    params={},
    shape=Shape.REDUCING,
    oracle=gain_to_pain_ratio_reference,
    # A ratio of sums, scale-invariant
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)},
    golden_output=(0.4444,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (0.02,)},
            expected=(math.inf,),
            reason="a one-element positive series has no loss, so the ratio is +inf ",
        ),
        SpecPin(
            label="no_losses_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(math.inf,),
            reason="an all-positive series has no loss, so the ratio is +inf ",
        ),
        SpecPin(
            label="all_negative_is_minus_one",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(-1.0,),
            reason="an all-negative series has net loss equal to its total loss, so the ratio is -1 ",
        ),
        SpecPin(
            label="all_zero_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series has zero total loss and zero net return, so the ratio is a 0/0, i.e. NaN — "
            "the degenerate-denominator NaN beside the +inf pin",
        ),
    ),
)
