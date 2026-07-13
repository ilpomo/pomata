"""Spec for ``pomata.metrics.omega_ratio`` — reducing, mean gain over mean loss about a threshold, scale-invariant."""

import math

from tests.metrics.oracles import omega_ratio_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import omega_ratio

OMEGA_RATIO = Spec(
    factory=omega_ratio,
    inputs=("returns",),
    params={"threshold": 0.0},
    shape=Shape.REDUCING,
    raises=(
        ({"threshold": math.nan}, r"threshold must be a finite number"),
        ({"threshold": math.inf}, r"threshold must be a finite number"),
        ({"threshold": -math.inf}, r"threshold must be a finite number"),
    ),
    oracle=omega_ratio_reference,
    # A ratio of means, scale-invariant
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)},
    golden_output=(1.4444,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(math.inf,),
            reason="one observation puts all mass on the gain side, so the ratio is +inf ",
        ),
        SpecPin(
            label="all_gain_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(math.inf,),
            reason="returns all above the threshold have no downside, so the ratio is +inf ",
        ),
        SpecPin(
            label="all_loss_is_zero",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(0.0,),
            reason="returns all below the threshold have no upside, so the ratio is 0 ",
        ),
        SpecPin(
            label="all_at_threshold_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="returns all exactly at the threshold give 0/0, so the ratio is NaN ",
        ),
    ),
)
