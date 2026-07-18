"""
Declaration for ``pomata.metrics.omega_ratio`` — reducing, mean gain over mean loss about a threshold, scale-
invariant.
"""

import math

from pomata.metrics import omega_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_omega_ratio
from tests.support.declaration import Golden, Pin, ScaleAxis

OMEGA_RATIO = suite_metrics(
    factory=omega_ratio,
    inputs=("returns",),
    params={"threshold": 0.0},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_omega_ratio,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    raises=(
        ({"threshold": math.nan}, r"threshold must be a finite number"),
        ({"threshold": math.inf}, r"threshold must be a finite number"),
        ({"threshold": -math.inf}, r"threshold must be a finite number"),
    ),
    golden=Golden(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, output=(1.4444,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(math.inf,),
            reason="one observation puts all mass on the gain side, so the ratio is +inf ",
        ),
        Pin(
            label="all_gain_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(math.inf,),
            reason="returns all above the threshold have no downside, so the ratio is +inf ",
        ),
        Pin(
            label="all_loss_is_zero",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(0.0,),
            reason="returns all below the threshold have no upside, so the ratio is 0 ",
        ),
        Pin(
            label="all_at_threshold_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="returns all exactly at the threshold give 0/0, so the ratio is NaN ",
        ),
        Pin(
            label="matches_reference_with_threshold",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(0.6,),
            reason="agreement at a non-default threshold — the shifted-gain/loss split every other tier "
            "leaves at the 0.0 default, mirroring the rolling twin's pin",
            params_override={"threshold": 0.01},
        ),
    ),
)
