"""
Declaration for ``pomata.metrics.downside_deviation`` — reducing, the annualized RMS of shortfall, degree-1
homogeneous.
"""

import math

from pomata.metrics import downside_deviation
from tests_new.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests_new.metrics.harness import suite_metrics
from tests_new.metrics.oracles import reference_downside_deviation
from tests_new.support.declaration import Golden, Pin, ScaleAxis

DOWNSIDE_DEVIATION = suite_metrics(
    factory=downside_deviation,
    inputs=("returns",),
    params={"periods_per_year": 252},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.SQRT_TIME,
    degenerate=Degenerate.EXACT_ZERO,
    oracle=reference_downside_deviation,
    scaling=(ScaleAxis(roles=("returns",), degree=1),),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"threshold": math.nan}, r"threshold must be a finite number"),
        ({"threshold": math.inf}, r"threshold must be a finite number"),
        ({"threshold": -math.inf}, r"threshold must be a finite number"),
    ),
    golden=Golden(inputs={"returns": (0.02, -0.04, 0.01, -0.06, 0.03)}, output=(0.5119,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (-0.02,)},
            expected=(0.3174901573277509,),
            reason="a one-element downside series annualizes its shortfall RMS ",
        ),
        Pin(
            label="no_downside_is_zero",
            inputs={"returns": (0.01, 0.02, 0.0, 0.03)},
            expected=(0.0,),
            reason="returns all at or above the threshold have zero downside, so the deviation is 0 ",
        ),
        Pin(
            label="threshold_nonzero",
            inputs={"returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018)},
            expected=(0.24936118382779626,),
            reason="a non-zero threshold shifts the shortfall ",
            params_override={"threshold": 0.01},
        ),
    ),
)
