"""Spec for ``pomata.metrics.downside_deviation`` — reducing, the annualized RMS of shortfall, degree-1 homogeneous."""

import math

from tests.metrics.oracles import downside_deviation_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import downside_deviation

DOWNSIDE_DEVIATION = Spec(
    factory=downside_deviation,
    inputs=("returns",),
    params={"periods_per_year": 252},
    shape=Shape.REDUCING,
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"threshold": math.nan}, r"threshold must be a finite number"),
        ({"threshold": math.inf}, r"threshold must be a finite number"),
        ({"threshold": -math.inf}, r"threshold must be a finite number"),
    ),
    oracle=downside_deviation_reference,
    # Degree-1 homogeneous in the returns at threshold=0 (test_downside_deviation.py::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("returns",), degree=1),),
    golden_input={"returns": (0.02, -0.04, 0.01, -0.06, 0.03)},
    golden_output=(0.5119,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (-0.02,)},
            expected=(0.3174901573277509,),
            reason="a one-element downside series annualizes its shortfall RMS "
            "(test_downside_deviation.py::test_single_row)",
        ),
        SpecPin(
            label="no_downside_is_zero",
            inputs={"returns": (0.01, 0.02, 0.0, 0.03)},
            expected=(0.0,),
            reason="returns all at or above the threshold have zero downside, so the deviation is 0 "
            "(test_downside_deviation.py::test_no_downside_is_zero)",
        ),
        SpecPin(
            label="threshold_nonzero",
            inputs={"returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018)},
            expected=(0.24936118382779626,),
            reason="a non-zero threshold shifts the shortfall "
            "(test_downside_deviation.py::test_matches_reference_with_threshold)",
            params_override={"threshold": 0.01},
        ),
    ),
)
