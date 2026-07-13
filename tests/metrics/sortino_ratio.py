"""Spec for ``pomata.metrics.sortino_ratio`` — reducing, annualized excess return over downside deviation,
scale-invariant.
"""

import math

from tests.metrics.oracles import sortino_ratio_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import sortino_ratio

SORTINO_RATIO = Spec(
    factory=sortino_ratio,
    inputs=("returns",),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    shape=Shape.REDUCING,
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    oracle=sortino_ratio_reference,
    # A ratio of mean to downside deviation at zero risk-free rate is scale-invariant
    # (test_sortino_ratio.py::test_scale_invariance).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)},
    golden_output=(4.4567,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (-0.02,)},
            expected=(-15.874507866387544,),
            reason="one observation has a defined population downside deviation, so the ratio is a real number "
            "(test_sortino_ratio.py::test_single_row)",
        ),
        SpecPin(
            label="no_downside_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(math.inf,),
            reason="all returns at/above the zero target: downside deviation is 0 with a positive mean, so +inf "
            "(test_sortino_ratio.py::test_no_downside_is_inf)",
        ),
        SpecPin(
            label="null_skipped_risk_free_rate",
            inputs={"returns": (0.012, -0.008, 0.02, None, 0.005, 0.0, -0.02, 0.018)},
            expected=(7.332599938409737,),
            reason="a null is excluded jointly with a non-default risk-free rate "
            "(test_sortino_ratio.py::test_null_skipped)",
            params_override={"risk_free_rate": 0.02},
        ),
        SpecPin(
            label="matches_reference_risk_free_rate",
            inputs={"returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018)},
            expected=(2.4195202806468132,),
            reason="reference agreement at a non-default risk-free rate "
            "(test_sortino_ratio.py::test_matches_reference)",
            params_override={"risk_free_rate": 0.02},
        ),
    ),
)
