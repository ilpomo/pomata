"""Spec for ``pomata.metrics.sharpe_ratio`` — reducing, annualized, null-skipping, NaN-poisoning, scale-invariant."""

import math

from tests_new.metrics.oracles import sharpe_ratio_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import sharpe_ratio

SHARPE_RATIO = Spec(
    factory=sharpe_ratio,
    inputs=("returns",),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    shape=Shape.REDUCING,
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    oracle=sharpe_ratio_reference,
    # A ratio of a mean to a standard deviation at a zero risk-free rate: scale-invariant, degree 0 (tests/metrics/
    # test_sharpe_ratio.py:181 test_scale_invariance).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)},
    golden_output=(2.4285,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(None,),
            reason="one observation has no sample dispersion, so the ratio is null (tests/metrics/test_sharpe_ratio.py"
            "::test_single_row)",
        ),
        SpecPin(
            label="zero_volatility",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            expected=(math.inf,),
            reason="a constant series has zero dispersion with a positive mean, so the ratio is +inf "
            "(tests/metrics/test_sharpe_ratio.py::test_zero_volatility_is_inf)",
        ),
        SpecPin(
            label="zero_excess_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series at a zero risk-free rate has zero mean AND zero dispersion, so the ratio is "
            "the 0/0 NaN — the exact-zero core of the near-constant regime the old suite's well-spread filter "
            "excluded (a first-moment ratio needs no such filter: the fuzz never rounds impl and oracle apart)",
        ),
    ),
)
