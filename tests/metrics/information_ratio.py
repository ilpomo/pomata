"""Spec for ``pomata.metrics.information_ratio`` — reducing, active return over tracking error, scale-invariant."""

import math

from tests.metrics.oracles import information_ratio_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import information_ratio

INFORMATION_RATIO = Spec(
    factory=information_ratio,
    inputs=("returns", "benchmark"),
    params={"periods_per_year": 252},
    shape=Shape.REDUCING,
    raises=(({"periods_per_year": 0}, r"periods_per_year must be >= 1"),),
    oracle=information_ratio_reference,
    # A mean of the active series over its standard deviation: a joint rescale of both legs by k leaves the ratio
    # unchanged.
    scale=(ScaleAxis(roles=("returns", "benchmark"), degree=0),),
    golden_input={
        "returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018),
        "benchmark": (0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015),
    },
    golden_output=(-0.842,),
    pins=(
        SpecPin(
            label="single_pair",
            inputs={"returns": (0.05,), "benchmark": (0.04,)},
            expected=(None,),
            reason="a single complete pair yields null — the tracking error needs two observations",
        ),
        SpecPin(
            label="zero_tracking_error_is_inf",
            inputs={"returns": (0.01, 0.01, 0.01), "benchmark": (0.0, 0.0, 0.0)},
            expected=(math.inf,),
            reason="a constant active series has zero tracking error with a positive mean, so the ratio is +inf",
        ),
        SpecPin(
            label="zero_active_is_nan",
            inputs={"returns": (0.01, 0.02, 0.03), "benchmark": (0.01, 0.02, 0.03)},
            expected=(math.nan,),
            reason="identical legs give an exactly-zero active series: zero mean over zero tracking error is the "
            "0/0 NaN, resolved by the exact dispersion guard — the exact-zero core of the near-constant regime; "
            "no conditioning filter is declared: a first-moment ratio needs none, and the fuzz never rounds impl "
            "and oracle apart",
        ),
    ),
)
