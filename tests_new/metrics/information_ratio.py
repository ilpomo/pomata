"""Spec for ``pomata.metrics.information_ratio`` — reducing, active return over tracking error, scale-invariant."""

import math

import polars as pl
from tests.metrics.oracles import information_ratio_reference
from tests.support import well_spread
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import information_ratio


def _well_spread_active(frame: pl.DataFrame) -> bool:
    """Reject a near-constant active series: the tracking-error denominator would be ill-conditioned."""
    returns = frame["returns"].to_list()
    benchmark = frame["benchmark"].to_list()
    active = [x - y for x, y in zip(returns, benchmark, strict=True) if x is not None and y is not None]
    return well_spread(active)


INFORMATION_RATIO = Spec(
    factory=information_ratio,
    inputs=("returns", "benchmark"),
    params={"periods_per_year": 252},
    shape=Shape.REDUCING,
    raises=(({"periods_per_year": 0}, r"periods_per_year must be >= 1"),),
    oracle=information_ratio_reference,
    conditioning=_well_spread_active,
    # A mean of the active series over its standard deviation: a joint rescale of both legs by k leaves the ratio
    # unchanged (tests/metrics/test_information_ratio.py::test_scale_invariance).
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
            reason="a single complete pair yields null — the tracking error needs two observations "
            "(tests/metrics/test_information_ratio.py::test_single_pair)",
        ),
        SpecPin(
            label="zero_tracking_error_is_inf",
            inputs={"returns": (0.01, 0.01, 0.01), "benchmark": (0.0, 0.0, 0.0)},
            expected=(math.inf,),
            reason="a constant active series has zero tracking error with a positive mean, so the ratio is +inf "
            "(tests/metrics/test_information_ratio.py::test_zero_tracking_error_is_inf)",
        ),
    ),
)
