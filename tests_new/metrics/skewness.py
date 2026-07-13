"""Spec for ``pomata.metrics.skewness`` — reducing, the standardized third moment, scale-invariant."""

import math

import polars as pl
from tests_new.metrics.oracles import skewness_reference
from tests_new.support import well_spread
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import skewness


def _well_spread(frame: pl.DataFrame) -> bool:
    """Reject a near-constant sample: the standardized moment is a 0/0 the one- and two-pass paths resolve apart."""
    return well_spread(frame.to_series(0).to_list())


SKEWNESS = Spec(
    factory=skewness,
    inputs=("returns",),
    params={},
    shape=Shape.REDUCING,
    oracle=skewness_reference,
    conditioning=_well_spread,
    # A standardized moment is scale-invariant, degree 0 (test_skewness.py::test_scale_invariance).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.01, -0.02, 0.015, -0.03, 0.005, -0.01, 0.02)},
    golden_output=(-0.384,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(math.nan,),
            reason="one observation has zero variance, so the standardized third moment is 0/0, i.e. NaN "
            "(test_skewness.py::test_single_row)",
        ),
        SpecPin(
            label="constant_is_nan",
            inputs={"returns": (0.01, 0.01, 0.01)},
            expected=(math.nan,),
            reason="a constant series has zero variance, so the standardized moment is 0/0, i.e. NaN "
            "(test_skewness.py::test_constant_is_nan)",
        ),
        SpecPin(
            label="subnormal_magnitude_is_nan",
            inputs={"returns": (0.0, 1e-160, 2e-160)},
            expected=(math.nan,),
            reason="a subnormal-magnitude series has m2**1.5 underflow to zero, yielding NaN "
            "(test_skewness.py::test_subnormal_magnitude_is_nan)",
        ),
    ),
)
