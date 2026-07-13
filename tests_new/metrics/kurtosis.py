"""Spec for ``pomata.metrics.kurtosis`` — reducing, the standardized fourth moment minus three, scale-invariant."""

import math

import polars as pl
from tests.metrics.oracles import kurtosis_reference
from tests.support import well_spread
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import kurtosis


def _well_spread(frame: pl.DataFrame) -> bool:
    """Reject a near-constant sample: the standardized moment is a 0/0 the one- and two-pass paths resolve apart."""
    return well_spread(frame.to_series(0).to_list())


KURTOSIS = Spec(
    factory=kurtosis,
    inputs=("returns",),
    params={},
    shape=Shape.REDUCING,
    oracle=kurtosis_reference,
    conditioning=_well_spread,
    # A standardized moment is scale-invariant, degree 0 (test_kurtosis.py::test_scale_invariance).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.01, -0.02, 0.015, -0.03, 0.005, -0.01, 0.02)},
    golden_output=(-1.3223,),
    pins=(
        SpecPin(
            label="constant_is_nan",
            inputs={"returns": (0.01, 0.01, 0.01)},
            expected=(math.nan,),
            reason="a constant series has zero variance, so the standardized fourth moment is 0/0, i.e. NaN "
            "(test_kurtosis.py::test_constant_is_nan)",
        ),
        SpecPin(
            label="subnormal_magnitude_is_nan",
            inputs={"returns": (0.0, 1e-160, 2e-160)},
            expected=(math.nan,),
            reason="a subnormal-magnitude series has m2**2 underflow to zero, yielding NaN "
            "(test_kurtosis.py::test_subnormal_magnitude_is_nan)",
        ),
    ),
)
