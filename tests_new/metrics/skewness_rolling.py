"""Spec for ``pomata.metrics.skewness_rolling`` — the standardized third moment per window, scale-invariant."""

import math

import polars as pl
from tests.metrics.oracles import skewness_rolling_reference
from tests.support import RELATIVE_TOLERANCE_SCALE, windows_well_conditioned
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import skewness_rolling


def _windows_well_conditioned(frame: pl.DataFrame) -> bool:
    """Reject a near-constant window: the standardized moment is a 0/0 the one- and two-pass paths resolve apart."""
    return windows_well_conditioned(frame.to_series(0).to_list(), 4)


SKEWNESS_ROLLING = Spec(
    factory=skewness_rolling,
    inputs=("returns",),
    params={"window": 4},
    shape=Shape.SERIES,
    warmup=3,
    raises=(({"window": 1}, r"window must be >= 2"),),
    oracle=skewness_rolling_reference,
    conditioning=_windows_well_conditioned,
    oracle_rel_tol=RELATIVE_TOLERANCE_SCALE,
    oracle_abs_tol=1e-7,
    # A standardized moment per window is scale-invariant, degree 0 (by analogy to the reducing skewness).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)},
    golden_output=(None, None, None, 0.278, 0.0, 0.0, 0.6568),
    pins=(
        SpecPin(
            label="constant_window_is_nan",
            inputs={"returns": (0.3, 0.3, 0.3, 0.3)},
            expected=(None, None, math.nan, math.nan),
            reason="a constant window has zero variance, so the standardized moment is 0/0, i.e. NaN "
            "(test_skewness_rolling.py::test_constant_window_is_nan)",
            params_override={"window": 3},
        ),
        SpecPin(
            label="constant_window_by_slide_is_nan",
            inputs={"returns": (0.03, 0.0, 0.0, 0.0, 0.0)},
            expected=(None, None, None, 1.1547005383792517, math.nan),
            reason="a window that becomes bit-constant only because a larger value slid out still reads as NaN "
            "(test_skewness_rolling.py::test_constant_window_by_slide_is_nan)",
            params_override={"window": 4},
        ),
        SpecPin(
            label="near_constant_window_is_finite",
            inputs={"returns": (100.0, 100.0, 100.0, 100.000001)},
            expected=(None, None, None, 1.1547005383792515),
            reason="a near-constant (non-bit-identical) window yields the finite reference skewness "
            "(test_skewness_rolling.py::test_near_constant_window_is_finite)",
        ),
    ),
)
