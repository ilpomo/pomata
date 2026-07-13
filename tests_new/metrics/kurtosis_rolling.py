"""Spec for ``pomata.metrics.kurtosis_rolling`` — the standardized fourth moment per window, scale-invariant."""

import math

import polars as pl
from tests_new.metrics.oracles import kurtosis_rolling_reference
from tests_new.support import RELATIVE_TOLERANCE_SCALE, windows_well_conditioned
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import kurtosis_rolling


def _windows_well_conditioned(frame: pl.DataFrame) -> bool:
    """Reject a near-constant window: the standardized moment is a 0/0 the one- and two-pass paths resolve apart."""
    return windows_well_conditioned(frame.to_series(0).to_list(), 4)


KURTOSIS_ROLLING = Spec(
    factory=kurtosis_rolling,
    inputs=("returns",),
    params={"window": 4},
    shape=Shape.SERIES,
    warmup=3,
    raises=(({"window": 1}, r"window must be >= 2"),),
    oracle=kurtosis_rolling_reference,
    conditioning=_windows_well_conditioned,
    oracle_rel_tol=RELATIVE_TOLERANCE_SCALE,
    oracle_abs_tol=1e-7,
    # A standardized moment per window is scale-invariant, degree 0 (by analogy to the reducing kurtosis).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)},
    golden_output=(None, None, None, -1.4266, -1.7785, -1.64, -1.099),
    pins=(
        SpecPin(
            label="constant_window_is_nan",
            inputs={"returns": (0.3, 0.3, 0.3, 0.3)},
            expected=(None, None, math.nan, math.nan),
            reason="a constant window has zero variance, so the standardized moment is 0/0, i.e. NaN "
            "(test_kurtosis_rolling.py::test_constant_window_is_nan)",
            params_override={"window": 3},
        ),
        SpecPin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(None, None, None, None, -1.4908058409951328),
            reason="when the window equals the series length only the last row is defined "
            "(test_kurtosis_rolling.py::test_window_equals_length)",
            params_override={"window": 5},
        ),
    ),
)
