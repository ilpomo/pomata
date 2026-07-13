"""Spec for ``pomata.metrics.downside_deviation_rolling`` — the rolling shortfall RMS, degree-1 homogeneous."""

import math

import polars as pl
from tests.metrics.oracles import downside_deviation_rolling_reference
from tests.support import windows_well_spread
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import downside_deviation_rolling


def _windows_well_spread(frame: pl.DataFrame) -> bool:
    """Reject a near-constant window: the one-pass shortfall square cannot track the two-pass oracle there."""
    return windows_well_spread(frame.to_series(0).to_list(), 3)


DOWNSIDE_DEVIATION_ROLLING = Spec(
    factory=downside_deviation_rolling,
    inputs=("returns",),
    params={"window": 3, "periods_per_year": 252, "threshold": 0.0},
    shape=Shape.SERIES,
    warmup=2,
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"threshold": math.nan}, r"threshold must be a finite number"),
        ({"threshold": math.inf}, r"threshold must be a finite number"),
        ({"threshold": -math.inf}, r"threshold must be a finite number"),
    ),
    oracle=downside_deviation_rolling_reference,
    conditioning=_windows_well_spread,
    # Degree-1 homogeneous per window at threshold=0 (by analogy to the reducing downside_deviation).
    scale=(ScaleAxis(roles=("returns",), degree=1),),
    golden_input={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)},
    golden_output=(None, None, 0.1833, 0.2049, 0.0917, 0.0917, 0.1375),
    pins=(
        SpecPin(
            label="no_downside_window_is_zero",
            inputs={"returns": (0.01, 0.02, 0.03, 0.04)},
            expected=(None, None, 0.0, 0.0),
            reason="a window with every return at or above the threshold has zero shortfall "
            "(test_downside_deviation_rolling.py::test_no_downside_window_is_zero)",
        ),
        SpecPin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03)},
            expected=(None, None, 0.18330302779823363),
            reason="when the window equals the series length only the last row is defined "
            "(test_downside_deviation_rolling.py::test_window_equals_length)",
        ),
    ),
)
