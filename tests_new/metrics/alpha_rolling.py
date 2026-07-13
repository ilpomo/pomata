"""Spec for ``pomata.metrics.alpha_rolling`` — Jensen's alpha over a trailing window, scale-exempt."""

import math

import polars as pl
from tests_new.metrics.oracles import alpha_rolling_reference
from tests_new.support import RELATIVE_TOLERANCE_SCALE, windows_well_conditioned
from tests_new.support.spec import ScaleExempt, Shape, Spec, SpecPin

from pomata.metrics import alpha_rolling


def _windows_well_conditioned(frame: pl.DataFrame) -> bool:
    """Reject any trailing window whose benchmark variance is too near zero for the one-pass slope to track."""
    return windows_well_conditioned(frame["benchmark"].to_list(), 4)


ALPHA_ROLLING = Spec(
    factory=alpha_rolling,
    inputs=("returns", "benchmark"),
    params={"window": 4, "periods_per_year": 252, "risk_free_rate": 0.0},
    shape=Shape.SERIES,
    warmup=3,
    raises=(
        ({"window": 1}, r"window must be >= 2"),
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    oracle=alpha_rolling_reference,
    conditioning=_windows_well_conditioned,
    oracle_rel_tol=RELATIVE_TOLERANCE_SCALE,
    # Annualizes an excess leg against a fixed per-period risk-free constant — not scale-invariant, by the same
    # reasoning as the reducing alpha (verified numerically).
    scale=ScaleExempt(
        reason="annualizes an excess leg against a fixed per-period risk-free constant — neither scale-homogeneous "
        "nor scale-invariant"
    ),
    golden_input={
        "returns": (0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.01, 0.02),
        "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018),
    },
    golden_params={"window": 4, "periods_per_year": 252},
    golden_output=(None, None, None, -0.0864, -0.0096, -0.0227, 0.4932, 0.7998),
    pins=(
        SpecPin(
            label="null_in_window_is_null",
            inputs={"returns": (0.01, None, 0.03, -0.01, 0.02), "benchmark": (0.008, -0.015, 0.025, None, 0.018)},
            expected=(None, None, None, None, None),
            reason="a null in either leg nulls every window touching it, disjoint rows and all "
            "(tests/metrics/test_alpha_rolling.py::test_null_in_window_is_null)",
            params_override={"window": 3},
        ),
        SpecPin(
            label="null_in_constant_benchmark_window_is_null",
            inputs={"returns": (0.02, None, 0.03, 0.01, 0.02), "benchmark": (0.1, 0.1, 0.1, 0.1, 0.1)},
            expected=(None, None, None, None, math.nan),
            reason="a null in the returns leg wins over the constant-benchmark NaN branch on incomplete windows "
            "(tests/metrics/test_alpha_rolling.py::test_null_in_constant_benchmark_window_is_null)",
            params_override={"window": 3},
        ),
        SpecPin(
            label="constant_benchmark_window_is_nan",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01), "benchmark": (0.1, 0.1, 0.1, 0.1)},
            expected=(None, None, math.nan, math.nan),
            reason="a window whose benchmark is exactly constant makes the embedded slope NaN "
            "(tests/metrics/test_alpha_rolling.py::test_constant_benchmark_window_is_nan)",
            params_override={"window": 3},
        ),
        SpecPin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01), "benchmark": (0.008, -0.015, 0.025, -0.008)},
            expected=(None, None, None, -0.14222632801543245),
            reason="when the window equals the series length only the last row is defined "
            "(tests/metrics/test_alpha_rolling.py::test_window_equals_length)",
            params_override={"window": 4},
        ),
    ),
)
