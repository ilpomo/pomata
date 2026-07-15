"""Spec for ``pomata.metrics.beta_rolling`` — the regression slope over a trailing window, scale-invariant."""

import math

import polars as pl
from tests.metrics.oracles import beta_rolling_reference
from tests.support import RELATIVE_TOLERANCE_ROLLING_ORACLE, windows_well_conditioned
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import beta_rolling

# Spec-local conditioning floor: the shared 1e-2 rejected ~19% of drawn frames while the bare slope's measured
# disagreement onset sits at var_rel ~4e-14..6e-14 (the exact bit-flat core below that is already snapped by the
# implementation's own rolling_max == rolling_min guard), so 1e-9 keeps 4-5 orders of margin above the onset while
# re-admitting the ~7 orders of well-behaved windows the shared cut threw away.
_CONDITIONING_FLOOR = 1e-9


def _windows_well_conditioned(frame: pl.DataFrame) -> bool:
    """Reject any trailing window whose benchmark variance is too near zero for the one-pass slope to track."""
    return windows_well_conditioned(frame["benchmark"].to_list(), 4, floor=_CONDITIONING_FLOOR)


BETA_ROLLING = Spec(
    factory=beta_rolling,
    inputs=("returns", "benchmark"),
    params={"window": 4},
    shape=Shape.SERIES,
    warmup=3,
    raises=(({"window": 1}, r"window must be >= 2"),),
    oracle=beta_rolling_reference,
    conditioning=_windows_well_conditioned,
    # A one-pass rolling covariance / variance regression against a recompute-per-window two-pass oracle.
    oracle_rel_tol=RELATIVE_TOLERANCE_ROLLING_ORACLE,
    # A slope (cov/var, both legs same units) is scale-invariant under a joint rescale of both legs (verified
    # numerically).
    scale=(ScaleAxis(roles=("returns", "benchmark"), degree=0),),
    golden_input={
        "returns": (0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.01, 0.02),
        "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018),
    },
    golden_output=(None, None, None, 1.2608, 1.2628, 1.2652, 1.2592, 1.0331),
    pins=(
        SpecPin(
            label="null_in_constant_benchmark_window_is_null",
            inputs={"returns": (0.02, None, 0.03, 0.01, 0.02), "benchmark": (0.1, 0.1, 0.1, 0.1, 0.1)},
            expected=(None, None, None, None, math.nan),
            reason="a window holding a null yields null (pairwise-complete gate) rather than the flat-benchmark NaN",
            params_override={"window": 3},
        ),
        SpecPin(
            label="constant_benchmark_window_is_nan",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01), "benchmark": (0.1, 0.1, 0.1, 0.1)},
            expected=(None, None, math.nan, math.nan),
            reason="a fully-defined window with an exactly-constant benchmark has zero variance, so the slope is "
            "NaN — the exact core of the near-constant regime the conditioning filter excludes from the property "
            "tiers",
            params_override={"window": 3},
            covers_conditioning=True,
        ),
    ),
)
