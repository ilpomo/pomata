"""Spec for ``pomata.metrics.information_ratio_rolling`` — active return over tracking error per window,
scale-invariant.
"""

import math

import polars as pl
from tests.metrics.oracles import information_ratio_rolling_reference
from tests.support import RELATIVE_TOLERANCE_ROLLING_ORACLE, windows_well_conditioned
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import information_ratio_rolling

# Spec-local conditioning floor: the shared 1e-2 rejects ~23% of drawn frames. Measured: the impl-vs-oracle
# crossing of the property band sits at active var_rel ~1.9e-4 in the worst realistic embedding
# (and lower elsewhere), so 1e-3 sits ~5x above the highest measured crossing; any floor below ~2e-4 (e.g. a
# seemingly-generous 1e-6) would sit UNDER that crossing and re-admit the measured disagreement band.
_CONDITIONING_FLOOR = 1e-3


def _active_windows_conditioned(frame: pl.DataFrame) -> bool:
    """Reject any trailing window whose active-return variance is too near zero for the one-pass ratio to track."""
    returns = frame["returns"].to_list()
    benchmark = frame["benchmark"].to_list()
    active = [x - y if (x is not None and y is not None) else None for x, y in zip(returns, benchmark, strict=True)]
    return windows_well_conditioned(active, 4, floor=_CONDITIONING_FLOOR)


INFORMATION_RATIO_ROLLING = Spec(
    factory=information_ratio_rolling,
    inputs=("returns", "benchmark"),
    params={"window": 4, "periods_per_year": 252},
    shape=Shape.SERIES,
    warmup=3,
    raises=(
        ({"window": 1}, r"window must be >= 2"),
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
    ),
    oracle=information_ratio_rolling_reference,
    # A one-pass rolling active mean over tracking error against a recompute-per-window two-pass oracle.
    oracle_rel_tol=RELATIVE_TOLERANCE_ROLLING_ORACLE,
    # A mean of the active series over its rolling standard deviation: a joint rescale of both legs by k leaves the
    # ratio unchanged (verified numerically).
    conditioning=_active_windows_conditioned,
    scale=(ScaleAxis(roles=("returns", "benchmark"), degree=0),),
    golden_input={
        "returns": (0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.01, 0.02),
        "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018),
    },
    golden_output=(None, None, None, 2.3539, 2.3539, 5.0387, 2.8393, 22.9129),
    pins=(
        SpecPin(
            label="null_in_window_is_null",
            inputs={"returns": (0.01, None, 0.03, -0.01, 0.02), "benchmark": (0.008, -0.015, 0.025, None, 0.018)},
            expected=(None, None, None, None, None),
            reason="two separate misses in two different legs poison every window=3 window over a 5-row series",
            params_override={"window": 3},
        ),
        SpecPin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01), "benchmark": (0.008, -0.015, 0.025, -0.008)},
            expected=(None, None, None, -3.1314621094842335e-15),
            reason="window equals series length, so only the last row is defined; the active series sums to (near) "
            "zero",
        ),
        SpecPin(
            label="constant_window_by_slide_is_inf",
            inputs={"returns": (1000000.0, 0.1, 0.1, 0.1, 0.1), "benchmark": (0.0, 0.0, 0.0, 0.0, 0.0)},
            expected=(None, None, 9.16515413945737, math.inf, math.inf),
            reason="once the outlier slides out, the window is bit-constant with zero tracking error, so the ratio is "
            "+inf",
            params_override={"window": 3},
        ),
        SpecPin(
            label="zero_tracking_error_is_inf",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01), "benchmark": (0.0, 0.0, 0.0, 0.0)},
            expected=(None, None, math.inf, math.inf),
            reason="a constant active-return window has zero tracking error with a positive mean, so the ratio is "
            "+inf — the exact core of the near-constant regime the conditioning filter excludes from the property "
            "tiers",
            params_override={"window": 3},
            covers_conditioning=True,
        ),
        SpecPin(
            label="flat_zero_active_window_by_slide_is_nan",
            inputs={
                "returns": (-0.3233, -0.6457, 0.0, 0.4404, 0.0, 0.0, 0.0, 0.0),
                "benchmark": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            },
            expected=(
                None,
                None,
                None,
                -4.522342411070932,
                -1.8213352931142792,
                7.9372539331937695,
                7.9372539331937695,
                math.nan,
            ),
            reason="a window whose active returns are all exactly zero degenerates to 0/0 -> NaN even after larger "
            "active values slid out: the exact rolling mean pins the numerator to zero, so the incremental "
            "running-sum residue cannot ride above the exactly-zero tracking error as a spurious inf; the "
            "residue is bit-sensitive, so the prefix must reach the kernel with exactly these bits (a zero "
            "benchmark keeps the active leg bit-identical to the returns)",
        ),
    ),
)
