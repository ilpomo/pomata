"""Spec for ``pomata.metrics.kurtosis_rolling`` — the standardized fourth moment per window, scale-invariant."""

import math

import polars as pl
from tests.metrics.oracles import kurtosis_rolling_reference
from tests.support import RELATIVE_TOLERANCE_ROLLING_ORACLE, windows_well_conditioned
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import kurtosis_rolling


def _windows_well_conditioned(frame: pl.DataFrame) -> bool:
    """
    Reject a near-constant CURRENT window: the standardized moment is a 0/0 the one- and two-pass paths resolve
    apart, the same genuine degeneracy the reducing kurtosis filters (its measured onset straddles the shared cut).
    This predicate cannot see failures rooted in a PAST window — a stale residue in the incremental kernel's
    running sums after a large value exits the buffer — which the kernel recomputes exactly (the src recompute
    trigger covers ratio**4 amplification); that regime is witnessed by the moderate_outlier_exit pin below.
    """
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
    oracle_rel_tol=RELATIVE_TOLERANCE_ROLLING_ORACLE,
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
            reason="a constant window has zero variance, so the standardized moment is 0/0, i.e. NaN — the exact "
            "core of the near-constant regime the conditioning filter excludes from the property tiers",
            params_override={"window": 3},
            covers_conditioning=True,
        ),
        SpecPin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(None, None, None, None, -1.4908058409951328),
            reason="when the window equals the series length only the last row is defined",
            params_override={"window": 5},
        ),
        SpecPin(
            label="moderate_outlier_exit_matches_reference",
            inputs={
                "returns": (
                    0.01,
                    0.01,
                    -1.0,
                    0.01,
                    0.01,
                    0.2174,
                    0.3097,
                    0.01,
                    0.8594,
                    0.01,
                    0.3,
                    0.01,
                    0.01,
                    0.01,
                    0.025,
                )
            },
            expected=(
                None,
                None,
                None,
                -0.666666666666667,
                -0.666666666666667,
                -0.7385644585582893,
                -1.7630178398191556,
                -1.7630178398191554,
                -0.9075115004157928,
                -1.0398296785270926,
                -1.016312919913764,
                -1.0163129199137637,
                -0.6666666666663654,
                -0.6666666666663654,
                -0.6666666666666656,
            ),
            reason="after a moderate outlier (tens of times the window scale) exits the buffer, the incremental "
            "kernel's running sums would keep a stale residue the fourth power amplifies as ratio**4; the src "
            "recompute trigger rebuilds those windows exactly (worst |impl - oracle| here 3.4e-13) — a regime "
            "rooted in a PAST window, which the conditioning predicate cannot see, so this pin witnesses it",
        ),
    ),
)
