"""
Declaration for ``pomata.metrics.kurtosis_rolling`` — the standardized fourth moment per window, scale-invariant.
"""

import math

import polars as pl

from pomata.metrics import kurtosis_rolling
from tests_new.metrics.enums import BehaviorNan, BehaviorNull
from tests_new.metrics.harness import suite_metrics
from tests_new.metrics.kurtosis import KURTOSIS
from tests_new.metrics.oracles import reference_kurtosis_rolling
from tests_new.support.declaration import Golden, Pin, ScaleAxis
from tests_new.support.strategies import windows_well_conditioned
from tests_new.support.tolerances import TOLERANCE_RELATIVE_ROLLING_ORACLE


def _windows_well_conditioned(frame: pl.DataFrame) -> bool:
    """
    Reject a near-constant CURRENT window: the standardized moment is a 0/0 the one- and two-pass paths resolve
    apart, the same genuine degeneracy the reducing kurtosis filters (its measured onset straddles the shared cut).
    This predicate cannot see failures rooted in a PAST window — a stale residue in the incremental kernel's
    running sums after a large value exits the buffer (the native kernels' own defect, pola-rs/polars#28290, fixed
    upstream in #28309) — which the kernel recomputes exactly for exits more than one order of magnitude above the
    window's scale (covering the ratio**4 amplification); a smaller exit onto a window whose spread sits under the
    conditioning floor is the untested tail of that upstream defect. The guarded regime is witnessed by the
    moderate_outlier_exit pin below.
    """
    return windows_well_conditioned(frame.to_series(0).to_list(), 4)


KURTOSIS_ROLLING = suite_metrics(
    factory=kurtosis_rolling,
    inputs=("returns",),
    params={"window": 4},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    rolling_of=KURTOSIS,
    window="window",
    warmup=3,
    oracle=reference_kurtosis_rolling,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    raises=(({"window": 1}, r"window must be >= 2"),),
    conditioning=_windows_well_conditioned,
    golden=Golden(
        inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)},
        output=(None, None, None, -1.4266, -1.7785, -1.64, -1.099),
    ),
    pins=(
        Pin(
            label="constant_window_is_nan",
            inputs={"returns": (0.3, 0.3, 0.3, 0.3)},
            expected=(None, None, math.nan, math.nan),
            reason="a constant window has zero variance, so the standardized moment is 0/0, i.e. NaN — the "
            "exact core of the near-constant regime the conditioning filter excludes from the "
            "property tiers",
            params_override={"window": 3},
            covers_conditioning=True,
        ),
        Pin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(None, None, None, None, -1.4908058409951328),
            reason="when the window equals the series length only the last row is defined",
            params_override={"window": 5},
        ),
        Pin(
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
            reason="after a moderate outlier (tens of times the window scale) exits the buffer, the "
            "incremental kernel's running sums would keep a stale residue the fourth power amplifies "
            "as ratio**4; the src recompute trigger rebuilds those windows exactly (worst |impl - "
            "oracle| here 3.4e-13) — a regime rooted in a PAST window, which the conditioning "
            "predicate cannot see, so this pin witnesses it",
        ),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=1e-07,
)
