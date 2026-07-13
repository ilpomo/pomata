"""Spec for ``pomata.metrics.skewness_rolling`` — the standardized third moment per window, scale-invariant."""

import math

import polars as pl
from tests.metrics.oracles import skewness_rolling_reference
from tests.support import RELATIVE_TOLERANCE_ROLLING_ORACLE, windows_well_conditioned
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import skewness_rolling


def _windows_well_conditioned(frame: pl.DataFrame) -> bool:
    """
    Reject a near-constant CURRENT window: the standardized moment is a 0/0 the one- and two-pass paths resolve
    apart, the same genuine degeneracy the reducing skewness filters. This predicate cannot see failures rooted in
    a PAST window — a stale residue in the incremental kernel's running sums after a large value exits the buffer
    (the same leak as kurtosis_rolling, one power lower: ratio**3) — which the kernel recomputes exactly; that
    regime is witnessed by the moderate_outlier_exit pin below, so the filter's stated reason matches what it
    actually guards.
    """
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
    oracle_rel_tol=RELATIVE_TOLERANCE_ROLLING_ORACLE,
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
            reason="a constant window has zero variance, so the standardized moment is 0/0, i.e. NaN — the exact "
            "core of the near-constant regime the conditioning filter excludes from the property tiers",
            params_override={"window": 3},
            covers_conditioning=True,
        ),
        SpecPin(
            label="constant_window_by_slide_is_nan",
            inputs={"returns": (0.03, 0.0, 0.0, 0.0, 0.0)},
            expected=(None, None, None, 1.1547005383792517, math.nan),
            reason="a window that becomes bit-constant only because a larger value slid out still reads as NaN",
            params_override={"window": 4},
        ),
        SpecPin(
            label="near_constant_window_is_finite",
            inputs={"returns": (100.0, 100.0, 100.0, 100.000001)},
            expected=(None, None, None, 1.1547005383792515),
            reason="a near-constant (non-bit-identical) window yields the finite reference skewness",
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
                -1.1547005383792515,
                -1.1547005383792515,
                -1.0412725912596517,
                0.1805399198333302,
                0.18053991983333023,
                0.7392763059526237,
                0.7801748612255287,
                0.8015573990524767,
                0.8015573990524767,
                1.1547005383792213,
                1.1547005383792213,
                1.154700538379252,
            ),
            reason="after a moderate outlier exits the buffer, the incremental kernel's running sums would keep a "
            "stale residue the third power amplifies as ratio**3 — a property of a PAST window the conditioning "
            "predicate cannot see; the kernel's recompute trigger rebuilds those windows exactly (worst "
            "|impl - oracle| here 3.0e-14)",
        ),
    ),
)
