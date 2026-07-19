"""
Declaration for ``pomata.metrics.kurtosis_rolling`` — the standardized fourth moment per window, scale-invariant.
"""

import math

import polars as pl

from pomata.metrics import kurtosis_rolling
from tests.metrics.enums import BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.kurtosis import KURTOSIS
from tests.metrics.oracles import reference_kurtosis_rolling
from tests.support.declaration import Example, Golden, Pin, ScaleAxis
from tests.support.strategies import windows_well_conditioned
from tests.support.tolerances import TOLERANCE_RELATIVE_ROLLING_ORACLE


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
    reference='Joanes, D. N. & Gill, C. A. (1998). "Comparing Measures of Sample Skewness and '
    'Kurtosis." *Journal of the Royal Statistical Society: Series D (The Statistician)*, '
    "47(1), 183-189.",
    doi="https://doi.org/10.1111/1467-9884.00122",
    wikipedia="https://en.wikipedia.org/wiki/Kurtosis",
    see_also=(
        ("kurtosis", "The whole-series reducing form."),
        ("skewness_rolling", "The rolling third-moment counterpart."),
        ("value_at_risk_modified", "Uses excess kurtosis in its Cornish-Fisher tail correction."),
    ),
    opener_override="Each window matches an independent reference oracle (the reducing :func:`kurtosis` "
    "recomputed over the window).",
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Degenerate denominator",
            "a constant window has zero variance, so the standardized moment is a ``0 / 0``, i.e. ``NaN``.",
        ),
        (
            "Stability",
            "the native kernel carries running sums, so a value leaving the window can leave a stale "
            "residue behind: an exit more than one order of magnitude above the window's scale is "
            "recomputed exactly, while a smaller exit onto a window whose own spread has collapsed "
            "can amplify that residue through the near-zero variance into a wrong finite value — "
            "reported, not clipped (the tail of pola-rs/polars#28290, fixed upstream in #28309).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The rolling excess kurtosis for each row, the same length as the input. The first "
    "``window - 1`` rows are ``null`` (warm-up): the window must hold ``window`` non-null "
    "values before a result is emitted.",
    raises_prose="ValueError: If ``window < 2``.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 2``.",
    },
    examples=(
        Example(inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)}, params={"window": 4}, round_to=4),
        Example(
            inputs={
                "returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015, 0.02, -0.01, 0.04, -0.03, 0.01, 0.025, -0.02)
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up on its own "
            "(the ``B`` group never borrows ``A``'s tail):",
            partition=("A", "A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B", "B"),
            params={"window": 4},
            round_to=4,
        ),
        Example(
            inputs={"returns": (None, 0.01, float("nan"), -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)},
            intro="A leading ``null`` and a later ``NaN`` show the per-window masking, with the result "
            "recovering once both leave the window:",
            params={"window": 4},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.3, 0.3, 0.3, 0.3)},
            intro="**Degenerate denominator** — a constant window has zero variance, so the standardized "
            "moment is ``0/0``, i.e. ``NaN``:",
            params={"window": 3},
        ),
    ),
)
