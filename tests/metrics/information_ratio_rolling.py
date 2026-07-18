"""Declaration for ``pomata.metrics.information_ratio_rolling`` — active return over tracking error per window."""

import math

import polars as pl

from pomata.metrics import information_ratio_rolling
from tests.metrics.enums import BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.information_ratio import INFORMATION_RATIO
from tests.metrics.oracles import reference_information_ratio_rolling
from tests.support.declaration import Golden, Pin, ScaleAxis
from tests.support.strategies import windows_well_conditioned
from tests.support.tolerances import TOLERANCE_RELATIVE_ROLLING_ORACLE

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


INFORMATION_RATIO_ROLLING = suite_metrics(
    factory=information_ratio_rolling,
    inputs=("returns", "benchmark"),
    params={"window": 4, "periods_per_year": 252},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    rolling_of=INFORMATION_RATIO,
    window="window",
    warmup=3,
    oracle=reference_information_ratio_rolling,
    scaling=(ScaleAxis(roles=("returns", "benchmark"), degree=0),),
    raises=(({"window": 1}, r"window must be >= 2"), ({"periods_per_year": 0}, r"periods_per_year must be >= 1")),
    conditioning=_active_windows_conditioned,
    golden=Golden(
        inputs={
            "returns": (0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.01, 0.02),
            "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018),
        },
        output=(None, None, None, 2.3539, 2.3539, 5.0387, 2.8393, 22.9129),
    ),
    pins=(
        Pin(
            label="null_in_window_is_null",
            inputs={"returns": (0.01, None, 0.03, -0.01, 0.02), "benchmark": (0.008, -0.015, 0.025, None, 0.018)},
            expected=(None, None, None, None, None),
            reason="two separate misses in two different legs poison every window=3 window over a 5-row series",
            params_override={"window": 3},
        ),
        Pin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01), "benchmark": (0.008, -0.015, 0.025, -0.008)},
            expected=(None, None, None, -3.1314621094842335e-15),
            reason="window equals series length, so only the last row is defined; the active series sums to "
            "(near) zero",
        ),
        Pin(
            label="constant_window_by_slide_is_inf",
            inputs={"returns": (1000000.0, 0.1, 0.1, 0.1, 0.1), "benchmark": (0.0, 0.0, 0.0, 0.0, 0.0)},
            expected=(None, None, 9.16515413945737, math.inf, math.inf),
            reason="once the outlier slides out, the window is bit-constant with zero tracking error, so the "
            "ratio is +inf",
            params_override={"window": 3},
        ),
        Pin(
            label="zero_tracking_error_is_inf",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01), "benchmark": (0.0, 0.0, 0.0, 0.0)},
            expected=(None, None, math.inf, math.inf),
            reason="a constant active-return window has zero tracking error with a positive mean, so the "
            "ratio is +inf — the exact core of the near-constant regime the conditioning filter "
            "excludes from the property tiers",
            params_override={"window": 3},
            covers_conditioning=True,
        ),
        Pin(
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
            reason="a window whose active returns are all exactly zero degenerates to 0/0 -> NaN even after "
            "larger active values slid out: the exact rolling mean pins the numerator to zero, so the "
            "incremental running-sum residue cannot ride above the exactly-zero tracking error as a "
            "spurious inf; the residue is bit-sensitive, so the prefix must reach the kernel with "
            "exactly these bits (a zero benchmark keeps the active leg bit-identical to the returns)",
        ),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    reference='Goodwin, T. H. (1998). "The Information Ratio." *Financial Analysts Journal*, 54(4), 34-43.',
    doi="https://doi.org/10.2469/faj.v54.n4.2196",
    wikipedia="https://en.wikipedia.org/wiki/Information_ratio",
    see_also=(
        ("information_ratio", "The whole-series reducing form."),
        ("sharpe_ratio_rolling", "The rolling total-risk analog measured against a risk-free rate."),
        ("alpha_rolling", "The rolling benchmark-active return measured per unit of beta."),
    ),
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Degenerate denominator",
            "a constant active window has zero tracking error, so the result is ``+/-inf`` (or "
            "``NaN`` when the mean active is also zero) — reported, not clipped.",
        ),
        (
            "Stability",
            "a near-flat (non-bit-identical) active-return window sits at the float-conditioning "
            "limit the documentation's *Correctness* page documents: the one-pass rolling tracking "
            "error and an exact two-pass recomputation can round a vanishing denominator apart "
            "without bound there. The bit-flat window is pinned exactly (a zero tracking error, the "
            "documented ``+/-inf`` / ``NaN``); real market windows are far from the regime.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The rolling information ratio for each row, the same length as the input. The first "
    "``window - 1`` rows are ``null`` (warm-up): the window must hold ``window`` complete "
    "pairs before a result is emitted.",
    raises_prose="ValueError: If ``window < 2``, or if ``periods_per_year < 1``.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 2``.",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
    intro_missing="A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make "
    "the handling visible:",
)
