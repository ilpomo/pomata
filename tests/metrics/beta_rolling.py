"""Declaration for ``pomata.metrics.beta_rolling`` — the regression slope over a trailing window, scale-invariant."""

import math

import polars as pl

from pomata.metrics import beta_rolling
from tests.metrics.beta import BETA
from tests.metrics.enums import BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_beta_rolling
from tests.support.declaration import Example, Golden, Pin, ScaleAxis
from tests.support.strategies import windows_well_conditioned
from tests.support.tolerances import TOLERANCE_RELATIVE_ROLLING_ORACLE

# Spec-local conditioning floor: the shared 1e-2 rejected ~19% of drawn frames while the bare slope's measured
# disagreement onset sits at var_rel ~4e-14..6e-14 (the exact bit-flat core below that is already snapped by the
# implementation's own rolling_max == rolling_min guard), so 1e-9 keeps 4-5 orders of margin above the onset while
# re-admitting the ~7 orders of well-behaved windows the shared cut threw away.
_CONDITIONING_FLOOR = 1e-9


def _windows_well_conditioned(frame: pl.DataFrame) -> bool:
    """Reject any trailing window whose benchmark variance is too near zero for the one-pass slope to track."""
    return windows_well_conditioned(frame["benchmark"].to_list(), 4, floor=_CONDITIONING_FLOOR)


BETA_ROLLING = suite_metrics(
    factory=beta_rolling,
    inputs=("returns", "benchmark"),
    params={"window": 4},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    rolling_of=BETA,
    window="window",
    warmup=3,
    oracle=reference_beta_rolling,
    scaling=(ScaleAxis(roles=("returns", "benchmark"), degree=0),),
    raises=(({"window": 1}, r"window must be >= 2"),),
    conditioning=_windows_well_conditioned,
    golden=Golden(
        inputs={
            "returns": (0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.01, 0.02),
            "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018),
        },
        output=(None, None, None, 1.2608, 1.2628, 1.2652, 1.2592, 1.0331),
    ),
    pins=(
        Pin(
            label="null_in_constant_benchmark_window_is_null",
            inputs={"returns": (0.02, None, 0.03, 0.01, 0.02), "benchmark": (0.1, 0.1, 0.1, 0.1, 0.1)},
            expected=(None, None, None, None, math.nan),
            reason="a window holding a null yields null (pairwise-complete gate) rather than the flat-benchmark NaN",
            params_override={"window": 3},
        ),
        Pin(
            label="constant_benchmark_window_is_nan",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01), "benchmark": (0.1, 0.1, 0.1, 0.1)},
            expected=(None, None, math.nan, math.nan),
            reason="a fully-defined window with an exactly-constant benchmark has zero variance, so the "
            "slope is NaN — the exact core of the near-constant regime the conditioning filter "
            "excludes from the property tiers",
            params_override={"window": 3},
            covers_conditioning=True,
        ),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    reference='Sharpe, W. F. (1964). "Capital Asset Prices: A Theory of Market Equilibrium under '
    'Conditions of Risk." *The Journal of Finance*, 19(3), 425-442.',
    doi="https://doi.org/10.1111/j.1540-6261.1964.tb02865.x",
    wikipedia="https://en.wikipedia.org/wiki/Beta_%28finance%29",
    see_also=(
        ("beta", "The whole-series reducing form."),
        ("alpha_rolling", "The benchmark-relative return built on this slope."),
        ("treynor_ratio_rolling", "The excess return per unit of this systematic risk."),
    ),
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Degenerate denominator",
            "a zero-variance window benchmark leaves the slope undefined, so the result is a ``0 / 0``, i.e. ``NaN``.",
        ),
        (
            "Stability",
            "a near-flat (non-bit-identical) benchmark window sits at the float-conditioning limit "
            "the documentation's *Correctness* page documents: the one-pass rolling covariance and an "
            "exact two-pass recomputation can round a vanishing denominator apart without bound "
            "there. The bit-flat window is guarded exactly (``NaN``); real market windows are far "
            "from the regime.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The rolling regression slope for each row, the same length as the input. The first "
    "``window - 1`` rows are ``null`` (warm-up): the window must hold ``window`` complete "
    "pairs before a result is emitted.",
    raises_prose="ValueError: If ``window < 2``.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 2``.",
    },
    examples=(
        Example(
            inputs={
                "returns": (0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.01, 0.02),
                "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018),
            },
            params={"window": 4},
            round_to=4,
        ),
        Example(
            inputs={
                "returns": (0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.01, 0.025, -0.015, 0.008, -0.005, 0.012),
                "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004, 0.012, 0.02, -0.01, 0.006, -0.004, 0.01),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("AAPL",) * 6 + ("NVDA",) * 6,
            params={"window": 4},
            round_to=4,
        ),
        Example(
            inputs={
                "returns": (None, float("nan"), 0.03, -0.02, 0.015, 0.005, -0.01, 0.02),
                "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018),
            },
            intro="A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make "
            "the handling visible:",
            params={"window": 4},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.02, None, 0.03, 0.01, 0.02), "benchmark": (0.1, 0.1, 0.1, 0.1, 0.1)},
            intro="**Degenerate denominator** — a window holding a ``null`` yields ``null`` under the "
            "pairwise-complete gate rather than the flat-benchmark ``NaN``, until the window clears "
            "and reports ``NaN``:",
            params={"window": 3},
        ),
    ),
)
