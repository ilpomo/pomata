"""
Declaration for ``pomata.metrics.sharpe_ratio_rolling`` — the annualized Sharpe over a trailing window, scale-
invariant.
"""

import math

import polars as pl

from pomata.metrics import sharpe_ratio_rolling
from tests.metrics.enums import BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_sharpe_ratio_rolling
from tests.metrics.sharpe_ratio import SHARPE_RATIO
from tests.support.declaration import Example, Golden, Pin, ScaleAxis
from tests.support.strategies import windows_well_spread
from tests.support.tolerances import TOLERANCE_RELATIVE_ROLLING_ORACLE


def _windows_well_spread(frame: pl.DataFrame) -> bool:
    """
    Reject a near-constant window: the one-pass rolling std cannot track the two-pass oracle there. JUSTIFIED by
    measurement — if anything slightly PERMISSIVE: real disagreement was measured right at the cut (var_rel 1e-9,
    ~1e-4 relative deviation) and agreement only returns ~23x above it, but that sliver of admitted-and-divergent
    windows is unreachable by the property tiers' independent draws, so the shared cut is kept rather than inventing
    a spec-local widening.
    """
    return windows_well_spread(frame.to_series(0).to_list(), 3)


SHARPE_RATIO_ROLLING = suite_metrics(
    factory=sharpe_ratio_rolling,
    inputs=("returns",),
    params={"window": 3, "periods_per_year": 252, "risk_free_rate": 0.0},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    rolling_of=SHARPE_RATIO,
    window="window",
    warmup=2,
    oracle=reference_sharpe_ratio_rolling,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    raises=(
        ({"window": 1}, r"window must be >= 2"),
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -2.0}, r"risk_free_rate must be >= -1"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    conditioning=_windows_well_spread,
    golden=Golden(
        inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.025, -0.005, 0.02)},
        output=(None, None, 10.1678, -1.3977, 7.2837, 1.271, 13.1689),
    ),
    pins=(
        Pin(
            label="zero_volatility",
            inputs={"returns": (0.5, 0.5, 0.5, 0.5)},
            expected=(None, None, math.inf, math.inf),
            reason="a constant window has zero dispersion with a positive mean, so the ratio is +inf — the "
            "exact core of the near-constant regime the conditioning filter excludes from the "
            "property tiers",
            params_override={"window": 3},
            covers_conditioning=True,
        ),
        Pin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(None, None, None, None, 4.593220484431882),
            reason="when the window equals the series length only the last row is defined",
            params_override={"window": 5},
        ),
        Pin(
            label="flat_zero_excess_window_by_slide_is_nan",
            inputs={"returns": (-0.3233, -0.6457, 0.0, 0.4404, 0.0, 0.0, 0.0, 0.0)},
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
            reason="a window whose excess returns are all exactly zero degenerates to 0/0 -> NaN even after "
            "larger values slid out: the exact rolling mean pins the numerator to zero, so the "
            "incremental running-sum residue cannot ride above the exactly-zero dispersion as a "
            "spurious inf; the residue is bit-sensitive, so the prefix must reach the kernel with "
            "exactly these bits",
            params_override={"window": 4},
        ),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    reference='Sharpe, W. F. (1994). "The Sharpe Ratio." *The Journal of Portfolio Management*, 21(1), 49-58.',
    doi="https://doi.org/10.3905/jpm.1994.409501",
    wikipedia="https://en.wikipedia.org/wiki/Sharpe_ratio",
    see_also=(
        ("sharpe_ratio", "The whole-series reducing form."),
        ("volatility_rolling", "The denominator."),
        ("sortino_ratio_rolling", "The downside-only rolling counterpart."),
    ),
    opener_override="Each window matches an independent reference oracle (the reducing :func:`sharpe_ratio` "
    "recomputed over the window).",
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Degenerate denominator",
            "a constant window has zero dispersion, so the ratio is ``+/-inf`` (or ``NaN`` when the "
            "mean excess is also zero, the exact-zero rolling mean pinning the numerator so no "
            "slid-out residue rides above it) — reported, not clipped.",
        ),
        (
            "Stability",
            "a near-constant (non-bit-identical) window sits at the float-conditioning limit the "
            "documentation's *Correctness* page documents: the one-pass rolling standard deviation "
            "and a two-pass recomputation can round a vanishing dispersion apart there. The "
            "bit-constant window is pinned exactly (the ``+/-inf`` / ``NaN`` above); real return "
            "windows are far from the regime.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The rolling Sharpe ratio for each row, the same length as the input. The first ``window "
    "- 1`` rows are ``null`` (warm-up): the window must hold ``window`` non-null values "
    "before a result is emitted.",
    raises_prose="ValueError: If ``window < 2``, ``periods_per_year < 1``, or if ``risk_free_rate`` is not "
    "finite or is ``< -1``.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 2``.",
        "risk_free_rate": "The annualized risk-free rate, converted to a per-period rate geometrically (default "
        "``0.0``). Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + "
        "risk_free_rate >= 0``).",
    },
    examples=(
        Example(
            inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.025, -0.005, 0.02)},
            params={"window": 3, "periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={
                "returns": (
                    0.03,
                    -0.01,
                    0.02,
                    -0.015,
                    0.025,
                    -0.005,
                    0.02,
                    0.02,
                    -0.005,
                    0.015,
                    -0.01,
                    0.025,
                    0.0,
                    -0.012,
                )
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("A", "A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B", "B"),
            params={"window": 3, "periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.03, None, 0.02, -0.015, 0.025, float("nan"), 0.02, -0.01, 0.015)},
            intro="A ``null`` (which voids every window that spans it) and a ``NaN`` (which propagates to "
            "its windows) make the missing-data handling visible:",
            params={"window": 3, "periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.5, 0.5, 0.5, 0.5)},
            intro="**Degenerate denominator** — a constant window has zero dispersion with a positive mean, "
            "so the ratio is ``+inf``:",
            params={"window": 3, "periods_per_year": 252},
        ),
        Example(
            inputs={"returns": (-0.3233, -0.6457, 0.0, 0.4404, 0.0, 0.0, 0.0, 0.0)},
            intro="**Degenerate denominator** — a trailing window of exact zeros, reached after larger "
            "values slide out, has zero mean and zero dispersion, so the ratio is a ``0 / 0``, i.e. "
            "``NaN``:",
            params={"window": 4, "periods_per_year": 252},
            round_to=4,
        ),
    ),
)
