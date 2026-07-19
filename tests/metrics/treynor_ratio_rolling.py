"""
Declaration for ``pomata.metrics.treynor_ratio_rolling`` — annualized excess return per rolling beta, degree-1 at
rf=0.
"""

import math
from collections.abc import Sequence

import polars as pl

from pomata.metrics import treynor_ratio_rolling
from tests.metrics.enums import BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_treynor_ratio_rolling
from tests.metrics.treynor_ratio import TREYNOR_RATIO
from tests.support.declaration import Example, Golden, Pin, ScaleAxis
from tests.support.tolerances import TOLERANCE_RELATIVE_ROLLING_ORACLE

# Spec-local conditioning floors. Measured: impl-vs-oracle agreement holds down to 1e-6 on BOTH axes (benchmark
# var_rel and |beta|) with the first breach at ~1e-8, so 1e-5 keeps a 10x margin above the last verified-agreeing
# point on each axis.
_VARIANCE_FLOOR = 1e-5


_BETA_FLOOR = 1e-5


def _treynor_windows_conditioned(
    returns: Sequence[float | None], benchmark: Sequence[float | None], window: int
) -> bool:
    """Whether every window has a well-conditioned benchmark variance and a slope bounded away from zero."""
    for index in range(window - 1, len(benchmark)):
        pairs = [
            (value_returns, value_benchmark)
            for value_returns, value_benchmark in zip(
                returns[index - window + 1 : index + 1], benchmark[index - window + 1 : index + 1], strict=True
            )
            if value_returns is not None
            and value_benchmark is not None
            and not math.isnan(value_returns)
            and not math.isnan(value_benchmark)
        ]
        if len(pairs) < 2:
            continue
        window_returns = [pair[0] for pair in pairs]
        window_benchmark = [pair[1] for pair in pairs]
        mean_returns = sum(window_returns) / len(pairs)
        mean_benchmark = sum(window_benchmark) / len(pairs)
        variance = sum((value - mean_benchmark) ** 2 for value in window_benchmark) / len(pairs)
        scale = max(abs(value) for value in window_benchmark) or 1.0
        if variance <= scale * scale * _VARIANCE_FLOOR:
            return False
        covariance = sum(
            (value_returns - mean_returns) * (value_benchmark - mean_benchmark)
            for value_returns, value_benchmark in zip(window_returns, window_benchmark, strict=True)
        ) / len(pairs)
        if abs(covariance / variance) < _BETA_FLOOR:
            return False
    return True


def _treynor_conditioning(frame: pl.DataFrame) -> bool:
    """Every window well-conditioned for the embedded slope — the regime treynor's rolling quotient needs."""
    return _treynor_windows_conditioned(frame["returns"].to_list(), frame["benchmark"].to_list(), 4)


TREYNOR_RATIO_ROLLING = suite_metrics(
    factory=treynor_ratio_rolling,
    inputs=("returns", "benchmark"),
    params={"window": 4, "periods_per_year": 252},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    rolling_of=TREYNOR_RATIO,
    window="window",
    warmup=3,
    oracle=reference_treynor_ratio_rolling,
    scaling=(ScaleAxis(roles=("returns", "benchmark"), degree=1),),
    raises=(
        ({"window": 1}, r"window must be >= 2"),
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -2.0}, r"risk_free_rate must be >= -1"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    conditioning=_treynor_conditioning,
    golden=Golden(
        inputs={
            "returns": (0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.01, 0.02),
            "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018),
        },
        output=(None, None, None, 0.9993, 0.7483, 1.4938, -0.5003, 1.8295),
        params={"window": 4, "periods_per_year": 252},
    ),
    pins=(
        Pin(
            label="null_in_constant_benchmark_window",
            inputs={"returns": (0.02, None, 0.03, 0.01, 0.02), "benchmark": (0.1, 0.1, 0.1, 0.1, 0.1)},
            expected=(None, None, None, None, math.nan),
            reason="a null in a window yields null (pairwise-complete gate) before the constant-benchmark NaN branch",
            params_override={"window": 3},
        ),
        Pin(
            label="constant_benchmark_window_is_nan",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01), "benchmark": (0.1, 0.1, 0.1, 0.1)},
            expected=(None, None, math.nan, math.nan),
            reason="a window whose benchmark is exactly constant makes the embedded slope NaN — the exact "
            "core of the near-constant regime the filter's variance clause excludes from the property "
            "tiers",
            params_override={"window": 3},
            covers_conditioning=True,
        ),
        Pin(
            label="zero_beta_window_is_inf",
            inputs={"returns": (3.0, 3.0, 1.0, 1.0), "benchmark": (1.0, -1.0, 1.0, -1.0)},
            expected=(None, None, None, math.inf),
            reason="a zero-beta window with a positive excess return gives +inf, reported not clipped — the "
            "exact core of the vanishing-slope regime the filter's beta clause excludes from the "
            "property tiers",
            params_override={"window": 4},
            covers_conditioning=True,
        ),
        Pin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02), "benchmark": (0.008, -0.015, 0.025, -0.008, 0.018)},
            expected=(None, None, None, None, 1.2350516405135523),
            reason="window equals series length, so only the last row is defined",
            params_override={"window": 5},
        ),
        Pin(
            label="matches_reference_with_risk_free_rate",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02), "benchmark": (0.008, -0.015, 0.025, -0.008, 0.018)},
            expected=(None, None, 1.324869757686746, -0.015994608829144833, 2.7922196417697887),
            reason="a non-default risk-free rate shifts every window's excess return through the geometric "
            "per-period conversion — the rate leg pinned on hand-checked values",
            params_override={"window": 3, "risk_free_rate": 0.02},
        ),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    reference='Treynor, J. L. (1965). "How to Rate Management of Investment Funds." *Harvard Business '
    "Review*, 43(1), 63-75.",
    wikipedia="https://en.wikipedia.org/wiki/Treynor_ratio",
    see_also=(
        ("treynor_ratio", "The whole-series reducing form."),
        ("beta_rolling", "The denominator (systematic risk)."),
        ("alpha_rolling", "The rolling benchmark-relative excess built on the same slope."),
    ),
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Degenerate denominator",
            "a window whose slope is zero gives ``+/-inf`` (or ``NaN``) — reported, not clipped; a "
            "zero-variance benchmark window instead makes the slope ``NaN``, which propagates here.",
        ),
        (
            "Stability",
            "a near-flat (non-bit-identical) benchmark window sits at the float-conditioning limit "
            "the documentation's *Correctness* page documents: the one-pass rolling slope and an "
            "exact two-pass recomputation can round a vanishing benchmark variance — and with it the "
            "``beta`` divisor — apart without bound there. The bit-flat window is guarded exactly "
            "(``NaN``); real market windows are far from the regime.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The rolling Treynor ratio for each row, the same length as the input. The first ``window "
    "- 1`` rows are ``null`` (warm-up): the window must hold ``window`` complete pairs before "
    "a result is emitted.",
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
            inputs={
                "returns": (0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.01, 0.02),
                "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018),
            },
            params={"window": 4, "periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={
                "returns": (0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.01, 0.025, -0.015, 0.008, -0.005, 0.012),
                "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004, 0.012, 0.02, -0.01, 0.006, -0.004, 0.01),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("AAPL",) * 6 + ("NVDA",) * 6,
            params={"window": 4, "periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={
                "returns": (None, float("nan"), 0.03, -0.02, 0.015, 0.005, -0.01, 0.02),
                "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018),
            },
            intro="A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make "
            "the handling visible:",
            params={"window": 4, "periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.02, None, 0.03, 0.01, 0.02), "benchmark": (0.1, 0.1, 0.1, 0.1, 0.1)},
            intro="**Degenerate denominator** — a ``null`` in a window yields ``null`` under the "
            "pairwise-complete gate before the constant-benchmark ``NaN`` branch, so the result stays "
            "``null`` until the window clears and then reports ``NaN``:",
            params={"window": 3, "periods_per_year": 252},
        ),
        Example(
            inputs={"returns": (3.0, 3.0, 1.0, 1.0), "benchmark": (1.0, -1.0, 1.0, -1.0)},
            intro="**Degenerate denominator** — a zero-beta window with a positive excess return gives "
            "``+inf``, reported not clipped:",
            params={"window": 4, "periods_per_year": 252},
        ),
    ),
)
