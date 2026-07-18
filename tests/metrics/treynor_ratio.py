"""
Declaration for ``pomata.metrics.treynor_ratio`` — reducing, annualized excess return per unit of beta, degree-1 at
rf=0.
"""

import math
from collections.abc import Sequence

import polars as pl

from pomata.metrics import beta, treynor_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_treynor_ratio
from tests.support.declaration import Golden, Pin, ScaleAxis

# Spec-local beta floor. Measured: impl and oracle agree at 1e-13..1e-16 relative deviation everywhere down to
# |beta| = 1e-5, and real disagreement starts only at |beta| ~3e-8..1e-7, so a floor of 1e-5 keeps a ~2-3 order
# margin above the crossing. The vanishing-slope quotient below is the only regime treynor's ratio genuinely needs.
_BETA_FLOOR = 1e-5


def _beta_nondegenerate(returns: Sequence[float | None], benchmark: Sequence[float | None]) -> bool:
    """Whether the regression slope is bounded away from zero: treynor divides the excess mean by it."""
    pairs = [
        (value_returns, value_benchmark)
        for value_returns, value_benchmark in zip(returns, benchmark, strict=True)
        if value_returns is not None
        and value_benchmark is not None
        and not math.isnan(value_returns)
        and not math.isnan(value_benchmark)
    ]
    if len(pairs) < 2:
        return True
    complete_returns = [value for value, _ in pairs]
    complete_bench = [value for _, value in pairs]
    mean_returns = sum(complete_returns) / len(pairs)
    mean_benchmark = sum(complete_bench) / len(pairs)
    variance = sum((value - mean_benchmark) ** 2 for value in complete_bench) / len(pairs)
    if variance == 0.0:
        return False
    covariance = sum(
        (value_returns - mean_returns) * (value_benchmark - mean_benchmark)
        for value_returns, value_benchmark in zip(complete_returns, complete_bench, strict=True)
    ) / len(pairs)
    return abs(covariance / variance) >= _BETA_FLOOR


def _treynor_conditioning(frame: pl.DataFrame) -> bool:
    """A beta bounded away from zero — the one regime treynor's quotient genuinely needs."""
    return _beta_nondegenerate(frame["returns"].to_list(), frame["benchmark"].to_list())


def _treynor_component() -> pl.Expr:
    """Treynor recomposed from the public ``beta`` factory at the spec's default params (rf 0.0, 252 periods)."""
    rf_period = math.pow(1.0 + 0.0, 1.0 / 252) - 1.0
    return (pl.col("returns") - rf_period).mean() * 252 / beta(pl.col("returns"), pl.col("benchmark"))


TREYNOR_RATIO = suite_metrics(
    factory=treynor_ratio,
    inputs=("returns", "benchmark"),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.LINEAR,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_treynor_ratio,
    recomposition=_treynor_component,
    scaling=(ScaleAxis(roles=("returns", "benchmark"), degree=1),),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -2.0}, r"risk_free_rate must be >= -1"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    conditioning=_treynor_conditioning,
    golden=Golden(
        inputs={
            "returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018),
            "benchmark": (0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015),
        },
        output=(0.3083,),
        params={"risk_free_rate": 0.02},
    ),
    pins=(
        Pin(
            label="single_pair",
            inputs={"returns": (0.05,), "benchmark": (0.04,)},
            expected=(None,),
            reason="a single complete pair yields null — the regression slope needs two observations",
        ),
        Pin(
            label="zero_beta_is_inf",
            inputs={"returns": (3.0, 3.0, 1.0, 1.0), "benchmark": (1.0, -1.0, 1.0, -1.0)},
            expected=(math.inf,),
            reason="a zero beta (uncorrelated returns) with a positive excess return gives +inf, reported "
            "not clipped — the exact core of the vanishing-slope regime the conditioning filter "
            "excludes from the property tiers",
            covers_conditioning=True,
        ),
        Pin(
            label="small_beta_matches_reference",
            inputs={"returns": (3.001, 2.999, 1.001, 0.999), "benchmark": (1.0, -1.0, 1.0, -1.0)},
            expected=(504000.0000000275,),
            reason="an exact-by-construction beta of 0.001, well inside the small-slope band the 1e-5 "
            "conditioning floor admits: the quotient matches the oracle to ~5.6e-14 relative "
            "deviation, holding the fact that a tiny-but-clean beta stays well-conditioned",
        ),
        Pin(
            label="constant_benchmark_is_nan_0_1",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (0.1, 0.1, 0.1)},
            expected=(math.nan,),
            reason="a constant benchmark makes the embedded beta NaN, so the excess-over-beta ratio is NaN",
        ),
        Pin(
            label="constant_benchmark_is_nan_1_3",
            inputs={
                "returns": (0.01, -0.02, 0.03),
                "benchmark": (0.3333333333333333, 0.3333333333333333, 0.3333333333333333),
            },
            expected=(math.nan,),
            reason="the same guard at a repeating-decimal constant magnitude",
        ),
        Pin(
            label="constant_benchmark_is_nan_0_123456789",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (0.123456789, 0.123456789, 0.123456789)},
            expected=(math.nan,),
            reason="the same guard at a third magnitude, firing regardless of the constant",
        ),
    ),
    reference='Treynor, J. L. (1965). "How to Rate Management of Investment Funds." *Harvard Business '
    "Review*, 43(1), 63-75.",
    wikipedia="https://en.wikipedia.org/wiki/Treynor_ratio",
    see_also=(
        ("beta", "The denominator (systematic risk)."),
        ("sharpe_ratio", "The total-risk analog."),
        ("alpha", "The benchmark-relative excess built on the same beta."),
        ("treynor_ratio_rolling", "The rolling (windowed) form."),
    ),
    bullets=(
        ("Null", "an observation is used only where both legs are present; a ``null`` in either drops that pair."),
        ("NaN", "a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "fewer than two complete pairs leaves the regression slope undefined, so the result is ``null``.",
        ),
        (
            "Degenerate denominator",
            "a zero beta gives ``+/-inf`` (or ``NaN`` when the excess return is also zero) — "
            "reported, not clipped; a zero-variance benchmark instead makes :func:`beta` ``NaN``, "
            "which propagates here.",
        ),
        (
            "Stability",
            "a beta bounded away from zero is the one regime the excess-over-beta quotient genuinely "
            "needs: as the slope vanishes the division amplifies rounding without bound, so a "
            "near-zero beta sits at the float-conditioning limit the documentation's *Correctness* "
            "page documents. The exact zero-beta case is guarded (``+/-inf``); real market betas are "
            "far from the regime.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the annualized Treynor ratio (one value in ``select``, one "
    "per group under ``.over``). ``null`` when fewer than two complete pairs are present.",
    raises_prose="ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite or is ``< -1``.",
    args_prose={
        "risk_free_rate": "The annualized risk-free rate, converted to a per-period rate geometrically (default "
        "``0.0``). Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + "
        "risk_free_rate >= 0``).",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
    intro_missing="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
    "handling visible:",
)
