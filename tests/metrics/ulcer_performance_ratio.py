"""
Declaration for ``pomata.metrics.ulcer_performance_ratio`` — reducing, excess CAGR per unit of ulcer index, scale-
exempt.
"""

import math

import polars as pl

from pomata.metrics import cagr, ulcer_index, ulcer_performance_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_ulcer_performance_ratio
from tests.support.declaration import Example, Golden, Pin, ScaleExempt

ULCER_PERFORMANCE_RATIO = suite_metrics(
    factory=ulcer_performance_ratio,
    inputs=("equity_curve",),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.GEOMETRIC,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_ulcer_performance_ratio,
    recomposition=lambda: (
        (cagr(pl.col("equity_curve"), periods_per_year=252) - 0.0) / ulcer_index(pl.col("equity_curve"))
    ),
    scaling=ScaleExempt(
        reason="a normalized growth-factor curve run through CAGR over a scale-invariant ulcer index — "
        "neither invariant nor homogeneous"
    ),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    golden=Golden(
        inputs={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)},
        output=(1.7927,),
        params={"periods_per_year": 1},
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(math.nan,),
            reason="one observation has zero excess growth and a zero ulcer index, so the ratio is 0/0, i.e. NaN",
        ),
        Pin(
            label="no_drawdown_is_inf",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(math.inf,),
            reason="a monotonically rising curve has a zero ulcer index with positive excess growth, so the "
            "ratio is +inf",
            params_override={"periods_per_year": 1},
        ),
        Pin(
            label="no_drawdown_zero_growth_is_nan",
            inputs={"equity_curve": (1.0, 1.0, 1.0)},
            expected=(math.nan,),
            reason="a flat multi-row curve has a zero ulcer index and zero excess growth, so the ratio is a "
            "0/0, i.e. NaN — the degenerate-denominator NaN beside the +inf pin",
        ),
    ),
    reference="Martin, P. G. & McCann, B. B. (1989). *The Investor's Guide to Fidelity Funds*. Wiley.",
    wikipedia="https://en.wikipedia.org/wiki/Ulcer_index",
    see_also=(
        ("ulcer_index", "The denominator (depth-and-duration drawdown)."),
        ("pain_ratio", "The average-drawdown counterpart in the same return-to-pain family."),
        ("calmar_ratio", "The companion return-to-pain ratio scaled by the single worst drawdown."),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` equity is skipped (excluded from both the growth and the ulcer index); an "
            "all-null (or empty) series yields ``null``.",
        ),
        ("NaN", "a ``NaN`` equity propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "a single observation has zero excess growth over a zero ulcer index, so the result is a "
            "``0 / 0``, i.e. ``NaN``.",
        ),
        (
            "Degenerate denominator",
            "a monotonically non-decreasing curve has a zero ulcer index, so the ratio is ``+/-inf`` "
            "(or ``NaN`` when the excess growth is also zero) — reported, not clipped.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the ulcer performance index (one value in ``select``, one "
    "per group under ``.over``). ``null`` when there are no observations.",
    raises_prose="ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite.",
    args_prose={
        "equity_curve": "Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.",
        "risk_free_rate": "The annualized risk-free rate subtracted from the growth (default ``0.0``). Must be finite.",
    },
    example_columns={"equity_curve": "equity"},
    examples=(
        Example(
            inputs={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)}, params={"periods_per_year": 1}, round_to=4
        ),
        Example(
            inputs={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4, 1.0, 1.02, 1.01, 1.05, 1.08, 1.06, 1.12)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("A", "A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B", "B"),
            params={"periods_per_year": 1},
            round_to=4,
        ),
        Example(
            inputs={"equity_curve": (1.1, None, 1.2, 1.15, float("nan"), 1.25, 1.4)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            params={"periods_per_year": 1},
            round_to=4,
        ),
        Example(
            inputs={"equity_curve": (1.0,)},
            intro="**Insufficient sample** — a single observation has zero excess growth over a zero ulcer "
            "index, so the ratio is a ``0 / 0``, i.e. ``NaN``:",
            params={"periods_per_year": 252},
        ),
        Example(
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            intro="**Degenerate denominator** — a monotonically rising curve has a zero ulcer index with "
            "positive excess growth, so the ratio is ``+inf``:",
            params={"periods_per_year": 1},
        ),
        Example(
            inputs={"equity_curve": (1.0, 1.0, 1.0)},
            intro="**Degenerate denominator** — a flat multi-row curve has a zero ulcer index and zero "
            "excess growth, so the ratio is a ``0 / 0``, i.e. ``NaN``:",
            params={"periods_per_year": 252},
        ),
    ),
)
