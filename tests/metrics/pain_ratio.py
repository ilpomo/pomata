"""Declaration for ``pomata.metrics.pain_ratio`` — reducing, excess CAGR per unit of pain index, scale-exempt."""

import math

import polars as pl

from pomata.metrics import cagr, pain_index, pain_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_pain_ratio
from tests.support.declaration import Example, Golden, Pin, ScaleExempt

PAIN_RATIO = suite_metrics(
    factory=pain_ratio,
    inputs=("equity_curve",),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.GEOMETRIC,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_pain_ratio,
    recomposition=lambda: (
        (cagr(pl.col("equity_curve"), periods_per_year=252) - 0.0) / pain_index(pl.col("equity_curve"))
    ),
    scaling=ScaleExempt(
        reason="a normalized growth-factor curve (excess CAGR over the average-drawdown pain index) — "
        "neither homogeneous nor invariant"
    ),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    golden=Golden(
        inputs={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)},
        output=(2.7447,),
        params={"periods_per_year": 1},
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(math.nan,),
            reason="one observation has zero excess growth and a zero pain index, so the ratio is the 0/0 NaN branch",
        ),
        Pin(
            label="no_drawdown_is_inf",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(math.inf,),
            reason="a monotonically rising curve has a zero pain index with positive excess growth, so the "
            "ratio is +inf",
            params_override={"periods_per_year": 1},
        ),
        Pin(
            label="no_drawdown_zero_growth_is_nan",
            inputs={"equity_curve": (1.0, 1.0, 1.0)},
            expected=(math.nan,),
            reason="a flat multi-row curve has a zero pain index and zero excess growth, so the ratio is a "
            "0/0, i.e. NaN — the degenerate-denominator NaN beside the +inf pin",
        ),
    ),
    reference='Becker, T. (2006). "The Pain Index and Pain Ratio." *Zephyr Associates*.',
    see_also=(
        ("pain_index", "The denominator (average drawdown depth)."),
        ("sterling_ratio", "The same average-drawdown denominator offset by a fixed cushion."),
        ("ulcer_performance_ratio", "The root-mean-square-drawdown counterpart."),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` equity is skipped (excluded from both the growth and the pain index); an "
            "all-null (or empty) series yields ``null``.",
        ),
        ("NaN", "a ``NaN`` equity propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "a single observation has zero excess growth over a zero pain index, so the result is a "
            "``0 / 0``, i.e. ``NaN``.",
        ),
        (
            "Degenerate denominator",
            "a monotonically non-decreasing curve has a zero pain index, so the ratio is ``+/-inf`` "
            "(or ``NaN`` when the excess growth is also zero) — reported, not clipped.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the pain ratio (one value in ``select``, one per group under "
    "``.over``). ``null`` when there are no observations.",
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
            intro="**Insufficient sample** — a single observation has zero excess growth over a zero pain "
            "index, so the ratio is a ``0 / 0``, i.e. ``NaN``:",
            params={"periods_per_year": 252},
        ),
        Example(
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            intro="**Degenerate denominator** — a monotonically rising curve has a zero pain index with "
            "positive excess growth, so the ratio is ``+inf``:",
            params={"periods_per_year": 1},
        ),
        Example(
            inputs={"equity_curve": (1.0, 1.0, 1.0)},
            intro="**Degenerate denominator** — a flat multi-row curve has a zero pain index and zero "
            "excess growth, so the ratio is a ``0 / 0``, i.e. ``NaN``:",
            params={"periods_per_year": 252},
        ),
    ),
)
