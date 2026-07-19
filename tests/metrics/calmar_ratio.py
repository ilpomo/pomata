"""Declaration for ``pomata.metrics.calmar_ratio`` — reducing, CAGR per unit of maximum drawdown, scale-exempt."""

import math

import polars as pl

from pomata.metrics import cagr, calmar_ratio, max_drawdown
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_calmar_ratio
from tests.support.declaration import Example, Golden, Pin, ScaleExempt

CALMAR_RATIO = suite_metrics(
    factory=calmar_ratio,
    inputs=("equity_curve",),
    params={"periods_per_year": 252},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.GEOMETRIC,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_calmar_ratio,
    recomposition=lambda: (
        cagr(pl.col("equity_curve"), periods_per_year=252) / max_drawdown(pl.col("equity_curve")).abs()
    ),
    scaling=ScaleExempt(
        reason="a normalized growth-factor curve run through CAGR over a scale-invariant drawdown "
        "magnitude — neither homogeneous nor invariant"
    ),
    raises=(({"periods_per_year": 0}, r"periods_per_year must be >= 1"),),
    golden=Golden(
        inputs={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)},
        output=(1.0833,),
        params={"periods_per_year": 1},
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(math.nan,),
            reason="a one-element series has zero growth and zero drawdown, so the ratio is 0/0, i.e. NaN ",
        ),
        Pin(
            label="no_drawdown_is_inf",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(math.inf,),
            reason="a monotonically rising curve has zero maximum drawdown with positive growth, so the ratio is +inf",
            params_override={"periods_per_year": 1},
        ),
        Pin(
            label="no_drawdown_zero_growth_is_nan",
            inputs={"equity_curve": (1.0, 1.0, 1.0)},
            expected=(math.nan,),
            reason="a flat multi-row curve has zero maximum drawdown and zero growth, so the ratio is a 0/0, "
            "i.e. NaN — the degenerate-denominator NaN beside the +inf pin",
        ),
    ),
    reference='Young, T. W. (1991). "Calmar Ratio: A Smoother Tool." *Futures Magazine*.',
    wikipedia="https://en.wikipedia.org/wiki/Calmar_ratio",
    see_also=(
        ("cagr", "The numerator (annualized growth)."),
        ("max_drawdown", "The denominator (worst decline)."),
        ("recovery_ratio", "The same worst-drawdown denominator with a total-return numerator."),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` equity is skipped (excluded from both the growth and the drawdown); an "
            "all-null (or empty) series yields ``null``.",
        ),
        ("NaN", "a ``NaN`` equity propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "a single observation has zero growth over zero maximum drawdown, so the result is a ``0 "
            "/ 0``, i.e. ``NaN``.",
        ),
        (
            "Degenerate denominator",
            "a monotonically non-decreasing curve has zero maximum drawdown, so the ratio is "
            "``+/-inf`` (or ``NaN`` when the growth is also zero) — reported, not clipped.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the Calmar ratio (one value in ``select``, one per group "
    "under ``.over``). ``null`` when there are no observations.",
    raises_prose="ValueError: If ``periods_per_year < 1``.",
    args_prose={
        "equity_curve": "Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.",
    },
    example_columns={"equity_curve": "equity"},
    examples=(
        Example(
            inputs={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)}, params={"periods_per_year": 1}, round_to=4
        ),
        Example(
            inputs={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4, 1.0, 1.02, 1.01, 1.05, 1.08, 1.06, 1.12)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("AAPL",) * 7 + ("NVDA",) * 7,
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
            intro="**Insufficient sample** — a single observation has zero growth and zero maximum "
            "drawdown, so the ratio is a ``0 / 0``, i.e. ``NaN``:",
            params={"periods_per_year": 252},
        ),
        Example(
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            intro="**Degenerate denominator** — a monotonically rising curve has zero maximum drawdown with "
            "positive growth, so the ratio is ``+inf``:",
            params={"periods_per_year": 1},
        ),
        Example(
            inputs={"equity_curve": (1.0, 1.0, 1.0)},
            intro="**Degenerate denominator** — a flat multi-row curve has zero maximum drawdown and zero "
            "growth, so the ratio is a ``0 / 0``, i.e. ``NaN``:",
            params={"periods_per_year": 252},
        ),
    ),
)
