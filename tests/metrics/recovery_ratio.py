"""
Declaration for ``pomata.metrics.recovery_ratio`` — reducing, total return per unit of maximum drawdown, scale-
exempt.
"""

import math

import polars as pl

from pomata.metrics import max_drawdown, recovery_ratio, total_return
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_recovery_ratio
from tests.support.declaration import Golden, Pin, ScaleExempt

RECOVERY_RATIO = suite_metrics(
    factory=recovery_ratio,
    inputs=("equity_curve",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_recovery_ratio,
    recomposition=lambda: total_return(pl.col("equity_curve")) / max_drawdown(pl.col("equity_curve")).abs(),
    scaling=ScaleExempt(
        reason="a normalized growth-factor total return over a scale-invariant max-drawdown magnitude — "
        "neither invariant nor homogeneous"
    ),
    golden=Golden(inputs={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)}, output=(8.8,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(math.nan,),
            reason="a one-element series has zero growth and zero drawdown, so the ratio is 0/0, i.e. NaN",
        ),
        Pin(
            label="no_drawdown_is_inf",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(math.inf,),
            reason="a monotonically rising curve has zero maximum drawdown with positive growth, so the ratio is +inf",
        ),
        Pin(
            label="losing_curve_is_negative",
            inputs={"equity_curve": (1.0, 0.9, 0.95, 0.7)},
            expected=(-1.0,),
            reason="a curve ending below its start reports a negative recovery factor: the total-return "
            "numerator keeps its sign over the drawdown magnitude",
        ),
        Pin(
            label="no_drawdown_zero_growth_is_nan",
            inputs={"equity_curve": (1.0, 1.0, 1.0)},
            expected=(math.nan,),
            reason="a flat multi-row curve has zero maximum drawdown and zero total return, so the ratio is "
            "a 0/0, i.e. NaN — the degenerate-denominator NaN beside the +inf pin",
        ),
    ),
    reference="Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies* (2nd ed.). Wiley.",
    see_also=(
        ("total_return", "The numerator (overall growth)."),
        ("max_drawdown", "The denominator (worst decline)."),
        ("calmar_ratio", "The annualized-growth counterpart over the same drawdown."),
    ),
    note_extension="\n\n"
    "Only the drawdown denominator is taken in magnitude; the total-return numerator keeps "
    "its sign, so a losing curve (a negative total return) reports a negative recovery "
    "factor.",
    bullets=(
        ("Null", "a ``null`` equity is skipped; an all-null (or empty) series yields ``null``."),
        ("NaN", "a ``NaN`` equity propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "a single observation has zero total return over zero maximum drawdown, so the result is "
            "a ``0 / 0``, i.e. ``NaN``.",
        ),
        (
            "Degenerate denominator",
            "a monotonically non-decreasing curve has zero maximum drawdown, so the ratio is "
            "``+/-inf`` with the sign of the total return (or ``NaN`` when the total return is also "
            "zero) — reported, not clipped.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the recovery factor (one value in ``select``, one per group "
    "under ``.over``). ``null`` when there are no observations.",
    args_prose={
        "equity_curve": "Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
    intro_missing="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
    "handling visible:",
)
