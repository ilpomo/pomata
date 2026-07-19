"""
Declaration for ``pomata.metrics.sterling_ratio`` — reducing, excess CAGR per cushioned average drawdown, scale-
exempt.
"""

import math

import polars as pl

from pomata.metrics import cagr, pain_index, sterling_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_sterling_ratio
from tests.support.declaration import Example, Golden, Pin, ScaleExempt

STERLING_RATIO = suite_metrics(
    factory=sterling_ratio,
    inputs=("equity_curve",),
    params={"periods_per_year": 252, "risk_free_rate": 0.0, "excess": 0.1},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.GEOMETRIC,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_sterling_ratio,
    recomposition=lambda: (
        (cagr(pl.col("equity_curve"), periods_per_year=252) - 0.0) / (pain_index(pl.col("equity_curve")) + 0.10)
    ),
    scaling=ScaleExempt(
        reason="a normalized growth factor over a scale-invariant average drawdown — neither homogeneous nor invariant"
    ),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
        ({"excess": math.nan}, r"excess must be a finite number"),
        ({"excess": math.inf}, r"excess must be a finite number"),
        ({"excess": -math.inf}, r"excess must be a finite number"),
        ({"excess": -0.1}, r"excess must be a finite number >= 0"),
    ),
    golden=Golden(
        inputs={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)},
        output=(0.4175,),
        params={"periods_per_year": 1},
    ),
    pins=(
        Pin(
            label="no_drawdown_is_zero",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a flat single-period growth has zero drawdown and zero excess growth, so the ratio is "
            "exactly 0 (the excess cushion keeps the denominator finite)",
        ),
        Pin(
            label="cushioned_no_drawdown_is_finite",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(0.656022367666107,),
            reason="a monotonically rising curve stays FINITE here — the excess cushion keeps the "
            "denominator positive where the cushion-less burke / calmar / pain / recovery / "
            "ulcer_performance twins diverge to +inf: sterling's distinguishing behavior, pinned like "
            "the twins pin theirs",
            params_override={"periods_per_year": 1},
        ),
        Pin(
            label="zero_cushion_no_drawdown_is_inf",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(math.inf,),
            reason="a zero excess cushion on a drawdown-free rising curve leaves a zero denominator with "
            "positive growth, so the ratio is +inf",
            params_override={"periods_per_year": 1, "excess": 0.0},
        ),
        Pin(
            label="zero_cushion_no_drawdown_zero_growth_is_nan",
            inputs={"equity_curve": (1.0, 1.0, 1.0)},
            expected=(math.nan,),
            reason="a zero excess cushion on a flat curve leaves a zero denominator and zero excess growth, "
            "so the ratio is a 0/0, i.e. NaN — the degenerate-denominator NaN beside the +inf pin",
            params_override={"periods_per_year": 1, "excess": 0.0},
        ),
    ),
    reference='Kestner, L. N. (1996). "Getting a Handle on True Performance." *Futures Magazine*.',
    wikipedia="https://en.wikipedia.org/wiki/Sterling_ratio",
    see_also=(
        ("pain_index", "The average drawdown in the denominator."),
        ("pain_ratio", "The same average-drawdown denominator without the cushion."),
        ("calmar_ratio", "The single-worst-drawdown counterpart."),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` equity is skipped (excluded from both the growth and the average drawdown); "
            "an all-null (or empty) series yields ``null``.",
        ),
        ("NaN", "a ``NaN`` equity propagates, yielding ``NaN``."),
        (
            "Degenerate denominator",
            "with the default positive cushion the denominator never vanishes (a drawdown-free curve "
            "gives exactly ``0`` when the excess growth is also zero, a finite ratio otherwise); only "
            "an ``excess`` of zero on a drawdown-free curve gives ``+/-inf`` (or ``NaN`` when the "
            "excess growth is also zero) — reported, not clipped.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the Sterling ratio (one value in ``select``, one per group "
    "under ``.over``). ``null`` when there are no observations.",
    raises_prose="ValueError: If ``periods_per_year < 1``, if ``risk_free_rate`` is not finite, or if "
    "``excess`` is not a finite number ``>= 0``.",
    args_prose={
        "equity_curve": "Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.",
        "risk_free_rate": "The annualized risk-free rate subtracted from the growth (default ``0.0``). Must be finite.",
        "excess": "The fixed cushion added to the average drawdown denominator (default ``0.10``). Must be "
        "a finite number ``>= 0``.",
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
            intro="**Degenerate denominator** — a flat single-period curve has zero drawdown and zero "
            "excess growth over the default cushion, so the ratio is exactly ``0``:",
            params={"periods_per_year": 252},
        ),
        Example(
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            intro="**Degenerate denominator** — a zero cushion on a drawdown-free rising curve leaves a "
            "zero denominator with positive growth, so the ratio is ``+inf``:",
            params={"periods_per_year": 1, "excess": 0.0},
        ),
        Example(
            inputs={"equity_curve": (1.0, 1.0, 1.0)},
            intro="**Degenerate denominator** — a zero cushion on a flat curve leaves a zero denominator "
            "and zero excess growth, so the ratio is a ``0 / 0``, i.e. ``NaN``:",
            params={"periods_per_year": 1, "excess": 0.0},
        ),
    ),
)
