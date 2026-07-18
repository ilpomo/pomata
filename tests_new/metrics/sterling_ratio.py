"""
Declaration for ``pomata.metrics.sterling_ratio`` — reducing, excess CAGR per cushioned average drawdown, scale-
exempt.
"""

import math

import polars as pl

from pomata.metrics import cagr, pain_index, sterling_ratio
from tests_new.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests_new.metrics.harness import suite_metrics
from tests_new.metrics.oracles import reference_sterling_ratio
from tests_new.support.declaration import Golden, Pin, ScaleExempt

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
)
