"""Spec for ``pomata.metrics.sterling_ratio`` — reducing, excess CAGR per cushioned average drawdown, scale-exempt."""

import math

import polars as pl
from tests.metrics.oracles import sterling_ratio_reference
from tests.support.spec import ScaleExempt, Shape, Spec, SpecPin

from pomata.metrics import cagr, pain_index, sterling_ratio

STERLING_RATIO = Spec(
    factory=sterling_ratio,
    inputs=("equity_curve",),
    params={"periods_per_year": 252, "risk_free_rate": 0.0, "excess": 0.10},
    shape=Shape.REDUCING,
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
    oracle=sterling_ratio_reference,
    # A normalized growth factor over a scale-invariant average drawdown — neither homogeneous nor invariant
    #
    scale=ScaleExempt(
        reason="a normalized growth factor over a scale-invariant average drawdown — neither homogeneous nor invariant"
    ),
    golden_params={"periods_per_year": 1},
    golden_input={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)},
    golden_output=(0.4175,),
    component_expr=lambda: (
        (cagr(pl.col("equity_curve"), periods_per_year=252) - 0.0) / (pain_index(pl.col("equity_curve")) + 0.10)
    ),
    pins=(
        SpecPin(
            label="no_drawdown_is_zero",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a flat single-period growth has zero drawdown and zero excess growth, so the ratio is exactly 0 "
            "(the excess cushion keeps the denominator finite) "
            "",
        ),
        SpecPin(
            label="cushioned_no_drawdown_is_finite",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            params_override={"periods_per_year": 1},
            expected=(0.656022367666107,),
            reason="a monotonically rising curve stays FINITE here — the excess cushion keeps the denominator "
            "positive where the cushion-less burke / calmar / pain / recovery / ulcer_performance twins diverge "
            "to +inf: sterling's distinguishing behavior, pinned like the twins pin theirs",
        ),
        SpecPin(
            label="zero_cushion_no_drawdown_is_inf",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            params_override={"periods_per_year": 1, "excess": 0.0},
            expected=(math.inf,),
            reason="a zero excess cushion on a drawdown-free rising curve leaves a zero denominator with positive "
            "growth, so the ratio is +inf",
        ),
        SpecPin(
            label="zero_cushion_no_drawdown_zero_growth_is_nan",
            inputs={"equity_curve": (1.0, 1.0, 1.0)},
            params_override={"periods_per_year": 1, "excess": 0.0},
            expected=(math.nan,),
            reason="a zero excess cushion on a flat curve leaves a zero denominator and zero excess growth, so the "
            "ratio is a 0/0, i.e. NaN — the degenerate-denominator NaN beside the +inf pin",
        ),
    ),
)
