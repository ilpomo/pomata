"""Spec for ``pomata.metrics.recovery_ratio`` — reducing, total return per unit of maximum drawdown, scale-exempt."""

import math

import polars as pl
from tests.metrics.oracles import recovery_ratio_reference
from tests.support.spec import ScaleExempt, Shape, Spec, SpecPin

from pomata.metrics import max_drawdown, recovery_ratio, total_return

RECOVERY_RATIO = Spec(
    factory=recovery_ratio,
    inputs=("equity_curve",),
    params={},
    shape=Shape.REDUCING,
    oracle=recovery_ratio_reference,
    # A normalized growth-factor total return over a scale-invariant max-drawdown magnitude — neither invariant nor
    # homogeneous
    scale=ScaleExempt(
        reason="a normalized growth-factor total return over a scale-invariant max-drawdown magnitude — neither "
        "invariant nor homogeneous"
    ),
    golden_input={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)},
    golden_output=(8.8,),
    component_expr=lambda: total_return(pl.col("equity_curve")) / max_drawdown(pl.col("equity_curve")).abs(),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(math.nan,),
            reason="a one-element series has zero growth and zero drawdown, so the ratio is 0/0, i.e. NaN",
        ),
        SpecPin(
            label="no_drawdown_is_inf",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(math.inf,),
            reason="a monotonically rising curve has zero maximum drawdown with positive growth, so the ratio is +inf",
        ),
        SpecPin(
            label="losing_curve_is_negative",
            inputs={"equity_curve": (1.0, 0.9, 0.95, 0.7)},
            expected=(-1.0,),
            reason="a curve ending below its start reports a negative recovery factor: the total-return numerator "
            "keeps its sign over the drawdown magnitude",
        ),
        SpecPin(
            label="no_drawdown_zero_growth_is_nan",
            inputs={"equity_curve": (1.0, 1.0, 1.0)},
            expected=(math.nan,),
            reason="a flat multi-row curve has zero maximum drawdown and zero total return, so the ratio is a 0/0, "
            "i.e. NaN — the degenerate-denominator NaN beside the +inf pin",
        ),
    ),
)
