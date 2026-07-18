"""Declaration for ``pomata.metrics.cagr`` — reducing, the annualized compound growth rate, scale-exempt."""

import math

from pomata.metrics import cagr
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_cagr
from tests.support.declaration import Golden, Pin, ScaleExempt

CAGR = suite_metrics(
    factory=cagr,
    inputs=("equity_curve",),
    params={"periods_per_year": 252},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.GEOMETRIC,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_cagr,
    scaling=ScaleExempt(
        reason="a growth factor normalized to a unit start, annualized by a fractional power — neither "
        "scale-invariant nor homogeneous"
    ),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"periods_per_year": -1}, r"periods_per_year must be >= 1"),
    ),
    golden=Golden(inputs={"equity_curve": (1.1, 1.21)}, output=(0.1,), params={"periods_per_year": 1}),
    pins=(
        Pin(
            label="single_period_annualizes",
            inputs={"equity_curve": (1.01,)},
            expected=(0.040604010000000024,),
            reason="a single observation annualizes its growth over one period (final ** periods_per_year - 1)",
            params_override={"periods_per_year": 4},
        ),
        Pin(
            label="single_period_overflow_is_inf",
            inputs={"equity_curve": (2.0,)},
            expected=(math.inf,),
            reason="a single observation raised to a huge annualization power overflows the float64 range, "
            "so the result is +inf — the defined geometric extrapolation, reported not clipped",
            params_override={"periods_per_year": 2000},
        ),
        Pin(
            label="non_positive_terminal_equity_negative",
            inputs={"equity_curve": (1.0, 0.5, -0.2)},
            expected=(math.nan,),
            reason="a negative terminal equity is out of the fractional-power domain; the factory's <= 0 "
            "guard returns a loud NaN",
        ),
        Pin(
            label="non_positive_terminal_equity_zero",
            inputs={"equity_curve": (1.0, 0.5, 0.0)},
            expected=(math.nan,),
            reason="a zero terminal equity is out of domain and the factory returns NaN by the same guard ",
        ),
    ),
)
