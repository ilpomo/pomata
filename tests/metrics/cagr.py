"""Spec for ``pomata.metrics.cagr`` — reducing, the annualized compound growth rate, scale-exempt."""

import math

from tests.metrics.oracles import cagr_reference
from tests.support.spec import ScaleExempt, Shape, Spec, SpecPin

from pomata.metrics import cagr

CAGR = Spec(
    factory=cagr,
    inputs=("equity_curve",),
    params={"periods_per_year": 252},
    shape=Shape.REDUCING,
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"periods_per_year": -1}, r"periods_per_year must be >= 1"),
    ),
    oracle=cagr_reference,
    # A growth factor normalized to a unit start, annualized by a fractional power — neither scale-invariant nor
    # homogeneous
    scale=ScaleExempt(
        reason="a growth factor normalized to a unit start, annualized by a fractional power — neither "
        "scale-invariant nor homogeneous"
    ),
    golden_params={"periods_per_year": 1},
    golden_input={"equity_curve": (1.1, 1.21)},
    golden_output=(0.1,),
    pins=(
        SpecPin(
            label="single_period_annualizes",
            inputs={"equity_curve": (1.01,)},
            expected=(1.01**4 - 1,),
            reason="a single observation annualizes its growth over one period (final ** periods_per_year - 1) ",
            params_override={"periods_per_year": 4},
        ),
        SpecPin(
            label="single_period_overflow_is_inf",
            inputs={"equity_curve": (2.0,)},
            expected=(math.inf,),
            reason="a single observation raised to a huge annualization power overflows the float64 range, so the "
            "result is +inf — the defined geometric extrapolation, reported not clipped",
            params_override={"periods_per_year": 2000},
        ),
        SpecPin(
            label="non_positive_terminal_equity_negative",
            inputs={"equity_curve": (1.0, 0.5, -0.2)},
            expected=(math.nan,),
            reason="a negative terminal equity is out of the fractional-power domain; the factory's <= 0 guard "
            "returns a loud NaN",
        ),
        SpecPin(
            label="non_positive_terminal_equity_zero",
            inputs={"equity_curve": (1.0, 0.5, 0.0)},
            expected=(math.nan,),
            reason="a zero terminal equity is out of domain and the factory returns NaN by the same guard ",
        ),
    ),
)
