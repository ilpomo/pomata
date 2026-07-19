"""Declaration for ``pomata.metrics.cagr`` — reducing, the annualized compound growth rate, scale-exempt."""

import math

from pomata.metrics import cagr
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_cagr
from tests.support.declaration import Example, Golden, Pin, ScaleExempt

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
    wikipedia="https://en.wikipedia.org/wiki/Compound_annual_growth_rate",
    see_also=(
        ("total_return", "The un-annualized total growth this is the per-year rate of."),
        ("cagr_rolling", "The windowed twin, computed over each trailing window."),
        ("calmar_ratio", "The CAGR-over-drawdown ratio built on this."),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` equity is skipped; an all-null (or empty) series yields ``null`` — the rate "
            "uses the last defined equity and the count of defined observations.",
        ),
        ("NaN", "a ``NaN`` equity propagates, yielding ``NaN``."),
        (
            "Domain",
            "a curve whose last defined value is ``<= 0`` has no fractional-power growth (the raw "
            "power would be parity-dependent garbage), so the result is a loud ``NaN`` — never a "
            "plausible wrong number.",
        ),
        (
            "Insufficient sample",
            "annualizing a handful of periods extrapolates aggressively (e.g. one period at "
            "``periods_per_year = 252`` raises the growth to the 252nd power, which can overflow to "
            "``+inf`` — reported, not clipped); this is the defined geometric behavior, not an error.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the compound annual growth rate (one value in ``select``, "
    "one per group under ``.over``). ``null`` when there are no observations.",
    raises_prose="ValueError: If ``periods_per_year < 1``.",
    args_prose={
        "equity_curve": "Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive; "
        "its ``N`` values are ``N`` period growth factors, and its final value is the total "
        "growth multiple.",
    },
    example_columns={"equity_curve": "equity"},
    examples=(
        Example(inputs={"equity_curve": (1.1, 1.21)}, params={"periods_per_year": 1}, round_to=4),
        Example(
            inputs={"equity_curve": (1.1, 1.21, 1.05, 1.1025)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("A", "A", "B", "B"),
            params={"periods_per_year": 1},
            round_to=4,
        ),
        Example(
            inputs={"equity_curve": (1.0, 1.1, None, 1.21, float("nan"), 1.3)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            params={"periods_per_year": 1},
            round_to=4,
        ),
        Example(
            inputs={"equity_curve": (1.0, 0.5, -0.2)},
            intro="**Domain** — a negative terminal equity falls outside the fractional-power domain, so "
            "the result is a loud ``NaN``:",
            params={"periods_per_year": 252},
        ),
        Example(
            inputs={"equity_curve": (1.01,)},
            intro="**Insufficient sample** — a single observation still annualizes its growth over the one "
            "period it spans, rather than requiring a minimum count:",
            params={"periods_per_year": 4},
            round_to=4,
        ),
    ),
)
