"""Declaration for ``pomata.metrics.modigliani_risk_adjusted_performance`` — reducing, M-squared, degree-1 at rf=0."""

import math

import polars as pl

from pomata.metrics import modigliani_risk_adjusted_performance, sharpe_ratio, volatility
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_modigliani_risk_adjusted_performance
from tests.support.declaration import Example, Golden, Pin, ScaleAxis


def _modigliani_component() -> pl.Expr:
    """M-squared recomposed from the public ``sharpe_ratio`` and ``volatility`` factories at the default params."""
    return 0.0 + sharpe_ratio(pl.col("returns"), periods_per_year=252, risk_free_rate=0.0) * volatility(
        pl.col("benchmark"), periods_per_year=252
    )


MODIGLIANI_RISK_ADJUSTED_PERFORMANCE = suite_metrics(
    factory=modigliani_risk_adjusted_performance,
    inputs=("returns", "benchmark"),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.LINEAR,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_modigliani_risk_adjusted_performance,
    recomposition=_modigliani_component,
    scaling=(ScaleAxis(roles=("returns", "benchmark"), degree=1),),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -2.0}, r"risk_free_rate must be >= -1"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    golden=Golden(
        inputs={
            "returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018),
            "benchmark": (0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015),
        },
        output=(0.3274,),
        params={"risk_free_rate": 0.02},
    ),
    pins=(
        Pin(
            label="single_pair",
            inputs={"returns": (0.05,), "benchmark": (0.04,)},
            expected=(None,),
            reason="a single complete pair leaves the embedded Sharpe and benchmark volatility undefined, so null",
        ),
        Pin(
            label="constant_portfolio_is_inf",
            inputs={"returns": (0.01, 0.01, 0.01), "benchmark": (0.02, -0.01, 0.03)},
            expected=(math.inf,),
            reason="a constant portfolio has zero dispersion by the exact pin, so the embedded Sharpe is "
            "+inf, which propagates to +inf; no conditioning filter is declared: the composed oracle "
            "mirrors the same exact min == max constancy detection on both legs, so the two sides "
            "agree in kind on every constant series",
        ),
    ),
    reference='Modigliani, F. & Modigliani, L. (1997). "Risk-Adjusted Performance." *The Journal of '
    "Portfolio Management*, 23(2), 45-54.",
    doi="https://doi.org/10.3905/jpm.23.2.45",
    wikipedia="https://en.wikipedia.org/wiki/Modigliani_risk-adjusted_performance",
    see_also=(
        ("sharpe_ratio", "The risk-adjusted ratio this expresses in return units."),
        ("volatility", "The benchmark dispersion it scales to."),
        ("information_ratio", "Another benchmark-relative performance measure, as a ratio."),
    ),
    bullets=(
        ("Null", "an observation is used only where both legs are present; a ``null`` in either drops that pair."),
        ("NaN", "a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "fewer than two complete pairs leaves the embedded Sharpe ratio and benchmark volatility "
            "undefined, so the result is ``null``.",
        ),
        (
            "Degenerate denominator",
            "a constant portfolio has zero volatility, so its :func:`sharpe_ratio` is infinite and "
            "the result is ``+/-inf`` — reported, not clipped.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the M-squared measure as an annualized return (one value in "
    "``select``, one per group under ``.over``). ``null`` when fewer than two complete pairs "
    "are present.",
    raises_prose="ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite or is ``< -1``.",
    args_prose={
        "risk_free_rate": "The annualized risk-free rate, used both to form the Sharpe excess (geometrically per "
        "period) and as the additive level here (default ``0.0``). Must be finite and ``>= -1``.",
    },
    example_alias="m_squared",
    examples=(
        Example(
            inputs={
                "returns": (0.02, -0.01, 0.03, -0.02, 0.015, 0.005),
                "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004),
            },
            params={"periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={
                "returns": (0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.01, 0.025, -0.015, 0.008, -0.005, 0.012),
                "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004, 0.012, 0.02, -0.01, 0.006, -0.004, 0.01),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("AAPL",) * 6 + ("NVDA",) * 6,
            params={"periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={
                "returns": (None, 0.02, 0.03, float("nan"), 0.015, 0.005),
                "benchmark": (0.015, -0.008, 0.025, -0.015, 0.01, 0.004),
            },
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            params={"periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.05,), "benchmark": (0.04,)},
            intro="**Insufficient sample** — a single complete pair leaves the embedded Sharpe ratio and "
            "benchmark volatility undefined, so the result is ``null``:",
            params={"periods_per_year": 252},
        ),
        Example(
            inputs={"returns": (0.01, 0.01, 0.01), "benchmark": (0.02, -0.01, 0.03)},
            intro="**Degenerate denominator** — a constant portfolio has zero dispersion, so the embedded "
            "Sharpe ratio is ``+inf``, which propagates to ``+inf``:",
            params={"periods_per_year": 252},
        ),
    ),
)
