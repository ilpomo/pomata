"""
Declaration for ``pomata.metrics.alpha`` — reducing, annualized Jensen's alpha over a benchmark baseline, scale-
exempt.
"""

import math

import polars as pl

from pomata.metrics import alpha, beta
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_alpha
from tests.support.declaration import Example, Golden, Pin, ScaleExempt


def _alpha_component() -> pl.Expr:
    """Jensen's alpha recomposed from the public ``beta`` factory at the spec's default params (rf 0.0, 252 periods)."""
    rf_period = math.pow(1.0 + 0.0, 1.0 / 252) - 1.0
    excess = (pl.col("returns") - rf_period) - beta(pl.col("returns"), pl.col("benchmark")) * (
        pl.col("benchmark") - rf_period
    )
    return (1.0 + excess.mean()) ** 252 - 1.0


ALPHA = suite_metrics(
    factory=alpha,
    inputs=("returns", "benchmark"),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.GEOMETRIC,
    degenerate=Degenerate.ZERO_DISPERSION_IS_NAN,
    oracle=reference_alpha,
    recomposition=_alpha_component,
    scaling=ScaleExempt(
        reason="annualizes a return beyond a benchmark-explained baseline — neither scale-invariant nor homogeneous"
    ),
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
        output=(-0.0903,),
        params={"risk_free_rate": 0.02},
    ),
    pins=(
        Pin(
            label="null_misalignment_drops_pair",
            inputs={"returns": (0.01, None, 0.03, -0.01, 0.02), "benchmark": (0.008, -0.01, None, -0.005, 0.018)},
            expected=(-0.47369237088902216,),
            reason="an observation with a null in either leg is dropped, matching the reference over the "
            "retained pairs",
        ),
        Pin(
            label="nan_poisons",
            inputs={"returns": (0.01, math.nan, 0.03, -0.01), "benchmark": (0.008, -0.01, 0.025, -0.005)},
            expected=(math.nan,),
            reason="a NaN in either leg of a retained pair poisons the result to NaN",
        ),
        Pin(
            label="single_pair",
            inputs={"returns": (0.05,), "benchmark": (0.04,)},
            expected=(None,),
            reason="a single complete pair yields null — the regression slope needs two observations",
        ),
        Pin(
            label="constant_benchmark_0_1",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (0.1, 0.1, 0.1)},
            expected=(math.nan,),
            reason="a constant benchmark makes the embedded beta NaN, which propagates to alpha — the "
            "exact-zero core of the near-constant regime; no conditioning filter is declared: the "
            "embedded cov/var slope matches the oracle within one ULP even at ULP-adjacent benchmark "
            "spreads (measured down to a 1e-15 spread on base 0.1)",
        ),
        Pin(
            label="constant_benchmark_one_third",
            inputs={
                "returns": (0.01, -0.02, 0.03),
                "benchmark": (0.3333333333333333, 0.3333333333333333, 0.3333333333333333),
            },
            expected=(math.nan,),
            reason="the same guard at a constant not exactly representable in float",
        ),
        Pin(
            label="constant_benchmark_many_digits",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (0.123456789, 0.123456789, 0.123456789)},
            expected=(math.nan,),
            reason="the same guard at a third, many-digit constant magnitude",
        ),
    ),
    reference='Jensen, M. C. (1968). "The Performance of Mutual Funds in the Period 1945-1964." *The '
    "Journal of Finance*, 23(2), 389-416.",
    doi="https://doi.org/10.1111/j.1540-6261.1968.tb00815.x",
    wikipedia="https://en.wikipedia.org/wiki/Jensen%27s_alpha",
    see_also=(
        ("beta", "The regression slope this corrects the return for."),
        ("treynor_ratio", "The excess return per unit of the same systematic risk."),
        ("alpha_rolling", "The same measure over a trailing window."),
    ),
    bullets=(
        ("Null", "an observation is used only where both legs are present; a ``null`` in either drops that pair."),
        ("NaN", "a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "fewer than two complete pairs leaves the regression slope undefined, so the result is ``null``.",
        ),
        (
            "Degenerate denominator",
            "a zero-variance benchmark makes :func:`beta` ``NaN`` (a ``0 / 0``), which propagates here.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the annualized Jensen's alpha (one value in ``select``, one "
    "per group under ``.over``). ``null`` when fewer than two complete pairs are present.",
    raises_prose="ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite or is ``< -1``.",
    args_prose={
        "risk_free_rate": "The annualized risk-free rate, converted to a per-period rate geometrically (default "
        "``0.0``). Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + "
        "risk_free_rate >= 0``).",
    },
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
            intro="**Insufficient sample** — a single complete pair yields ``null``, since the regression "
            "slope needs two observations:",
            params={"periods_per_year": 252},
        ),
        Example(
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (0.1, 0.1, 0.1)},
            intro="**Degenerate denominator** — a constant benchmark makes the embedded beta ``NaN``, which "
            "propagates to the result:",
            params={"periods_per_year": 252},
        ),
    ),
)
