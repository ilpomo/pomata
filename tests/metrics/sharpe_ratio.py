"""
Declaration for ``pomata.metrics.sharpe_ratio`` — reducing, annualized, null-skipping, NaN-poisoning, scale-
invariant.
"""

import math

from pomata.metrics import sharpe_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_sharpe_ratio
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

SHARPE_RATIO = suite_metrics(
    factory=sharpe_ratio,
    inputs=("returns",),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.SQRT_TIME,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_sharpe_ratio,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -2.0}, r"risk_free_rate must be >= -1"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    golden=Golden(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, output=(2.4285,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(None,),
            reason="one observation has no sample dispersion, so the ratio is null",
        ),
        Pin(
            label="zero_volatility",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            expected=(math.inf,),
            reason="a constant series has zero dispersion with a positive mean, so the ratio is +inf",
        ),
        Pin(
            label="zero_excess_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series at a zero risk-free rate has zero mean AND zero dispersion, so the "
            "ratio is the 0/0 NaN — the exact-zero core of the constant regime; no conditioning "
            "filter is declared: the oracle detects an exactly-constant excess via min == max, "
            "mirroring the kernel's exact zero-dispersion pin, so the two sides agree in kind on "
            "every constant series",
        ),
    ),
    reference='Sharpe, W. F. (1994). "The Sharpe Ratio." *The Journal of Portfolio Management*, 21(1), 49-58.',
    doi="https://doi.org/10.3905/jpm.1994.409501",
    wikipedia="https://en.wikipedia.org/wiki/Sharpe_ratio",
    see_also=(
        ("sortino_ratio", "The downside-only counterpart (penalizes only harmful volatility)."),
        ("volatility", "The denominator (total dispersion)."),
        ("adjusted_sharpe_ratio", "The higher-moment correction for non-normal returns."),
        ("sharpe_ratio_rolling", "The rolling (windowed) form."),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` return is skipped (excluded from the mean and the standard deviation); an "
            "all-null (or empty) series yields ``null``.",
        ),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "with fewer than two returns the sample standard deviation is undefined, so the result is ``null``.",
        ),
        (
            "Degenerate denominator",
            "a constant excess series has zero dispersion, so the ratio is ``+/-inf`` (or ``NaN`` "
            "when the mean excess is also zero) — reported, not clipped.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the annualized Sharpe ratio (one value in ``select``, one "
    "per group under ``.over``). ``null`` when fewer than two returns are present (the sample "
    "standard deviation is undefined).",
    raises_prose="ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite or is ``< -1``.",
    args_prose={
        "risk_free_rate": "The annualized risk-free rate, converted to a per-period rate geometrically (default "
        "``0.0``). Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + "
        "risk_free_rate >= 0``).",
    },
    examples=(
        Example(
            inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)},
            params={"periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={
                "returns": (
                    0.03,
                    -0.01,
                    0.02,
                    -0.015,
                    0.01,
                    0.005,
                    -0.02,
                    0.02,
                    -0.005,
                    0.015,
                    -0.01,
                    0.025,
                    0.0,
                    -0.012,
                )
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("AAPL",) * 7 + ("NVDA",) * 7,
            params={"periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.03, None, 0.02, -0.015, float("nan"), 0.005, -0.02)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            params={"periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.05,)},
            intro="**Insufficient sample** — a single return has no sample dispersion, so the result is ``null``:",
            params={"periods_per_year": 252},
        ),
        Example(
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            intro="**Degenerate denominator** — a constant series has zero dispersion with a positive mean "
            "excess, so the ratio is ``+inf``:",
            params={"periods_per_year": 252},
        ),
        Example(
            inputs={"returns": (0.0, 0.0, 0.0, 0.0)},
            intro="**Degenerate denominator** — an all-zero series at a zero risk-free rate is the ``0/0`` "
            "case (zero mean over zero dispersion), so the ratio is ``NaN``:",
            params={"periods_per_year": 252},
        ),
    ),
)
