"""
Declaration for ``pomata.metrics.information_ratio`` — reducing, active return over tracking error, scale-invariant.
"""

import math

from pomata.metrics import information_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_information_ratio
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

INFORMATION_RATIO = suite_metrics(
    factory=information_ratio,
    inputs=("returns", "benchmark"),
    params={"periods_per_year": 252},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.SQRT_TIME,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_information_ratio,
    scaling=(ScaleAxis(roles=("returns", "benchmark"), degree=0),),
    raises=(({"periods_per_year": 0}, r"periods_per_year must be >= 1"),),
    golden=Golden(
        inputs={
            "returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018),
            "benchmark": (0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015),
        },
        output=(-0.842,),
    ),
    pins=(
        Pin(
            label="single_pair",
            inputs={"returns": (0.05,), "benchmark": (0.04,)},
            expected=(None,),
            reason="a single complete pair yields null — the tracking error needs two observations",
        ),
        Pin(
            label="zero_tracking_error_is_inf",
            inputs={"returns": (0.01, 0.01, 0.01), "benchmark": (0.0, 0.0, 0.0)},
            expected=(math.inf,),
            reason="a constant active series has zero tracking error with a positive mean, so the ratio is +inf",
        ),
        Pin(
            label="zero_active_is_nan",
            inputs={"returns": (0.01, 0.02, 0.03), "benchmark": (0.01, 0.02, 0.03)},
            expected=(math.nan,),
            reason="identical legs give an exactly-zero active series: zero mean over zero tracking error is "
            "the 0/0 NaN, resolved by the exact dispersion guard on both sides — the oracle detects a "
            "constant active series via min == max, mirroring the kernel's exact zero-dispersion pin; "
            "no conditioning filter is declared",
        ),
    ),
    reference='Goodwin, T. H. (1998). "The Information Ratio." *Financial Analysts Journal*, 54(4), 34-43.',
    doi="https://doi.org/10.2469/faj.v54.n4.2196",
    wikipedia="https://en.wikipedia.org/wiki/Information_ratio",
    see_also=(
        ("sharpe_ratio", "The total-risk analog measured against a risk-free rate, not a benchmark."),
        ("information_ratio_rolling", "The same measure over a trailing window."),
        ("alpha", "The benchmark-active return measured per unit of beta instead of tracking error."),
    ),
    bullets=(
        ("Null", "an observation is used only where both legs are present; a ``null`` in either drops that pair."),
        ("NaN", "a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "fewer than two complete pairs leaves the sample tracking error undefined, so the result is ``null``.",
        ),
        (
            "Degenerate denominator",
            "a constant active series has zero tracking error, so the result is ``+/-inf`` (or "
            "``NaN`` when the mean active is also zero, the ``0 / 0``) — reported, not clipped.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the annualized information ratio (one value in ``select``, "
    "one per group under ``.over``). ``null`` when fewer than two complete pairs are present "
    "(the tracking error is undefined).",
    raises_prose="ValueError: If ``periods_per_year < 1``.",
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
            intro="**Insufficient sample** — a single complete pair yields ``null``, since the tracking "
            "error needs two observations:",
            params={"periods_per_year": 252},
        ),
        Example(
            inputs={"returns": (0.01, 0.01, 0.01), "benchmark": (0.0, 0.0, 0.0)},
            intro="**Degenerate denominator** — a constant active series has zero tracking error with a "
            "positive mean, so the ratio is ``+inf``:",
            params={"periods_per_year": 252},
        ),
        Example(
            inputs={"returns": (0.01, 0.02, 0.03), "benchmark": (0.01, 0.02, 0.03)},
            intro="**Degenerate denominator** — identical legs give an exactly-zero active series, the zero "
            "mean over zero tracking error ``0 / 0`` case, so the result is ``NaN``:",
            params={"periods_per_year": 252},
        ),
    ),
)
