"""
Declaration for ``pomata.metrics.volatility`` — reducing, the annualized sample standard deviation, degree-1
homogeneous.
"""

from pomata.metrics import volatility
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_volatility
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

VOLATILITY = suite_metrics(
    factory=volatility,
    inputs=("returns",),
    params={"periods_per_year": 252},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.SQRT_TIME,
    degenerate=Degenerate.EXACT_ZERO,
    oracle=reference_volatility,
    scaling=(ScaleAxis(roles=("returns",), degree=1),),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"periods_per_year": -1}, r"periods_per_year must be >= 1"),
        ({"periods_per_year": -252}, r"periods_per_year must be >= 1"),
    ),
    golden=Golden(inputs={"returns": (0.1, -0.1, 0.2, -0.2)}, output=(2.8983,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(None,),
            reason="one observation has no sample dispersion, so the result is null",
        ),
        Pin(
            label="flat_returns_zero",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            expected=(0.0,),
            reason="a constant series has zero dispersion, so the volatility is exactly 0 — the exact-zero "
            "core of the near-constant regime; the property tiers' absolute band absorbs the "
            "rounding-noise dispersion there, so no conditioning filter is declared",
        ),
        Pin(
            label="golden_periods_per_year_1",
            inputs={"returns": (0.1, -0.1, 0.2, -0.2)},
            expected=(0.18257418583505539,),
            reason="the un-annualized golden branch: the sample std of the four values",
            params_override={"periods_per_year": 1},
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Volatility_%28finance%29",
    see_also=(
        ("volatility_rolling", "The rolling (windowed) form."),
        ("downside_deviation", "The downside-only (one-sided) counterpart."),
        ("returns_net", "The usual source of the net-return series this measures."),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` return is skipped (excluded from the standard deviation); an all-null (or "
            "empty) series yields ``null``.",
        ),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "fewer than two returns leave the sample standard deviation undefined, so the result is ``null``.",
        ),
        ("Degenerate denominator", "a constant series has zero dispersion, so the result is exactly ``0``."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the annualized volatility of the series (one value in "
    "``select``, one per group under ``.over``). ``null`` when fewer than two returns are "
    "present (the sample standard deviation is undefined).",
    raises_prose="ValueError: If ``periods_per_year < 1``.",
    examples=(
        Example(inputs={"returns": (0.01, -0.02, 0.015, 0.005, -0.01)}, params={"periods_per_year": 252}, round_to=4),
        Example(
            inputs={"returns": (0.01, -0.02, 0.015, 0.005, -0.01, 0.02, 0.01, -0.03, 0.0, 0.01)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker's volatility is "
            "computed independently:",
            partition=("AAPL",) * 5 + ("NVDA",) * 5,
            params={"periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.01, None, -0.02, 0.015, float("nan"), 0.005, -0.01)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            params={"periods_per_year": 252},
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.05,)},
            intro="**Insufficient sample** — one observation has no sample dispersion, so the result is ``null``:",
            params={"periods_per_year": 252},
        ),
        Example(
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            intro="**Degenerate denominator** — a constant series has zero dispersion, so the volatility is "
            "exactly ``0``:",
            params={"periods_per_year": 252},
        ),
    ),
)
