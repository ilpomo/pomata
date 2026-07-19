"""Declaration for ``pomata.metrics.skewness`` — reducing, the standardized third moment, scale-invariant."""

import math

import polars as pl

from pomata.metrics import skewness
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_skewness
from tests.support.declaration import Example, Golden, Pin, ScaleAxis
from tests.support.strategies import well_spread


def _well_spread(frame: pl.DataFrame) -> bool:
    """
    Reject a near-constant sample: the standardized moment is a 0/0 the one- and two-pass paths resolve apart.
    KEPT deliberately over-wide: skewness' own divergence onset sits at var_rel ~2.8e-13, far below the shared cut
    of 1e-9, but the cut is sized on the worst family member (kurtosis, whose onset straddles it) and a spec-local
    narrowing would buy back <2% of draws at the price of one more magic constant — over-width here is a safe,
    conservative guard, not a hazard.
    """
    return well_spread(frame.to_series(0).to_list())


SKEWNESS = suite_metrics(
    factory=skewness,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.ZERO_DISPERSION_IS_NAN,
    oracle=reference_skewness,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    conditioning=_well_spread,
    golden=Golden(inputs={"returns": (0.01, -0.02, 0.015, -0.03, 0.005, -0.01, 0.02)}, output=(-0.384,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(math.nan,),
            reason="one observation has zero variance, so the standardized third moment is 0/0, i.e. NaN ",
        ),
        Pin(
            label="constant_is_nan",
            inputs={"returns": (0.01, 0.01, 0.01)},
            expected=(math.nan,),
            reason="a constant series has zero variance, so the standardized moment is 0/0, i.e. NaN — the "
            "exact core of the near-constant regime the conditioning filter excludes from the "
            "property tiers",
            covers_conditioning=True,
        ),
        Pin(
            label="subnormal_magnitude_is_nan",
            inputs={"returns": (0.0, 1e-160, 2e-160)},
            expected=(math.nan,),
            reason="a subnormal-magnitude series has m2**1.5 underflow to zero, yielding NaN ",
        ),
    ),
    reference='Joanes, D. N. & Gill, C. A. (1998). "Comparing Measures of Sample Skewness and '
    'Kurtosis." *Journal of the Royal Statistical Society: Series D (The Statistician)*, '
    "47(1), 183-189.",
    doi="https://doi.org/10.1111/1467-9884.00122",
    wikipedia="https://en.wikipedia.org/wiki/Skewness",
    see_also=(
        ("kurtosis", "The fourth-moment companion (tailedness)."),
        ("skewness_rolling", "The rolling (windowed) form."),
        ("value_at_risk_modified", "Uses this skewness in its Cornish-Fisher tail correction."),
    ),
    bullets=(
        ("Null", "a ``null`` return is skipped; an all-null (or empty) series yields ``null``."),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "a single observation has zero variance, so the standardized moment is a ``0 / 0``, i.e. ``NaN``.",
        ),
        (
            "Degenerate denominator",
            "a constant series has zero variance, so the standardized moment is a ``0 / 0``, i.e. ``NaN``.",
        ),
        (
            "Stability",
            "on a near-constant series (relative spread far below the property tests' conditioning "
            "floor) the standardized moment is a rounding-dominated ``0 / 0`` where the one-pass "
            "kernel and a two-pass computation resolve apart; that band is excluded from the property "
            "tiers, and the value is reported as computed.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the skewness (one value in ``select``, one per group under "
    "``.over``). ``null`` when there are no returns, and ``NaN`` when the returns have zero "
    "variance (fewer than two distinct values).",
    examples=(
        Example(inputs={"returns": (0.01, -0.02, 0.015, -0.03, 0.005, -0.01, 0.02)}, round_to=4),
        Example(
            inputs={
                "returns": (
                    0.01,
                    -0.02,
                    0.015,
                    -0.03,
                    0.005,
                    -0.01,
                    0.02,
                    0.02,
                    -0.01,
                    0.03,
                    -0.02,
                    0.01,
                    -0.005,
                    0.025,
                )
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("AAPL",) * 7 + ("NVDA",) * 7,
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.01, None, -0.02, 0.015, float("nan"), -0.03, 0.005)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.05,)},
            intro="**Insufficient sample** — one observation has zero variance, so the standardized third "
            "moment is ``0/0``, i.e. ``NaN``:",
        ),
        Example(
            inputs={"returns": (0.01, 0.01, 0.01)},
            intro="**Degenerate denominator** — a constant series has zero variance, so the standardized "
            "moment is ``0/0``, i.e. ``NaN``:",
        ),
    ),
)
