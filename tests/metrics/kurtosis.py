"""
Declaration for ``pomata.metrics.kurtosis`` — reducing, the standardized fourth moment minus three, scale-invariant.
"""

import math

import polars as pl

from pomata.metrics import kurtosis
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_kurtosis
from tests.support.declaration import Example, Golden, Pin, ScaleAxis
from tests.support.strategies import well_spread


def _well_spread(frame: pl.DataFrame) -> bool:
    """
    Reject a near-constant sample: the standardized moment is a 0/0 the one- and two-pass paths resolve apart.
    JUSTIFIED by measurement, and the member the shared cut is sized on: the impl-vs-oracle divergence onset sits at
    stdev_rel ~3.5e-6 (single-outlier structure) against the cut's 3.16e-5 — a factor of ~6, so the transition
    straddles the cut and the filter must not be narrowed (see the near_constant_diverges pin for the divergence
    itself, chaotic up to ~1.5e-4 relative deep inside the excluded band).
    """
    return well_spread(frame.to_series(0).to_list())


KURTOSIS = suite_metrics(
    factory=kurtosis,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.ZERO_DISPERSION_IS_NAN,
    oracle=reference_kurtosis,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    conditioning=_well_spread,
    golden=Golden(inputs={"returns": (0.01, -0.02, 0.015, -0.03, 0.005, -0.01, 0.02)}, output=(-1.3223,)),
    pins=(
        Pin(
            label="constant_is_nan",
            inputs={"returns": (0.01, 0.01, 0.01)},
            expected=(math.nan,),
            reason="a constant series has zero variance, so the standardized fourth moment is 0/0, i.e. NaN "
            "— the exact core of the near-constant regime the conditioning filter excludes from the "
            "property tiers",
            covers_conditioning=True,
        ),
        Pin(
            label="near_constant_diverges_from_reference",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0100000000001)},
            expected=(3.143213990994586,),
            reason="deep inside the excluded band (stdev_rel ~3.3e-12) the one-pass moment and the two-pass "
            "oracle genuinely diverge (impl 3.143213990994586 vs oracle 3.14273818915414, ~1.5e-4 "
            "relative — both pure rounding artifacts of a quantity that is not there), the measured "
            "fact that keeps this filter JUSTIFIED at the shared cut; pinned to the implementation's "
            "deterministic output",
        ),
        Pin(
            label="subnormal_magnitude_is_nan",
            inputs={"returns": (0.0, 1e-160, 2e-160)},
            expected=(math.nan,),
            reason="a subnormal-magnitude series has m2**2 underflow to zero, yielding NaN ",
        ),
    ),
    reference='Joanes, D. N. & Gill, C. A. (1998). "Comparing Measures of Sample Skewness and '
    'Kurtosis." *Journal of the Royal Statistical Society: Series D (The Statistician)*, '
    "47(1), 183-189.",
    doi="https://doi.org/10.1111/1467-9884.00122",
    wikipedia="https://en.wikipedia.org/wiki/Kurtosis",
    see_also=(
        ("skewness", "The third-moment companion (asymmetry)."),
        ("kurtosis_rolling", "The rolling (windowed) form."),
        ("value_at_risk_modified", "Uses this excess kurtosis in its Cornish-Fisher tail correction."),
    ),
    bullets=(
        ("Null", "a ``null`` return is skipped; an all-null (or empty) series yields ``null``."),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Degenerate denominator",
            "a constant series (or a single value) has zero variance, so the standardized moment is a "
            "``0 / 0``, i.e. ``NaN``.",
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
    returns_body="A single ``Float64`` value: the excess kurtosis (one value in ``select``, one per group "
    "under ``.over``). ``null`` when there are no returns, and ``NaN`` when the returns have "
    "zero variance (fewer than two distinct values).",
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
            partition=("A", "A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B", "B"),
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.01, None, -0.02, 0.015, float("nan"), -0.03, 0.005)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.01, 0.01, 0.01)},
            intro="**Degenerate denominator** — a constant series has zero variance, so the standardized "
            "fourth moment is ``0/0``, i.e. ``NaN``:",
        ),
    ),
)
