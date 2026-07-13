"""Spec for ``pomata.metrics.kurtosis`` — reducing, the standardized fourth moment minus three, scale-invariant."""

import math

import polars as pl
from tests_new.metrics.oracles import kurtosis_reference
from tests_new.support import well_spread
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import kurtosis


def _well_spread(frame: pl.DataFrame) -> bool:
    """
    Reject a near-constant sample: the standardized moment is a 0/0 the one- and two-pass paths resolve apart.
    JUSTIFIED by measurement, and the member the shared cut is sized on: the impl-vs-oracle divergence onset sits at
    stdev_rel ~3.5e-6 (single-outlier structure) against the cut's 3.16e-5 — a factor of ~6, so the transition
    straddles the cut and the filter must not be narrowed (see the near_constant_diverges pin for the divergence
    itself, chaotic up to ~1.5e-4 relative deep inside the excluded band).
    """
    return well_spread(frame.to_series(0).to_list())


KURTOSIS = Spec(
    factory=kurtosis,
    inputs=("returns",),
    params={},
    shape=Shape.REDUCING,
    oracle=kurtosis_reference,
    conditioning=_well_spread,
    # A standardized moment is scale-invariant, degree 0 (test_kurtosis.py::test_scale_invariance).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.01, -0.02, 0.015, -0.03, 0.005, -0.01, 0.02)},
    golden_output=(-1.3223,),
    pins=(
        SpecPin(
            label="constant_is_nan",
            inputs={"returns": (0.01, 0.01, 0.01)},
            expected=(math.nan,),
            reason="a constant series has zero variance, so the standardized fourth moment is 0/0, i.e. NaN — the "
            "exact core of the near-constant regime the conditioning filter excludes from the property tiers "
            "(test_kurtosis.py::test_constant_is_nan)",
            covers_conditioning=True,
        ),
        SpecPin(
            label="near_constant_diverges_from_reference",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01 * (1.0 + 1e-11))},
            expected=(3.143213990994586,),
            reason="deep inside the excluded band (stdev_rel ~3.3e-12) the one-pass moment and the two-pass oracle "
            "genuinely diverge (impl 3.143213990994586 vs oracle 3.14273818915414, ~1.5e-4 relative — both pure "
            "rounding artifacts of a quantity that is not there), the measured fact that keeps this filter "
            "JUSTIFIED at the shared cut; pinned to the implementation's deterministic output",
        ),
        SpecPin(
            label="subnormal_magnitude_is_nan",
            inputs={"returns": (0.0, 1e-160, 2e-160)},
            expected=(math.nan,),
            reason="a subnormal-magnitude series has m2**2 underflow to zero, yielding NaN "
            "(test_kurtosis.py::test_subnormal_magnitude_is_nan)",
        ),
    ),
)
