"""Declaration for ``pomata.metrics.skewness`` — reducing, the standardized third moment, scale-invariant."""

import math

import polars as pl

from pomata.metrics import skewness
from tests_new.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests_new.metrics.harness import suite_metrics
from tests_new.metrics.oracles import reference_skewness
from tests_new.support.declaration import Golden, Pin, ScaleAxis
from tests_new.support.strategies import well_spread


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
)
