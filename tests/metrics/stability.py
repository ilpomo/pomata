"""Spec for ``pomata.metrics.stability`` — reducing, the R-squared of the cumulative-log-return trend, scale-exempt."""

import math

from tests.metrics.oracles import stability_reference
from tests.support.spec import ScaleExempt, Shape, Spec, SpecPin

from pomata.metrics import stability

STABILITY = Spec(
    factory=stability,
    inputs=("returns",),
    params={},
    shape=Shape.REDUCING,
    oracle=stability_reference,
    # A coefficient of determination of the cumulative-log-return trend: the nonlinear log breaks homogeneity, so it is
    # neither scale-invariant nor homogeneous — its defining property is the [0, 1] bound.
    scale=ScaleExempt(
        reason="an R-squared of the cumulative log-return trend — the nonlinear log makes it neither scale-invariant "
        "nor scale-homogeneous"
    ),
    golden_input={"returns": (0.01, 0.012, 0.009, 0.011, 0.013, 0.008, 0.01, 0.012)},
    golden_output=(0.9984,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (0.01,)},
            expected=(None,),
            reason="one observation has no dispersion; the regression needs two points",
        ),
        SpecPin(
            label="constant_is_one",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            expected=(1.0,),
            reason="a constant non-zero return series has a perfectly linear cumulative log, so R-squared is 1.0",
        ),
        SpecPin(
            label="flat_path_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series has a flat (zero-variance) cumulative log, so R-squared is NaN — the "
            "exact-flat core of the near-constant regime; impl and oracle agree on every fuzz-reachable path, so no "
            "conditioning filter is declared, and only this pin reaches the exact flat",
        ),
        SpecPin(
            label="out_of_domain_is_nan",
            inputs={"returns": (0.02, -1.5, 0.01)},
            expected=(math.nan,),
            reason="a return at or below -1 makes log1p undefined, propagating to NaN",
        ),
    ),
)
