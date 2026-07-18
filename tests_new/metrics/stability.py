"""
Declaration for ``pomata.metrics.stability`` — reducing, the R-squared of the cumulative-log-return trend, scale-
exempt.
"""

import math

from pomata.metrics import stability
from tests_new.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests_new.metrics.harness import suite_metrics
from tests_new.metrics.oracles import reference_stability
from tests_new.support.declaration import Golden, Pin, ScaleExempt

STABILITY = suite_metrics(
    factory=stability,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.ZERO_DISPERSION_IS_NAN,
    oracle=reference_stability,
    scaling=ScaleExempt(
        reason="an R-squared of the cumulative log-return trend — the nonlinear log makes it neither "
        "scale-invariant nor scale-homogeneous"
    ),
    golden=Golden(inputs={"returns": (0.01, 0.012, 0.009, 0.011, 0.013, 0.008, 0.01, 0.012)}, output=(0.9984,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.01,)},
            expected=(None,),
            reason="one observation has no dispersion; the regression needs two points",
        ),
        Pin(
            label="lone_nan_poisons",
            inputs={"returns": (math.nan,)},
            expected=(math.nan,),
            reason="a NaN poisons the result even when it is the only observation — the poison guard wins "
            "over the two-point count guard, exactly as in the cagr / total_return siblings",
        ),
        Pin(
            label="constant_is_one",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            expected=(1.0,),
            reason="a constant non-zero return series has a perfectly linear cumulative log, so R-squared is 1.0",
        ),
        Pin(
            label="flat_path_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series has a flat (zero-variance) cumulative log, so R-squared is NaN — the "
            "exact-flat core of the near-constant regime; impl and oracle agree on every "
            "fuzz-reachable path, so no conditioning filter is declared, and only this pin reaches "
            "the exact flat",
        ),
        Pin(
            label="out_of_domain_is_nan",
            inputs={"returns": (0.02, -1.5, 0.01)},
            expected=(math.nan,),
            reason="a return at or below -1 makes log1p undefined, propagating to NaN",
        ),
    ),
)
