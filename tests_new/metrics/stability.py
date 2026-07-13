"""Spec for ``pomata.metrics.stability`` — reducing, the R-squared of the cumulative-log-return trend, scale-exempt."""

import math

import polars as pl
from tests_new.metrics.oracles import stability_reference
from tests_new.support import well_spread
from tests_new.support.spec import ScaleExempt, Shape, Spec, SpecPin

from pomata.metrics import stability


def _well_conditioned(frame: pl.DataFrame) -> bool:
    """Reject a series whose cumulative log path is degenerate: a return <= -1 makes the log undefined, and a flat
    cumulative log is a 0/0 the impl (NaN) and the oracle (0.0) resolve apart.
    """
    values = frame.to_series(0).to_list()
    finite = [value for value in values if value is not None and not math.isnan(value)]
    if any(value <= -1.0 for value in finite):
        return False
    cumulative: list[float] = []
    level = 0.0
    for value in finite:
        level += math.log1p(value)
        cumulative.append(level)
    return well_spread(cumulative)


STABILITY = Spec(
    factory=stability,
    inputs=("returns",),
    params={},
    shape=Shape.REDUCING,
    oracle=stability_reference,
    conditioning=_well_conditioned,
    # A coefficient of determination of the cumulative-log-return trend: the nonlinear log breaks homogeneity, so it is
    # neither scale-invariant nor homogeneous — its defining property is the [0, 1] bound (test_stability.py sizing
    # note).
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
            reason="one observation has no dispersion; the regression needs two points "
            "(test_stability.py::test_single_row)",
        ),
        SpecPin(
            label="constant_is_one",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            expected=(1.0,),
            reason="a constant non-zero return series has a perfectly linear cumulative log, so R-squared is 1.0 "
            "(test_stability.py::test_constant_is_one)",
        ),
        SpecPin(
            label="flat_path_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series has a flat (zero-variance) cumulative log, so R-squared is NaN "
            "(test_stability.py::test_flat_path_is_nan)",
        ),
        SpecPin(
            label="out_of_domain_is_nan",
            inputs={"returns": (0.02, -1.5, 0.01)},
            expected=(math.nan,),
            reason="a return at or below -1 makes log1p undefined, propagating to NaN "
            "(test_stability.py::test_out_of_domain_is_nan)",
        ),
    ),
)
