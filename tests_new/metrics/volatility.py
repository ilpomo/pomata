"""Spec for ``pomata.metrics.volatility`` — reducing, the annualized sample standard deviation, degree-1 homogeneous."""

import polars as pl
from tests.metrics.oracles import volatility_reference
from tests.support import well_spread
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import volatility


def _well_spread(frame: pl.DataFrame) -> bool:
    """Reject a near-constant series: the one-pass std and the two-pass oracle round its variance apart at 0."""
    return well_spread(frame.to_series(0).to_list())


VOLATILITY = Spec(
    factory=volatility,
    inputs=("returns",),
    params={"periods_per_year": 252},
    shape=Shape.REDUCING,
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"periods_per_year": -1}, r"periods_per_year must be >= 1"),
        ({"periods_per_year": -252}, r"periods_per_year must be >= 1"),
    ),
    oracle=volatility_reference,
    conditioning=_well_spread,
    # volatility(k*r) == |k|*volatility(r): degree-1 homogeneous (test_volatility.py::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("returns",), degree=1),),
    golden_input={"returns": (0.1, -0.1, 0.2, -0.2)},
    golden_output=(2.8983,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(None,),
            reason="one observation has no sample dispersion, so the result is null "
            "(test_volatility.py::test_single_row)",
        ),
        SpecPin(
            label="flat_returns_zero",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            expected=(0.0,),
            reason="a constant series has zero dispersion, so the volatility is exactly 0 "
            "(test_volatility.py::test_flat_returns_zero)",
        ),
        SpecPin(
            label="golden_periods_per_year_1",
            inputs={"returns": (0.1, -0.1, 0.2, -0.2)},
            expected=(0.18257418583505539,),
            reason="the un-annualized golden branch: the sample std of the four values "
            "(test_volatility.py::test_golden_master)",
            params_override={"periods_per_year": 1},
        ),
    ),
)
