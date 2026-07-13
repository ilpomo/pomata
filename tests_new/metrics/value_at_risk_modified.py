"""Spec for ``pomata.metrics.value_at_risk_modified`` — reducing, the Cornish-Fisher VaR, degree-1 homogeneous."""

import math

import polars as pl
from tests_new.metrics.oracles import value_at_risk_modified_reference
from tests_new.support import well_spread
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import value_at_risk_modified


def _well_spread(frame: pl.DataFrame) -> bool:
    """Reject a near-constant sample: the skew/kurtosis the Cornish-Fisher expansion uses are a 0/0 there."""
    return well_spread(frame.to_series(0).to_list())


VALUE_AT_RISK_MODIFIED = Spec(
    factory=value_at_risk_modified,
    inputs=("returns",),
    params={"confidence": 0.95},
    shape=Shape.REDUCING,
    raises=(
        ({"confidence": 0.0}, r"confidence must be in the open interval"),
        ({"confidence": 1.0}, r"confidence must be in the open interval"),
        ({"confidence": -0.1}, r"confidence must be in the open interval"),
        ({"confidence": 1.5}, r"confidence must be in the open interval"),
    ),
    oracle=value_at_risk_modified_reference,
    conditioning=_well_spread,
    # mean + z_cf * standard deviation: degree-1 homogeneous (test_value_at_risk_modified.py::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("returns",), degree=1),),
    golden_input={"returns": (0.02, -0.04, 0.01, -0.06, 0.03, -0.05, 0.04, -0.02, 0.01, -0.03)},
    golden_output=(-0.069,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(None,),
            reason="a one-element series yields null (sample std needs >= 2) "
            "(test_value_at_risk_modified.py::test_single_row)",
        ),
        SpecPin(
            label="zero_volatility",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            expected=(math.nan,),
            reason="a constant series has undefined skewness/kurtosis, so the result is NaN "
            "(test_value_at_risk_modified.py::test_zero_volatility_is_nan)",
        ),
        SpecPin(
            label="domain_nonmonotonic_slope",
            inputs={"returns": (1.0, -1.0, -1.0, -1.0)},
            expected=(math.nan,),
            reason="the one-term Cornish-Fisher quantile map is not locally monotonic, so the estimate is NaN "
            "(test_value_at_risk_modified.py::test_out_of_domain_is_nan)",
        ),
        SpecPin(
            label="domain_median_crossing_a",
            inputs={"returns": (*([0.0] * 12), 1.0)},
            expected=(math.nan,),
            reason="tail moments extreme enough that the corrected quantile crosses the median yield NaN "
            "(test_value_at_risk_modified.py::test_out_of_domain_is_nan)",
        ),
        SpecPin(
            label="domain_median_crossing_b",
            inputs={"returns": (-1.0, 1.0, *([0.0] * 198))},
            expected=(math.nan,),
            reason="the same median-crossing domain violation on a long series "
            "(test_value_at_risk_modified.py::test_out_of_domain_is_nan)",
        ),
    ),
)
