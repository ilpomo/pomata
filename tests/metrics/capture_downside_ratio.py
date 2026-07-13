"""Spec for ``pomata.metrics.capture_downside_ratio`` — reducing, down-market capture, scale-exempt."""

import math

from tests.metrics.oracles import capture_downside_ratio_reference
from tests.support.spec import ScaleExempt, Shape, Spec, SpecPin

from pomata.metrics import capture_downside_ratio

CAPTURE_DOWNSIDE_RATIO = Spec(
    factory=capture_downside_ratio,
    inputs=("returns", "benchmark"),
    params={"periods_per_year": 252},
    shape=Shape.REDUCING,
    raises=(({"periods_per_year": 0}, r"periods_per_year must be >= 1"),),
    oracle=capture_downside_ratio_reference,
    # A ratio of two annualized geometric returns — neither scale-invariant nor homogeneous
    # (tests/metrics/test_capture_downside_ratio.py module docstring).
    scale=ScaleExempt(reason="a ratio of two annualized geometric returns — neither scale-invariant nor homogeneous"),
    golden_input={
        "returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018),
        "benchmark": (0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015),
    },
    golden_output=(1.0224,),
    pins=(
        SpecPin(
            label="no_down_market_is_null",
            inputs={"returns": (0.01, 0.02, 0.03), "benchmark": (0.01, 0.02, 0.03)},
            expected=(None,),
            reason="with no negative-benchmark period the ratio is undefined, so the result is null "
            "(tests/metrics/test_capture_downside_ratio.py::test_no_down_market_is_null)",
        ),
        SpecPin(
            label="return_leg_wiped_out_is_nan",
            inputs={"returns": (0.02, -1.5, 0.01), "benchmark": (-0.01, -0.02, -0.03)},
            expected=(math.nan,),
            reason="a selected portfolio return <= -1 wipes that leg out of the geometric-growth domain, a loud NaN "
            "(tests/metrics/test_capture_downside_ratio.py::test_return_below_negative_one_is_nan)",
        ),
        SpecPin(
            label="benchmark_leg_wiped_out_is_nan",
            inputs={"returns": (0.02, -0.03, 0.01), "benchmark": (-0.01, -1.2, -0.03)},
            expected=(math.nan,),
            reason="a selected benchmark value <= -1 wipes that leg out of the geometric-growth domain, a loud NaN "
            "(tests/metrics/test_capture_downside_ratio.py::test_return_below_negative_one_is_nan)",
        ),
    ),
)
