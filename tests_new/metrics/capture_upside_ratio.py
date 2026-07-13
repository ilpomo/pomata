"""Spec for ``pomata.metrics.capture_upside_ratio`` — reducing, up-market capture, scale-exempt."""

import math

from tests.metrics.oracles import capture_upside_ratio_reference
from tests_new.support.spec import ScaleExempt, Shape, Spec, SpecPin

from pomata.metrics import capture_upside_ratio

CAPTURE_UPSIDE_RATIO = Spec(
    factory=capture_upside_ratio,
    inputs=("returns", "benchmark"),
    params={"periods_per_year": 252},
    shape=Shape.REDUCING,
    raises=(({"periods_per_year": 0}, r"periods_per_year must be >= 1"),),
    oracle=capture_upside_ratio_reference,
    # A ratio of two annualized geometric returns — neither scale-invariant nor homogeneous
    # (tests/metrics/test_capture_upside_ratio.py module docstring).
    scale=ScaleExempt(reason="a ratio of two annualized geometric returns — neither scale-invariant nor homogeneous"),
    golden_input={
        "returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018),
        "benchmark": (0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015),
    },
    golden_output=(1.3781,),
    pins=(
        SpecPin(
            label="null_misalignment_drops_pair",
            inputs={
                "returns": (0.01, None, 0.03, -0.01, 0.02),
                "benchmark": (0.008, -0.01, None, -0.005, 0.018),
            },
            expected=(1.669794280979366,),
            reason="a null in returns on one row and a null in benchmark on a different row each drop their pair "
            "(tests/metrics/test_capture_upside_ratio.py::test_null_misalignment_drops_pair)",
        ),
        SpecPin(
            label="nan_poisons_single_leg",
            inputs={"returns": (0.02, math.nan, 0.03, -0.01), "benchmark": (0.015, 0.01, 0.025, -0.008)},
            expected=(math.nan,),
            reason="a NaN in only one leg of a retained up-market pair poisons the whole scalar to NaN "
            "(tests/metrics/test_capture_upside_ratio.py::test_nan_poisons)",
        ),
        SpecPin(
            label="no_up_market_is_null",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (-0.01, -0.02, -0.03)},
            expected=(None,),
            reason="every pair is complete but no benchmark period is positive, so there is no up-market subset, null "
            "(tests/metrics/test_capture_upside_ratio.py::test_no_up_market_is_null)",
        ),
        SpecPin(
            label="return_below_negative_one_is_nan",
            inputs={"returns": (0.02, -1.5, 0.01), "benchmark": (0.01, 0.02, 0.03)},
            expected=(math.nan,),
            reason="a selected up-market return <= -1 is outside the geometric-growth domain, a loud NaN "
            "(tests/metrics/test_capture_upside_ratio.py::test_return_below_negative_one_is_nan)",
        ),
    ),
)
