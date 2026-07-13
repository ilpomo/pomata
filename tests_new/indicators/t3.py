"""Spec for ``pomata.indicators.t3`` — Tillson's six-EMA smoother, gap-bridging, NaN-latching, degree-1 homogeneous."""

import math

from tests.indicators.oracles import t3_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import t3

T3 = Spec(
    factory=t3,
    inputs=("expr",),
    params={"window": 3, "volume_factor": 0.7},
    shape=Shape.SERIES,
    warmup=12,
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"volume_factor": math.nan}, r"volume_factor must be a finite number"),
        ({"volume_factor": math.inf}, r"volume_factor must be a finite number"),
        ({"volume_factor": -math.inf}, r"volume_factor must be a finite number"),
    ),
    oracle=t3_reference,
    # The coefficients c1+c2+c3+c4 = 1, so a pure rescale passes through linearly (tests/indicators/test_t3.py
    # ::TestT3Properties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("expr",), degree=1),),
    golden_params={"window": 2},
    golden_input={"expr": (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)},
    golden_output=(None, None, None, None, None, None, 13.1, 15.1, 17.1, 19.1),
    pins=(
        SpecPin(
            label="window_one_is_identity",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(1.0, 2.0, 3.0),
            params_override={"window": 1},
            reason="window=1 makes every EMA the identity and the coefficients sum to 1 "
            "(test_t3.py::TestT3Edge::test_window_one_is_identity)",
        ),
        SpecPin(
            label="single_row_window_one",
            inputs={"expr": (42.0,)},
            expected=(42.0,),
            params_override={"window": 1},
            reason="a one-row series with window=1 is the identity (test_t3.py::TestT3Edge::test_single_row)",
        ),
        SpecPin(
            label="single_row_window_two",
            inputs={"expr": (42.0,)},
            expected=(None,),
            params_override={"window": 2},
            reason="a one-row series with window=2 cannot complete any EMA pass "
            "(test_t3.py::TestT3Edge::test_single_row)",
        ),
        SpecPin(
            label="window_exceeds_series_length",
            inputs={"expr": (1.0, 2.0)},
            expected=(None, None),
            params_override={"window": 2},
            reason="a series shorter than the window warm-up yields an entirely null output "
            "(test_t3.py::TestT3Edge::test_window_fills_entire_series_with_warmup)",
        ),
        SpecPin(
            label="all_zero_series_is_zero",
            inputs={"expr": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
            expected=(None, None, None, None, None, None, 0.0, 0.0),
            params_override={"window": 2},
            reason="the degenerate all-zero window stays exactly at zero "
            "(test_t3.py::TestT3Edge::test_all_zero_series)",
        ),
        SpecPin(
            label="null_bridged",
            inputs={"expr": (1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)},
            expected=(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                7.565358126301888,
                8.557239987129336,
                9.552943668252915,
            ),
            params_override={"window": 2},
            reason="an early null extends the warm-up; the value resumes once all six EMA passes re-seed "
            "(test_t3.py::TestT3Edge::test_null_bridged)",
        ),
        SpecPin(
            label="nan_latches",
            inputs={"expr": (1.0, math.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)},
            expected=(None, None, None, None, None, None, math.nan, math.nan, math.nan, math.nan),
            params_override={"window": 2},
            reason="a NaN poisons the recursion and latches as exactly NaN for every subsequent value "
            "(test_t3.py::TestT3Edge::test_nan_latches)",
        ),
        SpecPin(
            label="interior_null_bridged",
            inputs={"expr": (2.0, 4.0, None, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)},
            expected=(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                15.14916985651142,
                17.11616757027457,
                19.104042047403723,
            ),
            params_override={"window": 2},
            reason="an interior null nulls its position while the recursion bridges the gap "
            "(test_t3.py::TestT3Edge::test_interior_null_bridged)",
        ),
        SpecPin(
            label="constant_series",
            inputs={"expr": (5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0)},
            expected=(None, None, None, None, None, None, 5.0, 5.0, 5.0, 5.0),
            params_override={"window": 2},
            reason="T3 of a constant equals that constant once warmed up (test_t3.py::TestT3Correctness"
            "::test_constant_series)",
        ),
        SpecPin(
            label="golden_master_adjusted",
            inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)},
            expected=(
                None,
                None,
                None,
                None,
                None,
                None,
                13.065715164239307,
                15.090278304462984,
                17.0981497650465,
                19.10020550466045,
            ),
            params_override={"window": 2, "adjust": True},
            reason="the frozen adjusted-mode golden master "
            "(test_t3.py::TestT3Correctness::test_golden_master_adjusted)",
        ),
    ),
)
