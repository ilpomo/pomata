"""Spec for ``pomata.indicators.rsi`` — Wilder's relative strength index, gap-bridging, NaN-latching."""

import math

from tests_new.indicators.oracles import rsi_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import rsi

RSI = Spec(
    factory=rsi,
    inputs=("expr",),
    params={"window": 3},
    shape=Shape.SERIES,
    warmup=3,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=rsi_reference,
    # A ratio of averaged gains to losses, invariant to a common rescaling, degree 0
    # (tests/indicators/test_rsi.py::TestRsiProperties::test_scale_invariance).
    scale=(ScaleAxis(roles=("expr",), degree=0),),
    golden_input={"expr": (44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42)},
    golden_output=(None, None, None, 7.0588, 59.0674, 74.1408, 80.0819, 85.8581),
    pins=(
        SpecPin(
            label="single_row_window_1",
            inputs={"expr": (42.0,)},
            params_override={"window": 1},
            expected=(None,),
            reason="a one-element series has no difference to seed the recursion "
            "(test_rsi.py::TestRsiEdge::test_single_row)",
        ),
        SpecPin(
            label="single_row_window_2",
            inputs={"expr": (42.0,)},
            params_override={"window": 2},
            expected=(None,),
            reason="the same fact restated at window=2 (test_rsi.py::TestRsiEdge::test_single_row)",
        ),
        SpecPin(
            label="window_one_is_move_direction",
            inputs={"expr": (1.0, 3.0, 2.0, 5.0)},
            params_override={"window": 1},
            expected=(None, 100.0, 0.0, 100.0),
            reason="window=1 collapses the Wilder smoothing to the raw move direction: 100 up, 0 down "
            "(test_rsi.py::TestRsiEdge::test_window_one_is_move_direction)",
        ),
        SpecPin(
            label="constant_series_is_nan",
            inputs={"expr": (5.0, 5.0, 5.0, 5.0)},
            params_override={"window": 2},
            expected=(None, None, math.nan, math.nan),
            reason="zero gain and zero loss is the indeterminate 0/0 relative strength, surfaced as NaN "
            "(test_rsi.py::TestRsiEdge::test_constant_series_is_nan)",
        ),
        SpecPin(
            label="monotone_increasing_is_hundred",
            inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)},
            params_override={"window": 2},
            expected=(None, None, 100.0, 100.0, 100.0),
            reason="a strictly increasing series (no losses) saturates RSI to exactly 100 "
            "(test_rsi.py::TestRsiEdge::test_monotone_increasing_is_hundred)",
        ),
        SpecPin(
            label="monotone_decreasing_is_zero",
            inputs={"expr": (10.0, 8.0, 6.0, 4.0, 2.0)},
            params_override={"window": 2},
            expected=(None, None, 0.0, 0.0, 0.0),
            reason="a strictly decreasing series (no gains) saturates RSI to exactly 0 "
            "(test_rsi.py::TestRsiEdge::test_monotone_decreasing_is_zero)",
        ),
        SpecPin(
            label="leading_null_defers_first_difference",
            inputs={"expr": (None, 2.0, 4.0, 6.0, 8.0)},
            params_override={"window": 2},
            expected=(None, None, None, 100.0, 100.0),
            reason="a leading null defers the first difference; the warm-up is measured from the first non-null value "
            "(test_rsi.py::TestRsiEdge::test_leading_null_defers_first_difference)",
        ),
    ),
)
