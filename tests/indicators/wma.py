"""Spec for ``pomata.indicators.wma`` — the linear-weighted rolling mean, window-nulling, degree-1 homogeneous."""

import math

from tests.indicators.oracles import wma_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import wma

WMA = Spec(
    factory=wma,
    inputs=("expr",),
    params={"window": 3},
    shape=Shape.SERIES,
    warmup=2,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=wma_reference,
    # A fixed linear-weight normalization scales linearly with the series (tests/indicators/test_wma.py
    # ::TestWmaProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("expr",), degree=1),),
    golden_input={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)},
    golden_output=(None, None, 4.6667, 6.6667, 8.6667),
    pins=(
        SpecPin(
            label="null_in_window_is_null",
            inputs={"expr": (1.0, None, 3.0, 4.0)},
            expected=(None, None, None, 11.0 / 3.0),
            params_override={"window": 2},
            reason="a null inside the window yields null there, the value returns once the window clears "
            "(test_wma.py::TestWmaEdge::test_null_in_window_is_null)",
        ),
        SpecPin(
            label="nan_propagates",
            inputs={"expr": (1.0, math.nan, 3.0, 4.0)},
            expected=(None, math.nan, math.nan, 11.0 / 3.0),
            params_override={"window": 2},
            reason="a NaN inside the window yields NaN there and recovers once the window clears "
            "(test_wma.py::TestWmaEdge::test_nan_propagates)",
        ),
        SpecPin(
            label="window_equals_length",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(None, None, 14.0 / 6.0),
            params_override={"window": 3},
            reason="the single defined value when window exactly equals the series length "
            "(test_wma.py::TestWmaEdge::test_window_equals_length)",
        ),
        SpecPin(
            label="window_one_is_identity",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(1.0, 2.0, 3.0),
            params_override={"window": 1},
            reason="window=1 reproduces the input exactly, the single weight normalizes to one "
            "(test_wma.py::TestWmaEdge::test_window_one_is_identity)",
        ),
        SpecPin(
            label="recency_weighting",
            inputs={"expr": (1.0, 1.0, 4.0)},
            expected=(None, None, 15.0 / 6.0),
            params_override={"window": 3},
            reason="the recency lean: 1,1,4 with weights 1,2,3 gives 15/6 rather than the plain mean 2.0 "
            "(test_wma.py::TestWmaEdge::test_recency_weighting)",
        ),
        SpecPin(
            label="interior_null_propagates",
            inputs={"expr": (2.0, 4.0, None, 8.0, 10.0, 12.0)},
            expected=(None, None, None, None, None, 64.0 / 6.0),
            reason="an interior null nulls every overlapping window and warm-up resumes after the gap "
            "(test_wma.py::TestWmaEdge::test_interior_null_propagates)",
        ),
        SpecPin(
            label="single_row_window_one",
            inputs={"expr": (42.0,)},
            expected=(42.0,),
            params_override={"window": 1},
            reason="a one-element series with window=1 returns the value itself "
            "(test_wma.py::TestWmaEdge::test_single_row, first assertion)",
        ),
        SpecPin(
            label="single_row_window_exceeds",
            inputs={"expr": (42.0,)},
            expected=(None,),
            reason="a one-element series with window > length is all warm-up "
            "(test_wma.py::TestWmaEdge::test_single_row, second assertion)",
        ),
        SpecPin(
            label="constant_series",
            inputs={"expr": (5.0, 5.0, 5.0, 5.0, 5.0)},
            expected=(None, None, 5.0, 5.0, 5.0),
            reason="the weights sum to one, so a WMA of a constant equals that constant on every defined row "
            "(test_wma.py::TestWmaProperties::test_constant_series)",
        ),
    ),
)
