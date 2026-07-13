"""Spec for ``pomata.indicators.cci`` — the commodity channel index, window-nulling, scale-invariant."""

import math

from tests_new.indicators.oracles import cci_reference
from tests_new.support import ABSOLUTE_TOLERANCE_SCALE, RELATIVE_TOLERANCE_SCALE
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import cci

CCI = Spec(
    factory=cci,
    inputs=("high", "low", "close"),
    params={"window": 20},
    shape=Shape.SERIES,
    warmup=19,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=cci_reference,
    # A normalized deviation from the rolling mean, scale-INVARIANT, degree 0 (tests/indicators/test_cci.py
    # ::test_scale_invariance).
    scale=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    # A one-pass rolling mean-absolute-deviation denominator against a two-pass oracle: a magnitude-proportional band.
    oracle_rel_tol=RELATIVE_TOLERANCE_SCALE,
    oracle_abs_tol=ABSOLUTE_TOLERANCE_SCALE,
    golden_params={"window": 3},
    golden_input={
        "high": (10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0),
        "low": (8.0, 9.0, 9.0, 10.0, 12.0, 11.0, 13.0, 14.0),
        "close": (9.0, 11.0, 10.0, 12.0, 14.0, 12.0, 15.0, 17.0),
    },
    golden_output=(None, None, 12.5, 100.0, 100.0, -20.0, 90.9091, 89.4737),
    pins=(
        SpecPin(
            label="null_in_window_is_null",
            inputs={"high": (10.0, 12.0, 11.0, 13.0), "low": (8.0, 9.0, 9.0, 10.0), "close": (9.0, None, 10.0, 12.0)},
            params_override={"window": 2},
            expected=(None, None, None, 66.66666666666674),
            reason="a null in close taints exactly the windows that reach it, then recovers "
            "(test_cci.py::TestCciEdge::test_null_in_window_is_null)",
        ),
        SpecPin(
            label="null_takes_precedence_over_nan",
            inputs={
                "high": (10.0, 12.0, math.nan, 13.0),
                "low": (8.0, None, 9.0, 10.0),
                "close": (9.0, 11.0, 10.0, 12.0),
            },
            params_override={"window": 3},
            expected=(None, None, None, None),
            reason="a null and a NaN both reachable by the same window: null wins throughout "
            "(test_cci.py::TestCciEdge::test_null_takes_precedence_over_nan)",
        ),
        SpecPin(
            label="nan_propagates",
            inputs={
                "high": (10.0, 12.0, math.nan, 13.0, 15.0),
                "low": (8.0, 9.0, 9.0, 10.0, 12.0),
                "close": (9.0, 11.0, 10.0, 12.0, 14.0),
            },
            params_override={"window": 2},
            expected=(None, 66.66666666666674, math.nan, math.nan, 66.66666666666667),
            reason="a NaN in high propagates to NaN for the windows it reaches, then recovers "
            "(test_cci.py::TestCciEdge::test_nan_propagates)",
        ),
        SpecPin(
            label="window_one_is_nan",
            inputs={"high": (10.0, 12.0, 11.0), "low": (8.0, 9.0, 9.0), "close": (9.0, 11.0, 10.0)},
            params_override={"window": 1},
            expected=(math.nan, math.nan, math.nan),
            reason="window=1 makes every window trivially flat, the 0/0 boundary (test_cci.py::TestCciEdge"
            "::test_window_one_is_nan)",
        ),
        SpecPin(
            label="constant_series_is_nan",
            inputs={
                "high": (10.0, 10.0, 10.0, 10.0, 10.0),
                "low": (8.0, 8.0, 8.0, 8.0, 8.0),
                "close": (9.0, 9.0, 9.0, 9.0, 9.0),
            },
            params_override={"window": 3},
            expected=(None, None, math.nan, math.nan, math.nan),
            reason="a constant series gives a zero mean deviation, the flat-window 0/0 degenerate "
            "(test_cci.py::TestCciEdge::test_constant_series_is_nan)",
        ),
        SpecPin(
            label="all_zero_is_nan",
            inputs={
                "high": (0.0, 0.0, 0.0, 0.0, 0.0),
                "low": (0.0, 0.0, 0.0, 0.0, 0.0),
                "close": (0.0, 0.0, 0.0, 0.0, 0.0),
            },
            params_override={"window": 3},
            expected=(None, None, math.nan, math.nan, math.nan),
            reason="the exact-zero-denominator boundary (test_cci.py::TestCciEdge::test_all_zero_is_nan)",
        ),
        SpecPin(
            label="all_nan",
            inputs={
                "high": (math.nan, math.nan, math.nan),
                "low": (math.nan, math.nan, math.nan),
                "close": (math.nan, math.nan, math.nan),
            },
            params_override={"window": 2},
            expected=(None, math.nan, math.nan),
            reason="an all-NaN series warms up to null then poisons to NaN, distinct from the all-null rung "
            "(test_cci.py::TestCciEdge::test_all_nan)",
        ),
        SpecPin(
            label="window_equals_length",
            inputs={"high": (10.0, 12.0, 11.0), "low": (8.0, 9.0, 9.0), "close": (9.0, 11.0, 10.0)},
            params_override={"window": 3},
            expected=(None, None, 12.500000000000153),
            reason="window equal to the series length yields exactly one defined value "
            "(test_cci.py::TestCciEdge::test_window_equals_length)",
        ),
        SpecPin(
            label="single_row_window_one",
            inputs={"high": (10.0,), "low": (8.0,), "close": (9.0,)},
            params_override={"window": 1},
            expected=(math.nan,),
            reason="a one-row window is trivially flat, NaN immediately (test_cci.py::TestCciEdge::test_single_row)",
        ),
        SpecPin(
            label="single_row_window_exceeds",
            inputs={"high": (10.0,), "low": (8.0,), "close": (9.0,)},
            params_override={"window": 3},
            expected=(None,),
            reason="a one-row input with window > length never completes a window (test_cci.py::TestCciEdge"
            "::test_single_row)",
        ),
    ),
)
