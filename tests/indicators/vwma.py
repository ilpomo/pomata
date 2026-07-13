"""Spec for ``pomata.indicators.vwma`` — the volume-weighted rolling mean, window-nulling, degree-1 in price."""

import math

from tests.indicators.oracles import vwma_reference
from tests.support import ABSOLUTE_TOLERANCE_SCALE, RELATIVE_TOLERANCE_SCALE
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import vwma

VWMA = Spec(
    factory=vwma,
    inputs=("price", "volume"),
    params={"window": 3},
    shape=Shape.SERIES,
    warmup=2,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=vwma_reference,
    # A one-pass rolling volume-weighted sum against a two-pass oracle: a magnitude-proportional band.
    oracle_rel_tol=RELATIVE_TOLERANCE_SCALE,
    oracle_abs_tol=ABSOLUTE_TOLERANCE_SCALE,
    # Degree-1 homogeneous in price (a convex combination of the window's prices) and invariant to a positive common
    # rescaling of volume (tests/indicators/test_vwma.py::test_price_scale_homogeneity / test_volume_scale_invariance).
    scale=(
        ScaleAxis(roles=("price",), degree=1),
        ScaleAxis(roles=("volume",), degree=0),
    ),
    golden_input={
        "price": (10.0, 11.0, 12.0, 13.0, 14.0),
        "volume": (100.0, 200.0, 300.0, 400.0, 500.0),
    },
    golden_output=(None, None, 11.3333, 12.2222, 13.1667),
    pins=(
        SpecPin(
            label="single_row_window_one_identity",
            inputs={"price": (42.0,), "volume": (10.0,)},
            expected=(42.0,),
            params_override={"window": 1},
            reason="a one-row input with window=1 reproduces the price (test_vwma.py::test_single_row)",
        ),
        SpecPin(
            label="single_row_window_exceeds",
            inputs={"price": (42.0,), "volume": (10.0,)},
            expected=(None,),
            reason="a one-row input with window=3 never completes the window (test_vwma.py::test_single_row)",
        ),
        SpecPin(
            label="null_in_price_propagates",
            inputs={"price": (10.0, None, 12.0, 13.0, 14.0), "volume": (100.0, 200.0, 300.0, 400.0, 500.0)},
            expected=(None, None, None, 12.571428571428571, 13.555555555555555),
            params_override={"window": 2},
            reason="a null in price alone taints exactly the two windows it overlaps, then recovers "
            "(test_vwma.py::test_null_in_price_propagates); the shared flow rung cannot isolate a single-role null",
        ),
        SpecPin(
            label="null_in_volume_propagates",
            inputs={"price": (10.0, 11.0, 12.0, 13.0, 14.0), "volume": (100.0, None, 300.0, 400.0, 500.0)},
            expected=(None, None, None, 12.571428571428571, 13.555555555555555),
            params_override={"window": 2},
            reason="the isolated counterpart: a null in volume alone taints the same two windows "
            "(test_vwma.py::test_null_in_volume_propagates)",
        ),
        SpecPin(
            label="null_precedence_price_null_volume_nan",
            inputs={"price": (10.0, None, 12.0, 13.0), "volume": (100.0, 200.0, math.nan, 400.0)},
            expected=(None, None, None, None),
            params_override={"window": 3},
            reason="a null in price and a NaN in volume both reachable by the same window: null wins throughout "
            "(test_vwma.py::test_null_takes_precedence_over_nan)",
        ),
        SpecPin(
            label="nan_in_price_propagates",
            inputs={"price": (10.0, math.nan, 12.0, 13.0), "volume": (100.0, 200.0, 300.0, 400.0)},
            expected=(None, math.nan, math.nan, 12.571428571428571),
            params_override={"window": 2},
            reason="a NaN in price alone propagates through exactly the windows it overlaps "
            "(test_vwma.py::test_nan_in_price_propagates)",
        ),
        SpecPin(
            label="nan_in_volume_propagates",
            inputs={"price": (10.0, 11.0, 12.0, 13.0), "volume": (100.0, math.nan, 300.0, 400.0)},
            expected=(None, math.nan, math.nan, 12.571428571428571),
            params_override={"window": 2},
            reason="the isolated counterpart: a NaN in volume alone (test_vwma.py::test_nan_in_volume_propagates)",
        ),
        SpecPin(
            label="window_equals_length",
            inputs={"price": (2.0, 4.0, 6.0), "volume": (50.0, 50.0, 50.0)},
            expected=(None, None, 4.0),
            params_override={"window": 3},
            reason="window equal to the series length: exactly one defined value, the full-series volume-weighted mean "
            "(test_vwma.py::test_window_equals_length)",
        ),
        SpecPin(
            label="window_one_is_identity",
            inputs={"price": (10.0, 11.0, 12.0), "volume": (5.0, 6.0, 7.0)},
            expected=(10.0, 11.0, 12.0),
            params_override={"window": 1},
            reason="window=1 with non-zero volume reproduces price (test_vwma.py::test_window_one_is_identity)",
        ),
        SpecPin(
            label="equal_volume_reduces_to_sma",
            inputs={"price": (2.0, 4.0, 6.0, 8.0, 10.0), "volume": (50.0, 50.0, 50.0, 50.0, 50.0)},
            expected=(None, None, 4.0, 6.0, 8.0),
            params_override={"window": 3},
            reason="a constant volume in the window reduces VWMA to the SMA of price "
            "(test_vwma.py::test_equal_volume_reduces_to_sma)",
        ),
        SpecPin(
            label="zero_total_volume_is_nan",
            inputs={"price": (10.0, 11.0, 12.0), "volume": (0.0, 0.0, 0.0)},
            expected=(None, math.nan, math.nan),
            params_override={"window": 2},
            reason="an all-zero-volume window is the IEEE-754 0/0 degenerate "
            "(test_vwma.py::test_zero_total_volume_is_nan)",
        ),
        SpecPin(
            label="zero_volume_window_after_movement_is_nan",
            inputs={
                "price": (10.0, 50.0, 90.0, 1000.0, 2000.0, 3000.0),
                "volume": (0.1, 1.1, 1.1, 0.0, 0.0, 0.0),
            },
            expected=(None, None, 67.3913043478261, 70.0, 90.0, math.nan),
            params_override={"window": 3},
            reason="an all-zero-volume window following non-zero bars yields the exact NaN degenerate via the "
            "rolling-max(|volume|)==0 detector, not the +/-inf a rolling-sum residual would leak "
            "(test_vwma.py::test_zero_volume_window_after_movement_is_nan)",
        ),
    ),
)
