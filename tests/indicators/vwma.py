"""Declaration for ``pomata.indicators.vwma`` — the volume-weighted rolling mean, window-nulling, degree-1 in price."""

import math

from pomata.indicators import vwma
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_vwma
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

VWMA = suite_indicators(
    factory=vwma,
    inputs=("price", "volume"),
    params={"window": 3},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_vwma,
    scaling=(
        ScaleAxis(roles=("price",), degree=1),
        ScaleAxis(roles=("volume",), degree=0),
    ),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no volume-weighted moving average.",
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "price": (10.0, 11.0, 12.0, 13.0, 14.0),
            "volume": (100.0, 200.0, 300.0, 400.0, 500.0),
        },
        output=(None, None, 11.3333, 12.2222, 13.1667),
    ),
    pins=(
        Pin(
            label="single_row_window_one_identity",
            inputs={"price": (42.0,), "volume": (10.0,)},
            expected=(42.0,),
            params_override={"window": 1},
            reason="a one-row input with window=1 reproduces the price",
        ),
        Pin(
            label="single_row_window_exceeds",
            inputs={"price": (42.0,), "volume": (10.0,)},
            expected=(None,),
            reason="a one-row input with window=3 never completes the window",
        ),
        Pin(
            label="null_in_price_propagates",
            inputs={"price": (10.0, None, 12.0, 13.0, 14.0), "volume": (100.0, 200.0, 300.0, 400.0, 500.0)},
            expected=(None, None, None, 12.571428571428571, 13.555555555555555),
            params_override={"window": 2},
            reason="a null in price alone taints exactly the two windows it overlaps, then recovers "
            "; the shared flow rung cannot isolate a single-role null",
        ),
        Pin(
            label="null_in_volume_propagates",
            inputs={"price": (10.0, 11.0, 12.0, 13.0, 14.0), "volume": (100.0, None, 300.0, 400.0, 500.0)},
            expected=(None, None, None, 12.571428571428571, 13.555555555555555),
            params_override={"window": 2},
            reason="the isolated counterpart: a null in volume alone taints the same two windows ",
        ),
        Pin(
            label="null_precedence_price_null_volume_nan",
            inputs={"price": (10.0, None, 12.0, 13.0), "volume": (100.0, 200.0, math.nan, 400.0)},
            expected=(None, None, None, None),
            params_override={"window": 3},
            reason="a null in price and a NaN in volume both reachable by the same window: null wins throughout ",
        ),
        Pin(
            label="nan_in_price_propagates",
            inputs={"price": (10.0, math.nan, 12.0, 13.0), "volume": (100.0, 200.0, 300.0, 400.0)},
            expected=(None, math.nan, math.nan, 12.571428571428571),
            params_override={"window": 2},
            reason="a NaN in price alone propagates through exactly the windows it overlaps ",
        ),
        Pin(
            label="nan_in_volume_propagates",
            inputs={"price": (10.0, 11.0, 12.0, 13.0), "volume": (100.0, math.nan, 300.0, 400.0)},
            expected=(None, math.nan, math.nan, 12.571428571428571),
            params_override={"window": 2},
            reason="the isolated counterpart: a NaN in volume alone",
        ),
        Pin(
            label="window_equals_length",
            inputs={"price": (2.0, 4.0, 6.0), "volume": (50.0, 50.0, 50.0)},
            expected=(None, None, 4.0),
            params_override={"window": 3},
            reason="window equal to the series length: exactly one defined value, the full-series volume-weighted mean "
            "",
        ),
        Pin(
            label="window_one_is_identity",
            inputs={"price": (10.0, 11.0, 12.0), "volume": (5.0, 6.0, 7.0)},
            expected=(10.0, 11.0, 12.0),
            params_override={"window": 1},
            reason="window=1 with non-zero volume reproduces price",
        ),
        Pin(
            label="equal_volume_reduces_to_sma",
            inputs={"price": (2.0, 4.0, 6.0, 8.0, 10.0), "volume": (50.0, 50.0, 50.0, 50.0, 50.0)},
            expected=(None, None, 4.0, 6.0, 8.0),
            params_override={"window": 3},
            reason="a constant volume in the window reduces VWMA to the SMA of price ",
        ),
        Pin(
            label="zero_total_volume_is_nan",
            inputs={"price": (10.0, 11.0, 12.0), "volume": (0.0, 0.0, 0.0)},
            expected=(None, math.nan, math.nan),
            params_override={"window": 2},
            reason="an all-zero-volume window is the IEEE-754 0/0 degenerate ",
        ),
        Pin(
            label="zero_volume_window_after_movement_is_nan",
            inputs={
                "price": (10.0, 50.0, 90.0, 1000.0, 2000.0, 3000.0),
                "volume": (0.1, 1.1, 1.1, 0.0, 0.0, 0.0),
            },
            expected=(None, None, 67.3913043478261, 70.0, 90.0, math.nan),
            params_override={"window": 3},
            reason="an all-zero-volume window following non-zero bars yields the exact NaN degenerate via the "
            "rolling-max(|volume|)==0 detector, not the +/-inf a rolling-sum residual would leak "
            "",
        ),
    ),
)
