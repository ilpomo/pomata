"""
Declaration for ``pomata.indicators.williams_r`` — Williams %R, the window-nulling bounded oscillator, scale-
invariant.
"""

import math

from pomata.indicators import williams_r
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_williams_r
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

WILLIAMS_R = suite_indicators(
    factory=williams_r,
    inputs=("high", "low", "close"),
    params={"window": 3},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_williams_r,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.0, 12.0, 11.0, 13.0, 15.0, 14.0),
            "low": (8.0, 9.0, 10.0, 11.0, 12.0, 13.0),
            "close": (9.0, 11.0, 10.5, 12.0, 14.0, 13.5),
        },
        output=(None, None, -37.5, -25.0, -20.0, -37.5),
    ),
    pins=(
        Pin(
            label="single_row_window_one",
            inputs={"high": (10.0,), "low": (8.0,), "close": (9.0,)},
            params_override={"window": 1},
            expected=(-50.0,),
            reason="a single bar with window=1",
        ),
        Pin(
            label="single_row_window_two",
            inputs={"high": (10.0,), "low": (8.0,), "close": (9.0,)},
            params_override={"window": 2},
            expected=(None,),
            reason="a single bar with window=2 exceeds the length",
        ),
        Pin(
            label="window_equals_length",
            inputs={"high": (10.0, 12.0, 11.0), "low": (8.0, 9.0, 10.0), "close": (9.0, 11.0, 10.5)},
            expected=(None, None, -37.5),
            reason="window equal to the series length yields one defined value",
        ),
        Pin(
            label="window_one_single_bar",
            inputs={"high": (10.0, 12.0), "low": (8.0, 9.0), "close": (9.0, 11.0)},
            params_override={"window": 1},
            expected=(-50.0, -33.333333333333336),
            reason="window=1 collapses HH/LL to the single bar's own high/low",
        ),
        Pin(
            label="close_at_high_is_zero",
            inputs={"high": (10.0, 12.0, 11.0), "low": (8.0, 9.0, 10.0), "close": (10.0, 12.0, 12.0)},
            params_override={"window": 2},
            expected=(None, 0.0, 0.0),
            reason="close at the windowed highest high gives %R == 0, the overbought edge",
        ),
        Pin(
            label="close_at_low_is_minus_hundred",
            inputs={"high": (10.0, 12.0, 11.0), "low": (8.0, 9.0, 10.0), "close": (8.0, 8.0, 9.0)},
            params_override={"window": 2},
            expected=(None, -100.0, -100.0),
            reason="close at the windowed lowest low gives %R == -100, the oversold edge",
        ),
        Pin(
            label="constant_range_zero_over_zero_is_nan",
            inputs={"high": (5.0, 5.0, 5.0), "low": (5.0, 5.0, 5.0), "close": (5.0, 5.0, 5.0)},
            params_override={"window": 2},
            expected=(None, math.nan, math.nan),
            reason="a flat window with close on that level is the 0/0 IEEE degenerate",
        ),
        Pin(
            label="constant_range_nonzero_numerator_is_inf",
            inputs={"high": (5.0, 5.0), "low": (5.0, 5.0), "close": (3.0, 3.0)},
            params_override={"window": 2},
            expected=(None, -math.inf),
            reason="a flat window with close off that level is a non-zero numerator over a zero denominator, signed "
            "inf",
        ),
        Pin(
            label="all_zero_series_is_nan",
            inputs={"high": (0.0, 0.0, 0.0), "low": (0.0, 0.0, 0.0), "close": (0.0, 0.0, 0.0)},
            params_override={"window": 2},
            expected=(None, math.nan, math.nan),
            reason="an all-zero series collapses the range to zero, 0/0 NaN",
        ),
        Pin(
            label="all_nan",
            inputs={
                "high": (math.nan, math.nan, math.nan),
                "low": (math.nan, math.nan, math.nan),
                "close": (math.nan, math.nan, math.nan),
            },
            params_override={"window": 2},
            expected=(None, math.nan, math.nan),
            reason="an all-NaN input warms up to null then poisons to NaN, distinct from the all-null rung",
        ),
    ),
)
