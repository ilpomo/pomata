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
    reference="Williams, L. (1973). *How I Made One Million Dollars Last Year Trading Commodities*.",
    wikipedia="https://en.wikipedia.org/wiki/Williams_%25R",
    see_also=(
        ("stochastic_fast", "The Fast Stochastic %K this oscillator inverts."),
        ("rsi", "A bounded momentum oscillator on the same [0, 100]-style scale."),
        ("cci", "Another bounded oscillator over a rolling window."),
    ),
    notes=(
        (
            "Warm-up",
            "The warm-up is the canonical ``window - 1`` leading nulls of the rolling family, and the "
            "``null`` / ``NaN`` contract below matches the simple moving average.",
        ),
    ),
    bullets=(
        (
            "Null",
            "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null "
            "values) — this covers ``high`` and ``low``; the ``close`` enters elementwise, so a "
            "``null`` ``close`` nulls only its own bar (``null`` takes precedence over ``NaN``).",
        ),
        (
            "NaN",
            "a ``NaN`` inside the window propagates, yielding ``NaN`` there; a ``NaN`` ``close`` "
            "yields ``NaN`` only at its own bar.",
        ),
        ("Insufficient sample", "a window longer than the series never completes, so the result is ``null``."),
        (
            "Degenerate denominator",
            "when the windowed range collapses (:math:`\\mathrm{HH} = \\mathrm{LL}`, e.g. a flat "
            "high-low over the whole window) with the close on that level the ratio is indeterminate, "
            "so the result is a ``0 / 0``, i.e. ``NaN`` — a non-zero numerator over the zero range is "
            "``+/-inf``.",
        ),
        (
            "window == 1",
            "the highest high and lowest low collapse to the single bar's own ``high`` and ``low``, "
            "so :math:`\\%R = -100\\,(H - C) / (H - L)`.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The Williams %R for each row, the same length as the inputs. The first ``window - 1`` "
    "values are ``null`` (warm-up), matching the rolling moving-average family: the value is "
    "defined only once ``window`` observations have been seen.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 1``.",
    },
    intro_basic="Basic usage on high-low-close bars:",
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
    intro_missing="A ``null`` and a ``NaN`` in ``close`` (each confined to its own bar, since the close "
    "enters elementwise) make the exact handling visible at a glance:",
)
