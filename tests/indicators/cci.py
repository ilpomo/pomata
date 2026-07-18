"""Declaration for ``pomata.indicators.cci`` — the commodity channel index, window-nulling, scale-invariant."""

import math

from pomata.indicators import cci
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_cci
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape
from tests.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE

CCI = suite_indicators(
    factory=cci,
    inputs=("high", "low", "close"),
    params={"window": 20},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_cci,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    golden=Golden(
        inputs={
            "high": (10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0),
            "low": (8.0, 9.0, 9.0, 10.0, 12.0, 11.0, 13.0, 14.0),
            "close": (9.0, 11.0, 10.0, 12.0, 14.0, 12.0, 15.0, 17.0),
        },
        output=(None, None, 12.5, 100.0, 100.0, -20.0, 90.9091, 89.4737),
        params={"window": 3},
    ),
    pins=(
        Pin(
            label="null_in_window_is_null",
            inputs={"high": (10.0, 12.0, 11.0, 13.0), "low": (8.0, 9.0, 9.0, 10.0), "close": (9.0, None, 10.0, 12.0)},
            params_override={"window": 2},
            expected=(None, None, None, 66.66666666666674),
            reason="a null in close taints exactly the windows that reach it, then recovers",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={
                "high": (10.0, 12.0, math.nan, 13.0),
                "low": (8.0, None, 9.0, 10.0),
                "close": (9.0, 11.0, 10.0, 12.0),
            },
            params_override={"window": 3},
            expected=(None, None, None, None),
            reason="a null and a NaN both reachable by the same window: null wins throughout",
        ),
        Pin(
            label="nan_propagates",
            inputs={
                "high": (10.0, 12.0, math.nan, 13.0, 15.0),
                "low": (8.0, 9.0, 9.0, 10.0, 12.0),
                "close": (9.0, 11.0, 10.0, 12.0, 14.0),
            },
            params_override={"window": 2},
            expected=(None, 66.66666666666674, math.nan, math.nan, 66.66666666666667),
            reason="a NaN in high propagates to NaN for the windows it reaches, then recovers",
        ),
        Pin(
            label="window_one_is_nan",
            inputs={"high": (10.0, 12.0, 11.0), "low": (8.0, 9.0, 9.0), "close": (9.0, 11.0, 10.0)},
            params_override={"window": 1},
            expected=(math.nan, math.nan, math.nan),
            reason="window=1 makes every window trivially flat, the 0/0 boundary",
        ),
        Pin(
            label="constant_series_is_nan",
            inputs={
                "high": (10.0, 10.0, 10.0, 10.0, 10.0),
                "low": (8.0, 8.0, 8.0, 8.0, 8.0),
                "close": (9.0, 9.0, 9.0, 9.0, 9.0),
            },
            params_override={"window": 3},
            expected=(None, None, math.nan, math.nan, math.nan),
            reason="a constant series gives a zero mean deviation, the flat-window 0/0 degenerate",
        ),
        Pin(
            label="all_zero_is_nan",
            inputs={
                "high": (0.0, 0.0, 0.0, 0.0, 0.0),
                "low": (0.0, 0.0, 0.0, 0.0, 0.0),
                "close": (0.0, 0.0, 0.0, 0.0, 0.0),
            },
            params_override={"window": 3},
            expected=(None, None, math.nan, math.nan, math.nan),
            reason="the exact-zero-denominator boundary",
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
            reason="an all-NaN series warms up to null then poisons to NaN, distinct from the all-null rung",
        ),
        Pin(
            label="window_equals_length",
            inputs={"high": (10.0, 12.0, 11.0), "low": (8.0, 9.0, 9.0), "close": (9.0, 11.0, 10.0)},
            params_override={"window": 3},
            expected=(None, None, 12.500000000000153),
            reason="window equal to the series length yields exactly one defined value",
        ),
        Pin(
            label="single_row_window_one",
            inputs={"high": (10.0,), "low": (8.0,), "close": (9.0,)},
            params_override={"window": 1},
            expected=(math.nan,),
            reason="a one-row window is trivially flat, NaN immediately",
        ),
        Pin(
            label="single_row_window_exceeds",
            inputs={"high": (10.0,), "low": (8.0,), "close": (9.0,)},
            params_override={"window": 3},
            expected=(None,),
            reason="a one-row input with window > length never completes a window",
        ),
    ),
    reference='Lambert, D. R. (1980). "Commodity Channel Index: Tools for Trading Cyclic Trends." '
    "*Commodities* (now *Futures*) magazine.",
    wikipedia="https://en.wikipedia.org/wiki/Commodity_channel_index",
    see_also=(
        ("price_typical", "The typical price the index is built on."),
        ("sma", "The simple moving average of the typical price it composes."),
        ("rsi", "A bounded momentum oscillator."),
    ),
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        ("Insufficient sample", "a window longer than the series never completes, so the result is ``null``."),
        (
            "Degenerate denominator",
            "when every typical price in the window is equal there is no spread to normalize by, so "
            "the result is a ``0 / 0``, i.e. ``NaN``, detected exactly via the rolling extremes (its "
            "rolling maximum equals its rolling minimum) rather than the rounding noise a sub-ULP "
            "denominator residual would produce.",
        ),
        (
            "window == 1",
            "every one-bar window is trivially flat, so every non-null result is ``NaN`` (a ``null`` "
            "row stays ``null``).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The CCI for each row, the same length as the inputs. The first ``window - 1`` values are "
    "``null`` (warm-up), inherited from the :func:`sma` of the typical price: the value is "
    "defined only once a full window of typical prices is available.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 1``.",
    },
    intro_basic="Basic usage on high-low-close bars:",
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
    intro_missing="A ``null`` and a ``NaN`` in ``close`` (each voiding every window that covers it) make "
    "the exact handling visible at a glance:",
)
