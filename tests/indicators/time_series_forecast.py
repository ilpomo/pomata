"""
Declaration for ``pomata.indicators.time_series_forecast`` — the one-step least-squares forecast, degree-1
homogeneous.
"""

from pomata.indicators import time_series_forecast
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_time_series_forecast
from tests.support.declaration import Example, Golden, ScaleAxis, Shape

TIME_SERIES_FORECAST = suite_indicators(
    factory=time_series_forecast,
    inputs=("expr",),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_time_series_forecast,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 1}, r"window must be >= 2"),),
    golden=Golden(
        inputs={"expr": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)},
        output=(None, None, 14.3333, 13.0, 14.0, 14.0, 15.0),
        params={"window": 3},
    ),
    reference="Chande, T. S. & Kroll, S. (1994). *The New Technical Trader*. Wiley.",
    wikipedia="https://en.wikipedia.org/wiki/Simple_linear_regression",
    see_also=(
        ("linear_regression", "The same line evaluated at the current bar rather than one ahead."),
        ("linear_regression_slope", "The slope used for the projection."),
        ("linear_regression_intercept", "The same line's value at the oldest bar of the window."),
    ),
    note_extension="\n\n"
    "It is homogeneous of degree ``1`` in ``expr`` (a projected price scales with the price). "
    "For a perfectly linear input the forecast equals the next value of the line exactly.",
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The one-step-ahead forecast for each row, the same length as the input. The first "
    "``window - 1`` values are ``null`` (warm-up).",
    raises_prose="ValueError: If ``window < 2``.",
    args_prose={
        "window": "Number of observations in the regression window. Must be ``>= 2`` (a line needs at least "
        "two points).",
    },
    example_columns={"expr": "x"},
    examples=(
        Example(inputs={"expr": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)}, params={"window": 3}, round_to=4),
        Example(
            inputs={"expr": (10.0, 11.0, 13.0, 12.0, 20.0, 22.0, 21.0, 24.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("A", "A", "A", "A", "B", "B", "B", "B"),
            params={"window": 3},
            round_to=4,
        ),
        Example(
            inputs={"expr": (10.0, 11.0, 13.0, None, 14.0, float("nan"), 16.0)},
            intro="A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make "
            "the handling visible:",
            params={"window": 3},
            round_to=4,
        ),
    ),
)
