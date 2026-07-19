"""
Declaration for ``pomata.indicators.linear_regression_slope`` — the rolling least-squares slope, degree-1 homogeneous.
"""

from pomata.indicators import linear_regression_slope
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_linear_regression_slope
from tests.support.declaration import Example, Golden, ScaleAxis, Shape

LINEAR_REGRESSION_SLOPE = suite_indicators(
    factory=linear_regression_slope,
    inputs=("expr",),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_linear_regression_slope,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 1}, r"window must be >= 2"),),
    golden=Golden(
        inputs={"expr": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)},
        output=(None, None, 1.5, 0.5, 0.5, 0.5, 0.5),
        params={"window": 3},
    ),
    reference="Chande, T. S. & Kroll, S. (1994). *The New Technical Trader*. Wiley.",
    wikipedia="https://en.wikipedia.org/wiki/Simple_linear_regression",
    see_also=(
        ("linear_regression", "The fitted line's value at the most recent bar."),
        ("linear_regression_angle", "This slope expressed as an angle in degrees."),
        ("time_series_forecast", "The line projected one bar ahead using this slope."),
    ),
    note_extension="\n\n"
    "It is homogeneous of degree ``1`` in ``expr`` (the rise scales with the price while the "
    "run is fixed). For a perfectly linear input it returns the exact constant slope.",
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The fitted slope for each row, the same length as the input. The first ``window - 1`` "
    "values are ``null`` (warm-up).",
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
            partition=("AAPL",) * 4 + ("NVDA",) * 4,
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
