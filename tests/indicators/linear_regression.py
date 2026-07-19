"""
Declaration for ``pomata.indicators.linear_regression`` — the rolling least-squares endpoint, window-nulling,
degree-1.
"""

from pomata.indicators import linear_regression
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_linear_regression
from tests.support.declaration import Example, Golden, ScaleAxis, Shape

LINEAR_REGRESSION = suite_indicators(
    factory=linear_regression,
    inputs=("expr",),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_linear_regression,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 1}, r"window must be >= 2"),),
    golden=Golden(
        inputs={"expr": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)},
        output=(None, None, 12.8333, 12.5, 13.5, 13.5, 14.5),
        params={"window": 3},
    ),
    reference="Chande, T. S. & Kroll, S. (1994). *The New Technical Trader*. Wiley.",
    wikipedia="https://en.wikipedia.org/wiki/Simple_linear_regression",
    see_also=(
        ("linear_regression_slope", "The slope of the same fitted line."),
        ("linear_regression_intercept", "The line's value at the oldest bar of the window."),
        ("time_series_forecast", "The line extrapolated one bar into the future."),
    ),
    note_extension="\n\n"
    "It is homogeneous of degree ``1`` in ``expr`` (a fitted price scales with the price). "
    "For a perfectly linear input the endpoint reproduces the series exactly.",
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The fitted endpoint for each row, the same length as the input. The first ``window - 1`` "
    "values are ``null`` (warm-up): the window must hold ``window`` non-null values before a "
    "result is emitted.",
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
