"""Declaration for ``pomata.indicators.linear_regression_intercept`` — the rolling least-squares intercept, degree-1."""

from pomata.indicators import linear_regression_intercept
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_linear_regression_intercept
from tests.support.declaration import Example, Golden, ScaleAxis, Shape

LINEAR_REGRESSION_INTERCEPT = suite_indicators(
    factory=linear_regression_intercept,
    inputs=("expr",),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_linear_regression_intercept,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 1}, r"window must be >= 2"),),
    golden=Golden(
        inputs={"expr": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)},
        output=(None, None, 9.8333, 11.5, 12.5, 12.5, 13.5),
        params={"window": 3},
    ),
    reference="Chande, T. S. & Kroll, S. (1994). *The New Technical Trader*. Wiley.",
    wikipedia="https://en.wikipedia.org/wiki/Simple_linear_regression",
    see_also=(
        ("linear_regression", "The same line evaluated at the most recent bar instead of the oldest."),
        ("linear_regression_slope", "The slope of the same fitted line."),
        ("time_series_forecast", "The same line projected one bar past the most recent."),
    ),
    note_extension="\n\nIt is homogeneous of degree ``1`` in ``expr`` (a fitted price scales with the price).",
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The fitted intercept for each row, the same length as the input. The first ``window - "
    "1`` values are ``null`` (warm-up).",
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
