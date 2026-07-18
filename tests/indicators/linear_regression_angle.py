"""
Declaration for ``pomata.indicators.linear_regression_angle`` — the least-squares slope as a degree angle, scale-
exempt.
"""

from pomata.indicators import linear_regression_angle
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_linear_regression_angle
from tests.support.declaration import Golden, ScaleExempt, Shape

LINEAR_REGRESSION_ANGLE = suite_indicators(
    factory=linear_regression_angle,
    inputs=("expr",),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_linear_regression_angle,
    scaling=ScaleExempt(
        reason="atan(slope) in degrees: a rescaling scales the slope inside the arctangent, so the angle is neither "
        "invariant nor degree-1 homogeneous; it is a bounded O(1) value in (-90, 90)"
    ),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 1}, r"window must be >= 2"),),
    golden=Golden(
        inputs={"expr": (10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0)},
        output=(None, None, 56.3099, 26.5651, 26.5651, 26.5651, 26.5651),
        params={"window": 3},
    ),
    reference="Chande, T. S. & Kroll, S. (1994). *The New Technical Trader*. Wiley.",
    wikipedia="https://en.wikipedia.org/wiki/Simple_linear_regression",
    see_also=(
        ("linear_regression_slope", "The slope this takes the arctangent of."),
        ("linear_regression", "The fitted line's endpoint whose steepness this reports."),
        ("time_series_forecast", "The same line projected one bar ahead."),
    ),
    note_extension="\n\n"
    "Unlike the other regression outputs, the angle is **not** homogeneous in ``expr``: the "
    "arctangent is non-linear, so scaling the input does not scale the angle: amplifying it "
    "steepens the angle toward :math:`\\pm 90`, attenuating it flattens the angle toward "
    ":math:`0`. The angle depends on the numeric scale of ``expr`` versus its bar spacing, so "
    "it is most meaningful on a chart's own price/time units.",
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The angle in degrees for each row, the same length as the input, in :math:`(-90, 90)`. "
    "The first ``window - 1`` values are ``null`` (warm-up).",
    raises_prose="ValueError: If ``window < 2``.",
    args_prose={
        "window": "Number of observations in the regression window. Must be ``>= 2`` (a line needs at least "
        "two points).",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
    intro_missing="A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make "
    "the handling visible:",
)
