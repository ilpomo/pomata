"""
Declaration for ``pomata.indicators.midpoint`` — the rolling high/low midpoint of a series, window-nulling, degree-1.
"""

from pomata.indicators import midpoint
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_midpoint
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

MIDPOINT = suite_indicators(
    factory=midpoint,
    inputs=("expr",),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_midpoint,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={"expr": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)}, output=(None, None, 2.0, 3.0, 4.0, 5.0), params={"window": 3}
    ),
    pins=(
        Pin(
            label="window_one_is_identity",
            inputs={"expr": (1.0, 2.0, 3.0)},
            params_override={"window": 1},
            expected=(1.0, 2.0, 3.0),
            reason="window=1 makes the max and min the value itself, so the midpoint reproduces the input with no "
            "warm-up",
        ),
    ),
    reference="No canonical external source; the indicator is defined by the formula above.",
    see_also=(
        ("midprice", "The same midpoint taken across a bar's high and low instead of one series."),
        ("sma", "The moving mean of the window, which uses every value rather than only the extremes."),
        ("donchian_channels", "The high-low band system built from the same rolling extremes."),
    ),
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        ("window == 1", "the max and min are the single value, so the midpoint reproduces the input."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The window midpoint for each row, the same length as the input. The first ``window - 1`` "
    "values are ``null`` (warm-up): the window must hold ``window`` non-null values before a "
    "result is emitted.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 1``.",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
    intro_missing="A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make "
    "the handling visible:",
)
