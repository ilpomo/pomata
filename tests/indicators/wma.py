"""
Declaration for ``pomata.indicators.wma`` — the linear-weighted rolling mean, window-nulling, degree-1 homogeneous.
"""

import math

from pomata.indicators import wma
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_wma
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

WMA = suite_indicators(
    factory=wma,
    inputs=("expr",),
    params={"window": 3},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_wma,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)}, output=(None, None, 4.6667, 6.6667, 8.6667)),
    pins=(
        Pin(
            label="null_in_window_is_null",
            inputs={"expr": (1.0, None, 3.0, 4.0)},
            expected=(None, None, None, 11.0 / 3.0),
            params_override={"window": 2},
            reason="a null inside the window yields null there, the value returns once the window clears",
        ),
        Pin(
            label="nan_propagates",
            inputs={"expr": (1.0, math.nan, 3.0, 4.0)},
            expected=(None, math.nan, math.nan, 11.0 / 3.0),
            params_override={"window": 2},
            reason="a NaN inside the window yields NaN there and recovers once the window clears",
        ),
        Pin(
            label="window_equals_length",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(None, None, 14.0 / 6.0),
            params_override={"window": 3},
            reason="the single defined value when window exactly equals the series length",
        ),
        Pin(
            label="window_one_is_identity",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(1.0, 2.0, 3.0),
            params_override={"window": 1},
            reason="window=1 reproduces the input exactly, the single weight normalizes to one",
        ),
        Pin(
            label="recency_weighting",
            inputs={"expr": (1.0, 1.0, 4.0)},
            expected=(None, None, 15.0 / 6.0),
            params_override={"window": 3},
            reason="the recency lean: 1,1,4 with weights 1,2,3 gives 15/6 rather than the plain mean 2.0",
        ),
        Pin(
            label="interior_null_propagates",
            inputs={"expr": (2.0, 4.0, None, 8.0, 10.0, 12.0)},
            expected=(None, None, None, None, None, 64.0 / 6.0),
            reason="an interior null nulls every overlapping window and warm-up resumes after the gap",
        ),
        Pin(
            label="single_row_window_one",
            inputs={"expr": (42.0,)},
            expected=(42.0,),
            params_override={"window": 1},
            reason="a one-element series with window=1 returns the value itself",
        ),
        Pin(
            label="single_row_window_exceeds",
            inputs={"expr": (42.0,)},
            expected=(None,),
            reason="a one-element series with window > length is all warm-up",
        ),
        Pin(
            label="constant_series",
            inputs={"expr": (5.0, 5.0, 5.0, 5.0, 5.0)},
            expected=(None, None, 5.0, 5.0, 5.0),
            reason="the weights sum to one, so a WMA of a constant equals that constant on every defined row",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average",
    see_also=(
        ("sma", "The unweighted analog."),
        ("hma", "A low-lag average built by composing weighted means."),
        ("ema", "The exponentially-weighted analog."),
    ),
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        (
            "NaN",
            "a ``NaN`` inside the window propagates, yielding ``NaN`` there (``null`` takes precedence over ``NaN``).",
        ),
        ("Insufficient sample", "a series shorter than ``window`` observations, so the result is ``null``."),
        ("window == 1", "the single weight normalizes to one, so the WMA reproduces the input."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The WMA for each row, the same length as ``expr``. The first ``window - 1`` values are "
    "``null`` (warm-up) -- the value is defined only once ``window`` observations have been "
    "seen.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 1``.",
    },
    example_columns={"expr": "close"},
    examples=(
        Example(inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)}, params={"window": 3}, round_to=4),
        Example(
            inputs={"expr": (10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0, 22.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("A", "A", "A", "A", "A", "B", "B", "B", "B", "B"),
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={"expr": (10.0, 11.0, 12.0, 13.0, None, 15.0, float("nan"), 17.0, 18.0, 19.0)},
            intro="A ``null`` (skipped, and any window it touches yields ``null``) and a ``NaN`` (which "
            "propagates) make the exact handling visible at a glance:",
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={"expr": (42.0,)},
            intro="**Insufficient sample** — a one-element series holds only a single observation, and at "
            "``window=1`` that observation alone determines the weighted mean, so it returns the "
            "value itself:",
            params={"window": 1},
        ),
        Example(
            inputs={"expr": (1.0, 2.0, 3.0)},
            intro="**window == 1** — the single weight in the window normalizes to ``1``, so the WMA "
            "reproduces the input exactly:",
            params={"window": 1},
        ),
    ),
)
