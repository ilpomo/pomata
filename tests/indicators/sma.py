"""Declaration for ``pomata.indicators.sma`` — the simple rolling mean, window-nulling, degree-1 homogeneous."""

from pomata.indicators import sma
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_sma
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

SMA = suite_indicators(
    factory=sma,
    inputs=("expr",),
    params={"window": 3},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_sma,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)}, output=(None, None, 4.0, 6.0, 8.0)),
    pins=(
        Pin(
            label="single_row_window_one_identity",
            inputs={"expr": (42.0,)},
            expected=(42.0,),
            params_override={"window": 1},
            reason="a one-element series with window=1 returns the value itself",
        ),
        Pin(
            label="single_row_window_exceeds_length",
            inputs={"expr": (42.0,)},
            expected=(None,),
            reason="a one-element series with window=3 is entirely warm-up",
        ),
        Pin(
            label="window_equals_length",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(None, None, 2.0),
            reason="a window equal to the series length emits exactly one defined value, the whole-series mean",
        ),
        Pin(
            label="window_one_is_identity",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(1.0, 2.0, 3.0),
            params_override={"window": 1},
            reason="window=1 reproduces the input with no warm-up",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average",
    see_also=(
        ("ema", "The exponentially-weighted analog, more responsive to recent values."),
        ("wma", "The linearly-weighted analog."),
        ("trima", "The triangular average, a simple average of a simple average."),
    ),
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        (
            "NaN",
            "a ``NaN`` inside the window propagates, yielding ``NaN`` there (``null`` takes precedence over ``NaN``).",
        ),
        ("Insufficient sample", "a series shorter than ``window`` observations, so the result is ``null``."),
        ("window == 1", "the one-point mean is the input itself, so the SMA reproduces the input."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The SMA for each row, the same length as ``expr``. The first ``window - 1`` values are "
    "``null`` (warm-up) -- the value is defined only once ``window`` observations have been "
    "seen.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 1``.",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
    intro_missing="A ``null`` (skipped, and any window it touches yields ``null``) and a ``NaN`` (which "
    "propagates) make the exact handling visible at a glance:",
)
