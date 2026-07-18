"""
Declaration for ``pomata.indicators.accumulation_distribution_oscillator`` — the Chaikin A/D oscillator, gap-bridging.
"""

from pomata.indicators import accumulation_distribution_oscillator
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_accumulation_distribution_oscillator
from tests.support.declaration import Golden, ScaleAxis, Shape

ACCUMULATION_DISTRIBUTION_OSCILLATOR = suite_indicators(
    factory=accumulation_distribution_oscillator,
    inputs=("high", "low", "close", "volume"),
    params={"window_fast": 3, "window_slow": 10},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=9,
    oracle=reference_accumulation_distribution_oscillator,
    scaling=(
        ScaleAxis(roles=("volume",), degree=1),
        ScaleAxis(roles=("high", "low", "close"), degree=0),
    ),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window_fast": 0}, r"window_fast must be >= 1"),
        ({"window_slow": 0}, r"window_slow must be >= 1"),
        ({"window_fast": 10, "window_slow": 3}, r"windows must be ordered window_fast <= window_slow"),
    ),
    golden=Golden(
        inputs={
            "high": (10.2, 10.5, 10.7, 10.3, 10.8),
            "low": (9.8, 10.0, 10.2, 9.9, 10.3),
            "close": (10.0, 10.3, 10.5, 10.1, 10.6),
            "volume": (100.0, 150.0, 120.0, 200.0, 180.0),
        },
        output=(None, None, 13.0, 8.6667, 11.0556),
        params={"window_fast": 2, "window_slow": 3},
    ),
    reference='Chaikin, M. "Chaikin Oscillator."',
    wikipedia="https://en.wikipedia.org/wiki/Chaikin_Analytics#Chaikin_Oscillator",
    see_also=(
        ("accumulation_distribution", "The line this oscillates."),
        ("ema", "The exponential moving average of the two smoothings."),
        ("macd", "The same fast-minus-slow-EMA oscillator, applied to price."),
    ),
    notes=(
        (
            "Scaling",
            "Homogeneous of degree ``1`` — the accumulation/distribution multiplier is "
            "scale-invariant in price while the line scales with ``volume``, so multiplying all four "
            "inputs by ``k`` scales the oscillator by ``k``.",
        ),
    ),
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap.",
        ),
        (
            "NaN",
            "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null position.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The oscillator for each row, the same length as the inputs. The first ``window_slow - "
    "1`` values are ``null`` (warm-up), inherited from the slow :func:`ema` of the "
    "accumulation/distribution line, the later of the two to warm up.",
    raises_prose="ValueError: If ``window_fast < 1``, ``window_slow < 1``, or ``window_fast > window_slow``.",
    args_prose={
        "window_fast": "Span of the fast EMA (canonically ``3``). Must be ``>= 1``.",
        "window_slow": "Span of the slow EMA (canonically ``10``). Must be ``>= 1`` and ``>= window_fast``.",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
    intro_missing="A ``null`` (which voids the line and its EMAs) and a ``NaN`` (which propagates) make the "
    "exact handling visible at a glance:",
)
