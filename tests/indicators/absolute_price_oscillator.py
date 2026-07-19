"""
Declaration for ``pomata.indicators.absolute_price_oscillator`` — the EMA-difference oscillator, gap-bridging,
degree-1.
"""

from pomata.indicators import absolute_price_oscillator
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_absolute_price_oscillator
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape
from tests.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE

ABSOLUTE_PRICE_OSCILLATOR = suite_indicators(
    factory=absolute_price_oscillator,
    inputs=("expr",),
    params={"window_fast": 2, "window_slow": 3},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=2,
    oracle=reference_absolute_price_oscillator,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window_fast": 0}, r"window_fast must be >= 1"),
        ({"window_slow": 0}, r"window_slow must be >= 1"),
        ({"window_fast": 5, "window_slow": 3}, r"windows must be ordered window_fast <= window_slow"),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    golden=Golden(
        inputs={"expr": (10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0)},
        output=(None, None, 0.5, 0.1667, 0.3889, 0.463, 0.1543, 0.3848),
    ),
    pins=(
        Pin(
            label="equal_windows_are_zero",
            inputs={"expr": (10.0, 11.0, 12.0)},
            expected=(None, 0.0, 0.0),
            params_override={"window_fast": 2, "window_slow": 2},
            reason="equal fast/slow windows make the two EMAs identical so the oscillator cancels to exactly 0.0",
        ),
    ),
    reference="Appel, G. (2005). *Technical Analysis: Power Tools for Active Investors*. FT Press.",
    see_also=(
        ("percentage_price_oscillator", "The same gap expressed as a percentage of the slow EMA."),
        ("macd", "The oscillator this line underlies, adding a signal and histogram."),
        ("ema", "The exponential moving average each leg is built from."),
    ),
    notes=(
        (
            "Moving average",
            "Both legs use the exponential :func:`ema` (not a simple average), so APO is the MACD "
            "line without the signal; compose :func:`sma` directly for a simple-average oscillator.",
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
    returns_body="The oscillator for each row, the same length as the input. Values are ``null`` until "
    "both EMAs leave their warm-up (the first ``max(window_fast, window_slow) - 1`` rows).",
    raises_prose="ValueError: If ``window_fast < 1``, ``window_slow < 1``, or ``window_fast > "
    "window_slow`` (the fast leg must be the shorter one; ``window_fast == window_slow`` is "
    "allowed and gives an identically-zero oscillator).",
    args_prose={
        "window_fast": "Span of the fast EMA (canonically ``12``). Must be ``>= 1``.",
        "window_slow": "Span of the slow EMA (canonically ``26``). Must be ``>= 1`` and ``>= window_fast``.",
    },
    intro_basic="Basic usage on a single price series:",
    example_columns={"expr": "close"},
    examples=(
        Example(
            inputs={"expr": (10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0)},
            params={"window_fast": 2, "window_slow": 3},
            round_to=4,
        ),
        Example(
            inputs={"expr": (10.0, 11.0, 12.0, 11.0, 20.0, 22.0, 24.0, 22.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker's EMAs warm up independently:",
            partition=("A", "A", "A", "A", "B", "B", "B", "B"),
            params={"window_fast": 2, "window_slow": 3},
            round_to=4,
        ),
        Example(
            inputs={"expr": (10.0, 11.0, None, 13.0, float("nan"), 15.0)},
            intro="A ``null`` (which the recursive EMA bridges) and a ``NaN`` (which latches) make the "
            "handling visible:",
            params={"window_fast": 2, "window_slow": 3},
            round_to=4,
        ),
    ),
)
