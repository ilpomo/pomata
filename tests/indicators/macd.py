"""
Declaration for ``pomata.indicators.macd`` — the EMA-difference struct (line, signal, histogram), gap-bridging,
degree-1.
"""

import polars as pl

from pomata.indicators import absolute_price_oscillator, ema, macd
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_macd
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape
from tests.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE


def _macd_component() -> pl.Expr:
    """MACD recomposed from public functions: line = the APO, signal = its EMA, histogram = their difference."""
    line = absolute_price_oscillator(pl.col("expr"), window_fast=12, window_slow=26)
    signal = ema(line, 9)
    return pl.struct(macd=line, signal=signal, histogram=line - signal).name.keep()


MACD = suite_indicators(
    factory=macd,
    inputs=("expr",),
    params={"window_fast": 12, "window_slow": 26, "window_signal": 9},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.STRUCT,
    fields=("macd", "signal", "histogram"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"macd": 25, "signal": 33, "histogram": 33},
    oracle=reference_macd,
    scaling=(ScaleAxis(roles=("expr",), degree={"macd": 1, "signal": 1, "histogram": 1}),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window_fast": 0}, r"window_fast must be >= 1"),
        ({"window_slow": 0}, r"window_slow must be >= 1"),
        ({"window_signal": 0}, r"window_signal must be >= 1"),
        ({"window_fast": 26, "window_slow": 12}, r"windows must be ordered window_fast <= window_slow"),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    golden=Golden(
        inputs={"expr": (10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0)},
        output={
            "macd": (None, None, 0.5, 0.1667, 0.3889, 0.463, 0.1543, 0.3848),
            "signal": (None, None, None, 0.3333, 0.3704, 0.4321, 0.2469, 0.3388),
            "histogram": (None, None, None, -0.1667, 0.0185, 0.0309, -0.0926, 0.046),
        },
        params={"window_fast": 2, "window_slow": 3, "window_signal": 2},
    ),
    recomposition=_macd_component,
    pins=(
        Pin(
            label="fast_equals_slow_is_zero",
            inputs={"expr": (10.0, 11.0, 12.0, 13.0, 14.0, 15.0)},
            params_override={"window_fast": 3, "window_slow": 3, "window_signal": 2},
            expected={
                "macd": (None, None, 0.0, 0.0, 0.0, 0.0),
                "signal": (None, None, None, 0.0, 0.0, 0.0),
                "histogram": (None, None, None, 0.0, 0.0, 0.0),
            },
            reason="equal fast/slow windows make the MACD line identically zero (x - x is exact +0.0), so signal and "
            "histogram are zero too",
        ),
    ),
    reference="Appel, G. (2005). *Technical Analysis: Power Tools for Active Investors*. FT Press.",
    wikipedia="https://en.wikipedia.org/wiki/MACD",
    see_also=(
        ("absolute_price_oscillator", "The Absolute Price Oscillator, the MACD line without the signal and histogram."),
        ("percentage_price_oscillator", "The percentage counterpart of the MACD line."),
        ("ema", "The exponential moving average all three lines are built from."),
    ),
    notes=(
        (
            "Scaling",
            "Every field is homogeneous of degree ``1`` in ``expr`` (the EMAs and their differences "
            "all scale with the price), so multiplying the close by ``k`` scales all three fields by "
            "``k``.",
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
    returns_body="A struct ``pl.Expr`` with three ``Float64`` fields, the same length as ``expr``:"
    "\n\n"
    "- ``macd`` — the fast-minus-slow EMA gap, ``null`` for its ``max(window_fast, "
    "window_slow) - 1`` warm-up rows. - ``signal`` — the EMA of the MACD line, carrying the "
    "additional ``window_signal - 1`` warm-up rows on top. - ``histogram`` — ``macd`` minus "
    "``signal``, sharing the signal line's warm-up."
    "\n\n"
    'Access the fields with ``.struct.field("macd")`` / ``"signal"`` / ``"histogram"`` or '
    "``.struct.unnest()``.",
    raises_prose="ValueError: If ``window_fast < 1``, ``window_slow < 1``, ``window_signal < 1``, or "
    "``window_fast > window_slow`` (the fast leg must be the shorter one; ``window_fast == "
    "window_slow`` is allowed and gives an identically-zero MACD line, signal, and "
    "histogram).",
    args_prose={
        "window_fast": "Span of the fast EMA (canonically ``12``). Must be ``>= 1``.",
        "window_slow": "Span of the slow EMA (canonically ``26``). Must be ``>= 1`` and ``>= window_fast``.",
        "window_signal": "Span of the signal EMA over the MACD line (canonically ``9``). Must be ``>= 1``.",
    },
    intro_basic="Basic usage on a single price series:",
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker's EMAs warm up independently:",
    intro_missing="A ``null`` (which the recursive EMAs bridge) and a ``NaN`` (which latches) make the "
    "handling visible on the MACD line:",
)
