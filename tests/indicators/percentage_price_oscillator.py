"""Declaration for ``pomata.indicators.percentage_price_oscillator`` — the normalized EMA-difference, gap-bridging."""

import math

from pomata.indicators import percentage_price_oscillator
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_percentage_price_oscillator
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape
from tests.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE

PERCENTAGE_PRICE_OSCILLATOR = suite_indicators(
    factory=percentage_price_oscillator,
    inputs=("price",),
    params={"window_fast": 12, "window_slow": 26},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=25,
    oracle=reference_percentage_price_oscillator,
    scaling=(ScaleAxis(roles=("price",), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window_fast": 0}, r"window_fast must be >= 1"),
        ({"window_slow": 0}, r"window_slow must be >= 1"),
        ({"window_fast": 5, "window_slow": 3}, r"windows must be ordered window_fast <= window_slow"),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    golden=Golden(
        inputs={"price": (10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0)},
        output=(None, None, 4.5455, 1.5152, 3.2407, 3.5613, 1.1871, 2.7484),
        params={"window_fast": 2, "window_slow": 3},
    ),
    pins=(
        Pin(
            label="equal_windows_are_zero",
            inputs={"price": (10.0, 11.0, 12.0)},
            params_override={"window_fast": 2, "window_slow": 2},
            expected=(None, 0.0, 0.0),
            reason="equal fast/slow windows produce identical EMAs so the oscillator cancels to exactly 0 ",
        ),
        Pin(
            label="zero_slow_ema_is_nan",
            inputs={"price": (0.0, 0.0, 0.0, 0.0)},
            params_override={"window_fast": 2, "window_slow": 3},
            expected=(None, None, math.nan, math.nan),
            reason="an all-zero series drives both EMAs to exactly 0.0, so the 0/0 boundary surfaces as NaN ",
        ),
        Pin(
            label="nonzero_gap_zero_slow_ema_is_inf",
            inputs={"price": (1.0, 1.0, -2.0)},
            params_override={"window_fast": 2, "window_slow": 3},
            expected=(None, None, -math.inf),
            reason="a window summing to zero seeds the slow EMA at exactly 0.0 while the fast EMA stays non-zero, so "
            "the non-zero gap over the zero slow EMA is +/-inf — the infinity beside the 0/0 NaN pin",
        ),
    ),
    reference="Appel, G. (2005). *Technical Analysis: Power Tools for Active Investors*. FT Press.",
    see_also=(
        ("absolute_price_oscillator", "The same gap in price units, before dividing by the slow EMA."),
        ("macd", "The oscillator built on this gap, with an added signal line."),
        ("ema", "The exponential moving average each leg is built from."),
    ),
    notes=(
        (
            "Moving average",
            "Both legs use the exponential :func:`ema`. Being scale-free, PPO is invariant to the "
            "price's unit — multiplying the close by a constant leaves it unchanged.",
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
            "Degenerate denominator",
            "when both the gap and the slow EMA are ``0`` the ratio is indeterminate, so the result "
            "is a ``0 / 0``, i.e. ``NaN`` — a non-zero gap over a zero slow EMA is ``+/-inf``.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The oscillator (in percent) for each row, the same length as the input. Values are "
    "``null`` until both EMAs leave their warm-up (the first ``max(window_fast, window_slow) "
    "- 1`` rows).",
    raises_prose="ValueError: If ``window_fast < 1``, ``window_slow < 1``, or ``window_fast > "
    "window_slow`` (the fast leg must be the shorter one; ``window_fast == window_slow`` is "
    "allowed and gives an identically-zero oscillator).",
    args_prose={
        "window_fast": "Span of the fast EMA (canonically ``12``). Must be ``>= 1``.",
        "window_slow": "Span of the slow EMA (canonically ``26``). Must be ``>= 1`` and ``>= window_fast``.",
    },
    intro_basic="Basic usage on a single price series:",
    example_columns={"price": "close"},
    examples=(
        Example(
            inputs={"price": (10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0)},
            params={"window_fast": 2, "window_slow": 3},
            round_to=4,
        ),
        Example(
            inputs={"price": (10.0, 11.0, 12.0, 11.0, 20.0, 22.0, 24.0, 22.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker's EMAs warm up independently:",
            partition=("AAPL",) * 4 + ("NVDA",) * 4,
            params={"window_fast": 2, "window_slow": 3},
            round_to=4,
        ),
        Example(
            inputs={"price": (10.0, 11.0, None, 13.0, float("nan"), 15.0)},
            intro="A ``null`` (which the recursive EMA bridges) and a ``NaN`` (which latches) make the "
            "handling visible:",
            params={"window_fast": 2, "window_slow": 3},
            round_to=4,
        ),
        Example(
            inputs={"price": (0.0, 0.0, 0.0, 0.0)},
            intro="**Degenerate denominator** — an all-zero series drives both EMAs to exactly ``0.0``, so "
            "the ``0/0`` boundary surfaces as ``NaN``:",
            params={"window_fast": 2, "window_slow": 3},
        ),
        Example(
            inputs={"price": (1.0, 1.0, -2.0)},
            intro="**Degenerate denominator** — a window summing to zero seeds the slow EMA at exactly "
            "``0.0`` while the fast EMA stays non-zero, so the non-zero gap over the zero slow EMA is "
            "``+/-inf``:",
            params={"window_fast": 2, "window_slow": 3},
        ),
    ),
)
