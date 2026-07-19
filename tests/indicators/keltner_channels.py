"""Declaration for ``pomata.indicators.keltner_channels`` — the EMA-and-ATR band struct, gap-bridging, degree-1."""

import math

from pomata.indicators import keltner_channels
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_keltner_channels
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

KELTNER_CHANNELS = suite_indicators(
    factory=keltner_channels,
    inputs=("high", "low", "close"),
    params={"window": 20, "window_atr": 10, "multiplier": 2.0},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.STRUCT,
    fields=("lower", "middle", "upper"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"lower": 19, "middle": 19, "upper": 19},
    oracle=reference_keltner_channels,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree={"lower": 1, "middle": 1, "upper": 1}),),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no Keltner channels.",
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"window_atr": 0}, r"window_atr must be >= 1"),
        ({"multiplier": 0.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -1.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.nan}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.inf}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -math.inf}, r"multiplier must be a finite number > 0"),
    ),
    golden=Golden(
        inputs={
            "high": (10.0, 12.0, 11.0, 13.0, 15.0),
            "low": (8.0, 9.0, 9.5, 10.0, 12.0),
            "close": (9.0, 11.0, 10.0, 12.0, 14.0),
        },
        output={
            "lower": (None, None, 5.6667, 6.1111, 7.2407),
            "middle": (None, None, 10.0, 11.0, 12.5),
            "upper": (None, None, 14.3333, 15.8889, 17.7593),
        },
        params={"window": 3, "window_atr": 3},
    ),
    pins=(
        Pin(
            label="flat_range_zero_atr_collapses_to_ema",
            inputs={"high": (4.0, 4.0, 4.0, 4.0), "low": (4.0, 4.0, 4.0, 4.0), "close": (4.0, 4.0, 4.0, 4.0)},
            params_override={"window": 2, "window_atr": 2},
            expected={
                "lower": (None, 4.0, 4.0, 4.0),
                "middle": (None, 4.0, 4.0, 4.0),
                "upper": (None, 4.0, 4.0, 4.0),
            },
            reason="a flat series has zero ATR, so all three bands collapse onto the EMA of the close ",
        ),
    ),
    reference="Keltner, C. W. (1960). *How to Make Money in Commodities*.",
    wikipedia="https://en.wikipedia.org/wiki/Keltner_channel",
    see_also=(
        ("ema", "The midline."),
        ("atr", "The basis of the band half-width."),
        ("bollinger_bands", "The same idea with standard-deviation width instead of ATR."),
    ),
    notes=(
        (
            "Inputs",
            "``high``, ``low``, and ``close`` must share a length and alignment (the same row index "
            "is one bar). The original high-low band variant (Keltner's 1960 form) is not provided; "
            "compose it from :func:`ema` and the bar range if ever needed.",
        ),
    ),
    opener_override="Agrees with its independent reference oracle (a composition of the :func:`ema` and "
    ":func:`atr` references) to ten significant figures (a ``1e-10`` band); the "
    "documentation's *Correctness* page gives the method.",
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap — "
            "inherited from the recursive :func:`ema` (midline) and :func:`atr` (band width) legs, "
            "exactly as documented for each.",
        ),
        (
            "NaN",
            "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent "
            "non-null position — inherited from the same :func:`ema` and :func:`atr` legs, exactly as "
            "documented for each.",
        ),
        (
            "Degenerate denominator",
            "a constant ``high == low == close`` run has zero ATR, so the half-width vanishes and all "
            "three bands collapse onto the EMA.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A struct ``pl.Expr`` with three ``Float64`` fields, the same length as the inputs:"
    "\n\n"
    "- ``lower`` — the lower band, ``middle - multiplier * atr``. - ``middle`` — the center "
    "band, the :func:`ema` of ``close``. - ``upper`` — the upper band, ``middle + multiplier "
    "* atr``."
    "\n\n"
    'Read one band with ``.struct.field("middle")`` (etc.) or split all three into columns '
    "with ``.struct.unnest()``. Each band is ``null`` through its own warm-up: the midline's "
    "first ``window - 1`` rows, the outer bands' first ``max(window, window_atr) - 1`` rows "
    "(they also need the ATR).",
    raises_prose="ValueError: If ``window < 1``, ``window_atr < 1``, or ``multiplier`` is not a finite "
    "number ``> 0`` (i.e. ``<= 0``, ``NaN``, or ``±inf``).",
    args_prose={
        "window": "Number of observations in the EMA midline window (canonically ``20``). Must be ``>= 1``.",
        "window_atr": "Number of observations in the ATR window (canonically ``10``). Must be ``>= 1``.",
        "multiplier": "Band half-width as a multiple of the ATR (canonically ``2.0``). Must be a finite number "
        "``> 0`` (a non-positive multiplier would collapse or invert the bands).",
    },
    examples=(
        Example(
            inputs={"high": (3.0, 4.0, 5.0, 6.0), "low": (1.0, 2.0, 3.0, 4.0), "close": (2.0, 3.0, 4.0, 5.0)},
            params={"window": 2, "window_atr": 2},
            round_to=4,
            fields=("middle",),
        ),
        Example(
            inputs={
                "high": (3.0, 4.0, 5.0, 6.0, 13.0, 14.0, 15.0, 16.0),
                "low": (1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0),
                "close": (2.0, 3.0, 4.0, 5.0, 12.0, 13.0, 14.0, 15.0),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker's bands warm up independently:",
            partition=("AAPL",) * 4 + ("NVDA",) * 4,
            params={"window": 2, "window_atr": 2},
            round_to=4,
            fields=("middle",),
        ),
        Example(
            inputs={
                "high": (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
                "low": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
                "close": (2.0, 3.0, None, 5.0, float("nan"), 7.0, 8.0),
            },
            intro="A ``null`` (yields ``null`` at that row) and a ``NaN`` (which propagates) in ``close`` "
            "flow through the midline:",
            params={"window": 2, "window_atr": 2},
            round_to=4,
            fields=("middle",),
        ),
        Example(
            inputs={"high": (4.0, 4.0, 4.0, 4.0), "low": (4.0, 4.0, 4.0, 4.0), "close": (4.0, 4.0, 4.0, 4.0)},
            intro="**Degenerate denominator** — a flat series has zero ATR, so all three bands collapse "
            "onto the EMA of the close:",
            params={"window": 2, "window_atr": 2},
            fields=("lower",),
        ),
    ),
)
