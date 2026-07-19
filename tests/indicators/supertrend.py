"""Declaration for ``pomata.indicators.supertrend`` — the ATR-banded trend struct (line, direction), gap-bridging."""

import math

from pomata.indicators import supertrend
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_supertrend
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

_GOLDEN_HIGH = (10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0)

_GOLDEN_LOW = (9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0)

_GOLDEN_CLOSE = (9.5, 10.5, 11.5, 12.5, 13.5, 12.0, 11.0, 10.2)

SUPERTREND = suite_indicators(
    factory=supertrend,
    inputs=("high", "low", "close"),
    params={"window": 10, "multiplier": 3.0},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.STRUCT,
    fields=("line", "direction"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"line": 9, "direction": 9},
    oracle=reference_supertrend,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree={"line": 1, "direction": 0}),),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no SuperTrend.",
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"multiplier": 0.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -1.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.nan}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.inf}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -math.inf}, r"multiplier must be a finite number > 0"),
    ),
    golden=Golden(
        inputs={"high": _GOLDEN_HIGH, "low": _GOLDEN_LOW, "close": _GOLDEN_CLOSE},
        output={
            "line": (None, None, 8.8333, 9.7222, 10.6481, 10.6481, 10.6481, 12.9005),
            "direction": (None, None, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0),
        },
        params={"window": 3, "multiplier": 2.0},
    ),
    pins=(
        Pin(
            label="flat_series_zero_atr_collapses_to_midpoint",
            inputs={
                "high": (5.0, 5.0, 5.0, 5.0, 5.0),
                "low": (5.0, 5.0, 5.0, 5.0, 5.0),
                "close": (5.0, 5.0, 5.0, 5.0, 5.0),
            },
            params_override={"window": 2},
            expected={
                "line": (None, 5.0, 5.0, 5.0, 5.0),
                "direction": (None, -1.0, -1.0, -1.0, -1.0),
            },
            reason="a constant high==low==close run has zero ATR, so both bands collapse onto the midpoint and the "
            "line tracks it with direction -1 forever",
        ),
        Pin(
            label="single_row_window_one_seeds_long",
            inputs={"high": (10.0,), "low": (8.0,), "close": (9.0,)},
            params_override={"window": 1},
            expected={"line": (3.0,), "direction": (1.0,)},
            reason="window=1 defines the bar (ATR is the true range); close > lower seeds the trend long and the line "
            "reads the lower band",
        ),
        Pin(
            label="lower_band_exact_touch_stays_up",
            inputs={"high": (11.0, 13.0, 12.5), "low": (9.0, 11.0, 10.5), "close": (10.5, 12.0, 11.375)},
            params_override={"window": 1, "multiplier": 0.25},
            expected={"line": (9.5, 11.375, 11.375), "direction": (1.0, 1.0, 1.0)},
            reason="a flip requires a strict break, so a close exactly on the carried band holds the trend",
        ),
        Pin(
            label="downtrend_seed_nondefault_multiplier_golden",
            inputs={
                "high": (20.0, 19.0, 18.0, 17.0, 18.0, 19.0, 20.0, 21.0),
                "low": (19.0, 18.0, 17.0, 16.0, 17.0, 18.0, 19.0, 20.0),
                "close": (19.2, 18.2, 17.2, 16.2, 17.8, 18.8, 19.8, 20.8),
            },
            params_override={"window": 2, "multiplier": 1.0},
            expected={
                "line": (None, 17.4, 18.65, 17.675, 16.0125, 17.15625, 18.228125000000002, 19.2640625),
                "direction": (None, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0),
            },
            reason="a second frozen reference exercising every branch (seed long, flip short, flip back long) at a "
            "non-default multiplier",
        ),
        Pin(
            label="large_magnitude_micro_scale",
            inputs={
                "high": tuple(v * 1e-6 for v in _GOLDEN_HIGH),
                "low": tuple(v * 1e-6 for v in _GOLDEN_LOW),
                "close": tuple(v * 1e-6 for v in _GOLDEN_CLOSE),
            },
            params_override={"window": 3, "multiplier": 2.0},
            expected={
                "line": (
                    None,
                    None,
                    8.833333333333332e-06,
                    9.722222222222221e-06,
                    1.0648148148148146e-05,
                    1.0648148148148146e-05,
                    1.0648148148148146e-05,
                    1.2900548696844994e-05,
                ),
                "direction": (None, None, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0),
            },
            reason="the golden scaled by 1e-6 stays exact, the small-magnitude extreme of the numeric range",
        ),
        Pin(
            label="large_magnitude_macro_scale",
            inputs={
                "high": tuple(v * 1e9 for v in _GOLDEN_HIGH),
                "low": tuple(v * 1e9 for v in _GOLDEN_LOW),
                "close": tuple(v * 1e9 for v in _GOLDEN_CLOSE),
            },
            params_override={"window": 3, "multiplier": 2.0},
            expected={
                "line": (
                    None,
                    None,
                    8833333333.333334,
                    9722222222.222221,
                    10648148148.148148,
                    10648148148.148148,
                    10648148148.148148,
                    12900548696.844994,
                ),
                "direction": (None, None, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0),
            },
            reason="the golden scaled by 1e9, pinned against a regression to inf/NaN at the large-magnitude extreme",
        ),
    ),
    reference="Seban, O. (2009). *Tout le monde mérite d'être riche*.",
    see_also=(
        ("parabolic_sar", "The other trailing-stop trend tool, accelerating rather than ATR-scaled."),
        ("atr", "The volatility average that sets the band half-width."),
        ("keltner_channels", "The other ATR-scaled band envelope, centered on an EMA rather than ratcheting."),
    ),
    notes=(
        (
            "Tie-break and seeding",
            "A flip needs a *strict* cross, so a close exactly on the active band holds the current "
            "trend; over a flat series the bands collapse onto the midpoint and the line tracks it. "
            "The trend seeds short when the first valid close is at or below the lower band, else "
            "long -- chosen so the line sits on the correct side of price from row one.",
        ),
    ),
    note_extension="\n\n"
    "The ``line`` is homogeneous of degree ``1`` under a positive common rescaling of "
    "``high`` / ``low`` / ``close`` (a price level), while ``direction`` is scale-invariant "
    "(the crossings compare like-scaled quantities).",
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap "
            "(on both struct fields, the running state and the last valid close the ratchet reads "
            "bridging it).",
        ),
        (
            "NaN",
            "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent "
            "non-null position (on both struct fields; a later ``null`` row shows ``null`` there only "
            "— nothing flushes the poisoned state).",
        ),
        ("window == 1", "the ATR has no memory term, so a ``NaN`` self-heals once the true range is finite again."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A struct ``pl.Expr`` with two ``Float64`` fields, the same length as the inputs:"
    "\n\n"
    "- ``line`` — the trailing stop. - ``direction`` — ``+1.0`` in an up-trend (the line "
    "below price); ``-1.0`` in a down-trend (the line above price)."
    "\n\n"
    "The first ``window - 1`` rows are ``null`` (the ATR's warm-up). Read a field with "
    '``.struct.field("line")`` or split both with ``.struct.unnest()``.',
    raises_prose="ValueError: If ``window < 1`` or ``multiplier`` is not a finite number ``> 0`` (i.e. "
    "``<= 0``, ``NaN``, or ``±inf``).",
    args_prose={
        "window": "Number of observations in the ATR moving window (canonically ``10``). Must be ``>= 1``.",
        "multiplier": "Band half-width as a multiple of the ATR (canonically ``3.0``). Must be a finite number "
        "``> 0`` (a non-positive multiplier would collapse or invert the bands).",
    },
    examples=(
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0),
                "low": (9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0),
                "close": (9.5, 10.8, 11.8, 10.2, 12.8, 11.2, 13.8),
            },
            params={"window": 2, "multiplier": 2.0},
            round_to=4,
            fields=("line", "direction"),
        ),
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 21.0, 22.0, 21.0, 23.0),
                "low": (9.0, 10.0, 11.0, 10.0, 12.0, 19.0, 20.0, 21.0, 20.0, 22.0),
                "close": (9.5, 10.8, 11.8, 10.2, 12.8, 19.5, 20.8, 21.8, 20.2, 22.8),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker's ratchet warms up "
            "independently:",
            partition=("AAPL",) * 5 + ("NVDA",) * 5,
            params={"window": 2, "multiplier": 2.0},
            round_to=4,
            fields=("line",),
        ),
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0),
                "low": (9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0),
                "close": (9.5, 10.8, 11.8, None, 12.8, float("nan"), 13.8),
            },
            intro="A ``null`` in ``close`` is skipped and bridged by the running state, while a ``NaN`` "
            "poisons the ATR recursion and latches ``NaN`` thereafter:",
            params={"window": 2, "multiplier": 2.0},
            round_to=4,
            fields=("line",),
        ),
        Example(
            inputs={"high": (10.0,), "low": (8.0,), "close": (9.0,)},
            intro="**window == 1** — a single-bar window makes the ATR the bar's own true range, and a "
            "close above the lower band seeds the trend long, so the line reads the lower band:",
            params={"window": 1},
            fields=("line",),
        ),
    ),
)
