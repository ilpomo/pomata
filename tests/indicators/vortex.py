"""
Declaration for ``pomata.indicators.vortex`` — the plus/minus vortex movement pair, window-nulling, scale-invariant.
"""

import math

from pomata.indicators import vortex
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_vortex
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

VORTEX = suite_indicators(
    factory=vortex,
    inputs=("high", "low", "close"),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.STRUCT,
    fields=("plus", "minus"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"plus": 14, "minus": 14},
    oracle=reference_vortex,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree={"plus": 0, "minus": 0}),),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no Vortex indicator.",
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (2.0, 4.0, 6.0, 5.0, 7.0),
            "low": (1.0, 3.0, 4.0, 4.0, 5.0),
            "close": (1.5, 3.5, 5.0, 4.5, 6.0),
        },
        output={
            "plus": (None, None, 1.2, 1.1429, 1.1429),
            "minus": (None, None, 0.2, 0.5714, 0.5714),
        },
        params={"window": 2},
    ),
    pins=(
        Pin(
            label="window_one_single_bar_ratio",
            inputs={"high": (2.0, 4.0, 6.0), "low": (1.0, 3.0, 4.0), "close": (1.5, 3.5, 5.0)},
            params_override={"window": 1},
            expected={"plus": (None, 1.2, 1.2), "minus": (None, 0.4, 0.0)},
            reason="window=1 reduces each line to a single bar's vortex movement over its true range, the first bar "
            "warm-up (no prior bar)",
        ),
        Pin(
            label="flat_window_is_nan",
            inputs={"high": (10.0,) * 6, "low": (10.0,) * 6, "close": (10.0,) * 6},
            params_override={"window": 2},
            expected={
                "plus": (None, None, math.nan, math.nan, math.nan, math.nan),
                "minus": (None, None, math.nan, math.nan, math.nan, math.nan),
            },
            reason="a flat window has zero summed true range and zero summed movement, so both lines are the "
            "indeterminate 0/0 == NaN after warm-up",
        ),
    ),
    reference='Botes, E. & Siepman, D. (2010). "The Vortex Indicator." *Technical Analysis of Stocks & '
    "Commodities*, 28(1), 20-30.",
    wikipedia="https://en.wikipedia.org/wiki/Vortex_indicator",
    see_also=(
        ("di_plus", "The Wilder directional indicator, the same movement-over-range idea, exponentially smoothed."),
        ("true_range", "The per-bar basis of the shared denominator."),
        ("di_minus", "The minus directional indicator, the Wilder analog of the negative vortex line."),
    ),
    notes=(
        ("Inputs", "``high`` / ``low`` / ``close`` must share a length and alignment (the same row index is one bar)."),
    ),
    bullets=(
        (
            "Null",
            "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null "
            "values) — including via the one-bar lag, which makes the first movement ``null``.",
        ),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Degenerate denominator",
            "a window has zero summed true range and zero summed movement, so the result is a ``0 / "
            "0``, i.e. ``NaN`` — detected per line via the residual-free rolling maxima of the true "
            "range and the movement; a near-flat window (tiny ranges after a much larger one has slid "
            "out) is not silenced, since ``VI+`` is unbounded above and the streaming quotient cannot "
            "be clipped to a range, degrading in precision past a sane dynamic range.",
        ),
        (
            "window == 1",
            "each line reduces to a single bar's vortex movement over its own true range; the first "
            "row is ``null`` (there is no prior bar for the lag).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A struct ``pl.Expr`` with two ``Float64`` fields, the same length as the inputs:"
    "\n\n"
    "- ``plus`` — the positive vortex line ``VI+``. - ``minus`` — the negative vortex line "
    "``VI-``."
    "\n\n"
    'Read one line with ``.struct.field("plus")`` (etc.) or split both into columns with '
    "``.struct.unnest()``. The first ``window`` rows are ``null`` (warm-up): each line needs "
    "``window`` defined vortex movements, and the first movement is ``null`` (it reads the "
    "previous bar).",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the moving window. Must be ``>= 1``.",
    },
    intro_basic="On a small OHLC frame, reading each vortex line with ``.struct.field``:",
    examples=(
        Example(
            inputs={
                "high": (2.0, 4.0, 6.0, 5.0, 7.0, 6.5, 8.0, 7.5),
                "low": (1.0, 3.0, 4.0, 4.0, 5.0, 5.5, 6.0, 6.5),
                "close": (1.5, 3.5, 5.0, 4.5, 6.0, 6.0, 7.0, 7.0),
            },
            params={"window": 2},
            round_to=4,
            fields=("plus", "minus"),
        ),
        Example(
            inputs={
                "high": (2.0, 4.0, 6.0, 5.0, 7.0, 12.0, 11.0, 13.0, 10.0, 12.0),
                "low": (1.0, 3.0, 4.0, 4.0, 5.0, 10.0, 9.0, 11.0, 8.0, 10.0),
                "close": (1.5, 3.5, 5.0, 4.5, 6.0, 11.0, 10.0, 12.0, 9.0, 11.0),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("A", "A", "A", "A", "A", "B", "B", "B", "B", "B"),
            params={"window": 2},
            round_to=4,
            fields=("plus", "minus"),
        ),
        Example(
            inputs={
                "high": (2.0, 4.0, 6.0, 5.0, 7.0, 6.5, 8.0, 7.5),
                "low": (1.0, 3.0, 4.0, 4.0, 5.0, 5.5, 6.0, 6.5),
                "close": (None, 3.5, 5.0, 4.5, float("nan"), 6.0, 7.0, 7.0),
            },
            intro="A leading ``null`` ``close`` (absorbed by the true-range maximum) and a later ``NaN`` "
            "(which contaminates only the bars whose window spans it, then clears) make the handling "
            "visible:",
            params={"window": 2},
            round_to=4,
            fields=("plus", "minus"),
        ),
        Example(
            inputs={
                "high": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
                "low": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
                "close": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
            },
            intro="**Degenerate denominator** — a flat window has zero summed true range and zero summed "
            "movement, so both lines are the indeterminate ``0/0``, i.e. ``NaN``, after warm-up:",
            params={"window": 2},
            fields=("plus",),
        ),
        Example(
            inputs={"high": (2.0, 4.0, 6.0), "low": (1.0, 3.0, 4.0), "close": (1.5, 3.5, 5.0)},
            intro="**window == 1** — each line reduces to a single bar's vortex movement over its own true "
            "range; the first row is ``null`` (no prior bar for the lag):",
            params={"window": 1},
            fields=("plus",),
        ),
    ),
)
