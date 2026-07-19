"""
Declaration for ``pomata.indicators.stochastic_slow`` — the slowed stochastic struct (k, d), window-nulling,
invariant.
"""

import math

from pomata.indicators import stochastic_slow
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_stochastic_slow
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

STOCHASTIC_SLOW = suite_indicators(
    factory=stochastic_slow,
    inputs=("high", "low", "close"),
    params={"window_k": 14, "window_slowing": 3, "window_d": 3},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.STRUCT,
    fields=("k", "d"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"k": 15, "d": 17},
    oracle=reference_stochastic_slow,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree={"k": 0, "d": 0}),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window_k": 0}, r"window_k must be >= 1"),
        ({"window_slowing": 0}, r"window_slowing must be >= 1"),
        ({"window_d": 0}, r"window_d must be >= 1"),
    ),
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5),
            "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
            "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0),
        },
        output={
            "k": (None, None, None, None, None, None, 79.9603, 74.6032, 80.9524, 76.1905),
            "d": (None, None, None, None, None, None, None, None, 78.5053, 77.2487),
        },
        params={"window_k": 5, "window_slowing": 3, "window_d": 3},
    ),
    pins=(
        Pin(
            label="flat_range_is_nan",
            inputs={"high": (10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0), "close": (10.0, 10.0, 10.0)},
            params_override={"window_k": 2, "window_slowing": 1, "window_d": 1},
            expected={"k": (None, math.nan, math.nan), "d": (None, math.nan, math.nan)},
            reason="a flat look-back makes the raw %K's 0/0 division NaN, passed through by the slowing and %D SMAs",
        ),
        Pin(
            label="flat_range_close_off_is_inf",
            inputs={"high": (10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0), "close": (20.0, 20.0, 20.0)},
            params_override={"window_k": 2, "window_slowing": 1, "window_d": 1},
            expected={"k": (None, math.inf, math.inf), "d": (None, math.inf, math.inf)},
            reason="a malformed bar whose close sits above a flat high==low look-back makes the raw %K a non-zero "
            "over zero, so %K is +inf, passed through by the slowing and %D SMAs — the infinity beside the 0/0 NaN pin",
        ),
    ),
    reference='Lane, G. C. (1984). "Lane\'s Stochastics." *Technical Analysis of Stocks & Commodities*, 2(3), 87-90.',
    wikipedia="https://en.wikipedia.org/wiki/Stochastic_oscillator",
    see_also=(
        ("stochastic_fast", "The unsmoothed variant, whose raw %K this smooths."),
        ("rsi_stochastic", "The stochastic applied to :func:`rsi` instead of price."),
        ("sma", "The moving average behind both the slowing and %D."),
    ),
    notes=(
        (
            "Composition",
            "The slow %K is the :func:`sma` of the raw %K, and %D is the :func:`sma` of the slow %K, "
            "so each averaging inherits the warm-up and ``null`` / ``NaN`` handling on top of the raw "
            "%K's own.",
        ),
    ),
    note_extension="\n\n"
    "Both lines are scale-invariant under a positive common rescaling of ``high``, ``low``, "
    "and ``close``, and lie in ``[0, 100]`` for well-formed bars (``low <= close <= high``). "
    "The slow %K equals the fast %D of :func:`stochastic_fast` when ``window_slowing`` "
    "matches that call's ``window_d``.",
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window_k`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Degenerate denominator",
            "the highest ``high`` equals the lowest ``low`` over the look-back and the close sits on "
            "that flat level, so the result is a ``0 / 0``, i.e. ``NaN`` — off that level it is "
            "``+/-inf`` instead, and either value then propagates through the slowing and %D "
            "averages.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A struct ``pl.Expr`` with two ``Float64`` fields, the same length as the inputs:"
    "\n\n"
    "- ``k`` — the slow %K line, the :func:`sma` of the raw %K over ``window_slowing``. - "
    "``d`` — the %D signal line, the :func:`sma` of the slow %K over ``window_d``."
    "\n\n"
    'Read one line with ``.struct.field("k")`` (etc.) or split both into columns with '
    "``.struct.unnest()``. The first ``window_k + window_slowing - 2`` rows are ``null`` on "
    "``k``, and a further ``window_d - 1`` on ``d``.",
    raises_prose="ValueError: If ``window_k < 1``, ``window_slowing < 1``, or ``window_d < 1``.",
    args_prose={
        "window_k": "Number of observations in the raw %K look-back range (canonically ``14``). Must be ``>= 1``.",
        "window_slowing": "Number of observations in the slowing average that turns the raw %K into the slow %K "
        "(canonically ``3``). Must be ``>= 1``.",
        "window_d": "Number of observations in the %D moving average of the slow %K (canonically ``3``). Must "
        "be ``>= 1``.",
    },
    examples=(
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5),
                "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0),
            },
            params={"window_k": 3, "window_slowing": 2, "window_d": 2},
            round_to=4,
            fields=("k", "d"),
        ),
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 11.5, 13.0, 20.0, 21.0, 22.0, 21.5, 23.0),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 19.0, 20.0, 21.0, 20.5, 22.0),
                "close": (9.5, 10.5, 11.5, 11.0, 12.5, 19.5, 20.5, 21.5, 21.0, 22.5),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("AAPL",) * 5 + ("NVDA",) * 5,
            params={"window_k": 2, "window_slowing": 2, "window_d": 2},
            round_to=4,
            fields=("k",),
        ),
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5),
                "close": (9.5, 10.5, 11.5, None, 12.5, float("nan"), 13.5, 13.0),
            },
            intro="A ``null`` (nulls every slow %K window it falls in) and a ``NaN`` (which propagates the "
            "same way) in ``close`` surface on the slow %K line:",
            params={"window_k": 2, "window_slowing": 2, "window_d": 2},
            round_to=4,
            fields=("k",),
        ),
        Example(
            inputs={"high": (10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0), "close": (10.0, 10.0, 10.0)},
            intro="**Degenerate denominator** — a flat look-back makes the raw %K's ``0/0`` division "
            "``NaN``, passed through by the slowing and %D SMAs:",
            params={"window_k": 2, "window_slowing": 1, "window_d": 1},
            fields=("k",),
        ),
        Example(
            inputs={"high": (10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0), "close": (20.0, 20.0, 20.0)},
            intro="**Degenerate denominator** — a malformed bar whose close sits above a flat ``high == "
            "low`` look-back makes the raw %K a nonzero over zero, so %K is ``+inf``, passed through "
            "by the slowing and %D SMAs:",
            params={"window_k": 2, "window_slowing": 1, "window_d": 1},
            fields=("k",),
        ),
    ),
)
