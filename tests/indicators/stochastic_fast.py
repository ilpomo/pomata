"""Declaration for ``pomata.indicators.stochastic_fast`` — the fast stochastic struct (k, d), window-nulling."""

import math

from pomata.indicators import stochastic_fast
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_stochastic_fast
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

STOCHASTIC_FAST = suite_indicators(
    factory=stochastic_fast,
    inputs=("high", "low", "close"),
    params={"window_k": 14, "window_d": 3},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.STRUCT,
    fields=("k", "d"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"k": 13, "d": 15},
    oracle=reference_stochastic_fast,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree={"k": 0, "d": 0}),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window_k": 0}, r"window_k must be >= 1"),
        ({"window_d": 0}, r"window_d must be >= 1"),
    ),
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5),
            "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
            "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0),
        },
        output={
            "k": (None, None, None, None, 87.5, 66.6667, 85.7143, 71.4286, 85.7143, 71.4286),
            "d": (None, None, None, None, None, None, 79.9603, 74.6032, 80.9524, 76.1905),
        },
        params={"window_k": 5, "window_d": 3},
    ),
    pins=(
        Pin(
            label="flat_window_is_nan",
            inputs={"high": (10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0), "close": (10.0, 10.0, 10.0)},
            params_override={"window_k": 2, "window_d": 1},
            expected={"k": (None, math.nan, math.nan), "d": (None, math.nan, math.nan)},
            reason="a flat window makes the raw %K's 0/0 division NaN, which the %D pass carries through",
        ),
        Pin(
            label="flat_window_close_off_is_inf",
            inputs={"high": (10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0), "close": (20.0, 20.0, 20.0)},
            params_override={"window_k": 2, "window_d": 1},
            expected={"k": (None, math.inf, math.inf), "d": (None, math.inf, math.inf)},
            reason="a malformed bar whose close sits above a flat high==low look-back makes the raw %K a non-zero "
            "over zero, so %K is +inf, which the %D pass carries through — the infinity beside the 0/0 NaN pin",
        ),
    ),
    reference='Lane, G. C. (1984). "Lane\'s Stochastics." *Technical Analysis of Stocks & Commodities*, 2(3), 87-90.',
    wikipedia="https://en.wikipedia.org/wiki/Stochastic_oscillator",
    see_also=(
        ("stochastic_slow", "The slow variant, %K smoothed once more before %D."),
        ("rsi_stochastic", "The same oscillator applied to :func:`rsi` instead of price."),
        ("sma", "The moving average that forms %D."),
    ),
    notes=(
        (
            "Composition",
            "%D is the :func:`sma` of %K, so it inherits that warm-up and the ``null`` / ``NaN`` "
            "handling on top of %K's own.",
        ),
    ),
    note_extension="\n\n"
    "Both lines are scale-invariant under a positive common rescaling of ``high``, ``low``, "
    "and ``close`` (a ratio of price ranges), and lie in ``[0, 100]`` for well-formed bars "
    "(``low <= close <= high``).",
    bullets=(
        (
            "Null",
            "a window containing a ``null`` yields ``null`` (the window must hold ``window_k`` "
            "non-null values) — including a ``null`` in the current ``close``, which %K reads "
            "outright rather than through a window.",
        ),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Degenerate denominator",
            "the highest ``high`` equals the lowest ``low`` over the look-back and the close sits on "
            "that flat level, so the result is a ``0 / 0``, i.e. ``NaN`` — off that level it is "
            "``+/-inf`` instead (a malformed bar whose close sits outside its own high-low range; "
            "unreachable when ``low <= close <= high``).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A struct ``pl.Expr`` with two ``Float64`` fields, the same length as the inputs:"
    "\n\n"
    "- ``k`` — the raw %K line, ``100 * (close - LL) / (HH - LL)``. - ``d`` — the %D signal "
    "line, the :func:`sma` of %K over ``window_d``."
    "\n\n"
    'Read one line with ``.struct.field("k")`` (etc.) or split both into columns with '
    "``.struct.unnest()``. The first ``window_k - 1`` rows are ``null`` on ``k`` (the "
    "look-back warm-up), and a further ``window_d - 1`` on ``d``.",
    raises_prose="ValueError: If ``window_k < 1`` or ``window_d < 1``.",
    args_prose={
        "window_k": "Number of observations in the %K look-back range (canonically ``14``). Must be ``>= 1``.",
        "window_d": "Number of observations in the %D moving average of %K (canonically ``3``). Must be ``>= 1``.",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
    intro_missing="A ``null`` (yields ``null`` on ``k`` at that row) and a ``NaN`` (which propagates) in "
    "``close`` surface on the %K line:",
)
