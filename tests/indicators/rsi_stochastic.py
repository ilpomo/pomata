"""Declaration for ``pomata.indicators.rsi_stochastic`` — the stochastic of RSI struct (k, d), gap-bridging."""

import math

from pomata.indicators import rsi_stochastic
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_rsi_stochastic
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

RSI_STOCHASTIC = suite_indicators(
    factory=rsi_stochastic,
    inputs=("wave",),
    params={"window_rsi": 14, "window_k": 14, "window_d": 3},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.STRUCT,
    fields=("k", "d"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"k": 27, "d": 29},
    oracle=reference_rsi_stochastic,
    scaling=(ScaleAxis(roles=("wave",), degree={"k": 0, "d": 0}),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window_rsi": 0}, r"window_rsi must be >= 1"),
        ({"window_k": 0}, r"window_k must be >= 1"),
        ({"window_d": 0}, r"window_d must be >= 1"),
    ),
    golden=Golden(
        inputs={"wave": (50.0, 51.0, 50.5, 52.0, 51.5, 53.0, 52.0, 54.0, 53.5, 55.0)},
        output={
            "k": (None, None, None, None, None, 94.7368, 0.0, 81.5861, 44.2237, 100.0),
            "d": (None, None, None, None, None, None, 47.3684, 40.793, 62.9049, 72.1118),
        },
        params={"window_rsi": 3, "window_k": 3, "window_d": 2},
    ),
    pins=(
        Pin(
            label="flat_rsi_window_is_nan",
            inputs={"wave": (10.0, 11.0, 12.0, 13.0, 14.0)},
            params_override={"window_rsi": 2, "window_k": 2, "window_d": 1},
            expected={
                "k": (None, None, None, math.nan, math.nan),
                "d": (None, None, None, math.nan, math.nan),
            },
            reason="a monotone run gives an exactly-flat RSI, so the %K channel normalization is the 0/0 degenerate "
            "NaN",
        ),
    ),
    reference="Chande, T. S. & Kroll, S. (1994). *The New Technical Trader*. Wiley.",
    see_also=(
        ("rsi", "The oscillator this is the stochastic of."),
        ("stochastic_fast", "The same %K / %D construction applied to price."),
        ("stochastic_slow", "The smoothed %K / %D stochastic variant."),
    ),
    notes=(
        (
            "Composition",
            "Built from :func:`rsi` (whose recursive Wilder seeding it inherits — see that function's "
            "``Seeding`` note), then the %K range ratio, then the :func:`sma` of %K, so every stage's "
            "warm-up and ``null`` / ``NaN`` handling stacks.",
        ),
    ),
    note_extension="\n\n"
    "Both lines lie in ``[0, 100]``. Because the underlying :func:`rsi` is already "
    "scale-invariant, so is this; there is no homogeneity to test.",
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
            "when the RSI does not move over the look-back (its highest equals its lowest, e.g. a "
            "sustained trend pinning the RSI) the ``%K`` normalization is indeterminate, so the "
            "result is a ``0 / 0``, i.e. ``NaN``.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A struct ``pl.Expr`` with two ``Float64`` fields, the same length as ``expr``:"
    "\n\n"
    "- ``k`` — the raw %K line, ``100 * (rsi - RSImin) / (RSImax - RSImin)``. - ``d`` — the "
    "%D signal line, the :func:`sma` of %K over ``window_d``."
    "\n\n"
    'Read one line with ``.struct.field("k")`` (etc.) or split both into columns with '
    "``.struct.unnest()``. The warm-up stacks the :func:`rsi` warm-up (``window_rsi`` rows), "
    "the ``window_k - 1`` range look-back, and the ``window_d - 1`` of %D.",
    raises_prose="ValueError: If ``window_rsi < 1``, ``window_k < 1``, or ``window_d < 1``.",
    args_prose={
        "window_rsi": "Number of observations in the underlying :func:`rsi` (canonically ``14``). Must be ``>= 1``.",
        "window_k": "Number of observations in the %K look-back range over the RSI (canonically ``14``). Must "
        "be ``>= 1``.",
        "window_d": "Number of observations in the %D moving average of %K (canonically ``3``). Must be ``>= 1``.",
    },
    intro_basic="Basic usage on a single price series:",
    example_columns={"wave": "close"},
    examples=(
        Example(
            inputs={"wave": (50.0, 51.0, 50.5, 52.0, 51.5, 53.0, 52.0, 54.0, 53.5, 55.0)},
            params={"window_rsi": 3, "window_k": 3, "window_d": 2},
            round_to=4,
            fields=("k", "d"),
        ),
        Example(
            inputs={
                "wave": (50.0, 51.0, 50.5, 52.0, 51.5, 53.0, 52.0, 54.0, 40.0, 41.0, 40.5, 42.0, 41.5, 43.0, 42.0, 44.0)
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("AAPL",) * 8 + ("NVDA",) * 8,
            params={"window_rsi": 3, "window_k": 3, "window_d": 2},
            round_to=4,
            fields=("k",),
        ),
        Example(
            inputs={
                "wave": (50.0, 51.0, 50.5, 52.0, 51.5, 53.0, 52.0, 54.0, None, 55.0, float("nan"), 56.0, 57.0, 58.0)
            },
            intro="A ``null`` (which nulls the dependent %K) and a ``NaN`` (which propagates) make the "
            "handling visible:",
            params={"window_rsi": 3, "window_k": 3, "window_d": 2},
            round_to=4,
            fields=("k",),
        ),
        Example(
            inputs={"wave": (10.0, 11.0, 12.0, 13.0, 14.0)},
            intro="**Degenerate denominator** — a monotone run gives an exactly-flat RSI, so the %K channel "
            "normalization is the ``0/0`` degenerate ``NaN``:",
            params={"window_rsi": 2, "window_k": 2, "window_d": 1},
            fields=("k",),
        ),
    ),
)
