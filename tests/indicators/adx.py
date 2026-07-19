"""Declaration for ``pomata.indicators.adx`` — Wilder's average directional index, gap-bridging, scale-invariant."""

import math

from pomata.indicators import adx
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_adx
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

ADX = suite_indicators(
    factory=adx,
    inputs=("high", "low", "close"),
    params={"window": 14},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=26,
    oracle=reference_adx,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5),
            "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
            "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0),
        },
        output=(None, None, 100.0, 60.0, 68.2353, 44.1176, 58.3602, 39.1801, 55.4486, 37.7243),
        params={"window": 2},
    ),
    pins=(
        Pin(
            label="flat_window_is_nan",
            inputs={"high": (10.0,) * 8, "low": (10.0,) * 8, "close": (10.0,) * 8},
            params_override={"window": 3},
            expected=(None, None, None, None, math.nan, math.nan, math.nan, math.nan),
            reason="a fully flat window makes the underlying dx the indeterminate 0/0 (both directional indicators are "
            "zero), which then poisons the Wilder smoothing recursion",
        ),
    ),
    reference="Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.",
    wikipedia="https://en.wikipedia.org/wiki/Average_directional_movement_index",
    see_also=(
        ("dx", "The directional index this smooths."),
        ("adxr", "The ADX rating (this averaged with its own past)."),
        ("di_plus", "The plus directional indicator."),
    ),
    notes=(
        ("Seeding", "The warm-up inherits the recursive Wilder seeding of :func:`rma` used throughout the cluster."),
    ),
    note_extension="\n\n"
    "It is scale-invariant under a positive common rescaling of ``high``, ``low``, and "
    "``close`` (it is built from ratios of directional movement to the average true range).",
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap.",
        ),
        (
            "NaN",
            "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent "
            "non-null position — except at ``window == 1``, where every Wilder smoothing in the stack "
            "is the identity and nothing latches: the ``NaN`` clears once it leaves the inputs' "
            "finite reach.",
        ),
        (
            "Degenerate denominator",
            "``di+`` and ``di-`` are both zero, so the result is a ``0 / 0``, i.e. ``NaN`` — the "
            "underlying :func:`dx` is the immediate ``0 / 0``, which then poisons the ADX recursion.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The ADX for each row, the same length as the inputs, in ``[0, 100]``. It carries a deep "
    "warm-up — roughly ``2 * (window - 1)`` rows of ``null`` — since it smooths the "
    "already-smoothed :func:`dx`.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the Wilder moving window. Must be ``>= 1``.",
    },
    intro_basic="On a small OHLC frame with a short window:",
    examples=(
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
                "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0),
            },
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 20.0, 22.0, 19.0, 23.0, 20.0, 24.0),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 18.0, 20.0, 17.0, 21.0, 18.0, 22.0),
                "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 19.0, 21.0, 18.0, 22.0, 19.0, 23.0),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("AAPL",) * 6 + ("NVDA",) * 6,
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5),
                "close": (None, 10.5, 11.5, 11.0, float("nan"), 12.0, 13.5, 13.0),
            },
            intro="A leading ``null`` ``close`` (absorbed by the true-range maximum) and a later ``NaN`` "
            "(which poisons the recursion and latches) make the handling visible:",
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={
                "high": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
                "low": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
                "close": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
            },
            intro="**Degenerate denominator** — a fully flat window leaves both directional indicators at "
            "zero, the indeterminate ``0/0``, which poisons the Wilder smoothing recursion, so the "
            "result is ``NaN`` after warm-up:",
            params={"window": 3},
        ),
    ),
)
