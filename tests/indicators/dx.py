"""Declaration for ``pomata.indicators.dx`` — Wilder's directional movement index, gap-bridging, scale-invariant."""

import math

from pomata.indicators import dx
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_dx
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

DX = suite_indicators(
    factory=dx,
    inputs=("high", "low", "close"),
    params={"window": 14},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_dx,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0),
            "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0),
            "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5),
        },
        output=(None, 100.0, 100.0, 20.0, 76.4706, 20.0, 72.6027),
        params={"window": 2},
    ),
    pins=(
        Pin(
            label="flat_window_is_nan",
            inputs={"high": (10.0,) * 8, "low": (10.0,) * 8, "close": (10.0,) * 8},
            params_override={"window": 3},
            expected=(None, None, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan),
            reason="a fully flat window has no movement either way, so both directional indicators are NaN and the "
            "indeterminate 0/0 spread propagates",
        ),
    ),
    reference="Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.",
    wikipedia="https://en.wikipedia.org/wiki/Average_directional_movement_index",
    see_also=(
        ("di_plus", "The plus directional indicator."),
        ("di_minus", "The minus directional indicator."),
        ("adx", "The Wilder-smoothed average of this."),
    ),
    notes=(
        ("Seeding", "The warm-up inherits the recursive Wilder seeding of :func:`rma` used throughout the cluster."),
    ),
    note_extension="\n\nIt is scale-invariant under a positive common rescaling of ``high``, ``low``, and ``close``.",
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap.",
        ),
        (
            "NaN",
            "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent "
            "non-null position — except at ``window == 1``, where the smoothing is the identity and "
            "nothing latches: the ``NaN`` clears once it leaves the one-bar reach of the differencing "
            "and the true range.",
        ),
        (
            "Degenerate denominator",
            "``+DI`` and ``-DI`` are both zero (no movement either way), so the result is a ``0 / 0``, i.e. ``NaN``.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The directional index for each row, the same length as the inputs, in ``[0, 100]``. The "
    "first ``window - 1`` values are ``null`` (warm-up), inherited from the directional "
    "indicators.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the Wilder moving window. Must be ``>= 1``.",
    },
    intro_basic="On a small OHLC frame with a short window:",
    examples=(
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5),
                "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0),
            },
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 11.5, 13.0, 20.0, 22.0, 19.0, 23.0, 20.0),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 18.0, 20.0, 17.0, 21.0, 18.0),
                "close": (9.5, 10.5, 11.5, 11.0, 12.5, 19.0, 21.0, 18.0, 22.0, 19.0),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("A", "A", "A", "A", "A", "B", "B", "B", "B", "B"),
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5),
                "close": (None, 10.5, 11.5, 11.0, float("nan"), 12.0, 13.5, 13.0),
            },
            intro="A leading ``null`` ``close`` (absorbed by the underlying ATR's true-range maximum) and a "
            "later ``NaN`` (which propagates through the directional indicators) make the handling "
            "visible:",
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={
                "high": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
                "low": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
                "close": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
            },
            intro="**Degenerate denominator** — a fully flat window has no movement either way, so both "
            "directional indicators are ``NaN`` and the indeterminate ``0/0`` spread propagates:",
            params={"window": 3},
        ),
    ),
)
