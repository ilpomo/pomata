"""
Declaration for ``pomata.indicators.di_plus`` — Wilder's positive directional indicator, gap-bridging, scale-
invariant.
"""

import math

from pomata.indicators import di_plus
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_di_plus
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

DI_PLUS = suite_indicators(
    factory=di_plus,
    inputs=("high", "low", "close"),
    params={"window": 14},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_di_plus,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0),
            "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0),
            "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5),
        },
        output=(None, 40.0, 54.5455, 31.5789, 58.8235, 36.1446, 59.7156),
        params={"window": 2},
    ),
    pins=(
        Pin(
            label="flat_window_is_nan",
            inputs={"high": (10.0,) * 8, "low": (10.0,) * 8, "close": (10.0,) * 8},
            params_override={"window": 3},
            expected=(None, None, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan),
            reason="a fully flat window makes the average true range zero, so the smoothed movement over it is the "
            "indeterminate 0/0",
        ),
    ),
    reference="Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.",
    wikipedia="https://en.wikipedia.org/wiki/Average_directional_movement_index",
    see_also=(
        ("di_minus", "The minus counterpart."),
        ("dm_plus", "The smoothed plus directional movement in the numerator."),
        ("dx", "The directional index built from the two indicators."),
    ),
    notes=(
        ("Seeding", "The warm-up inherits the recursive Wilder seeding of :func:`rma` used throughout the cluster."),
    ),
    note_extension="\n\n"
    "It is scale-invariant under a positive common rescaling of ``high``, ``low``, and "
    "``close`` (the smoothed movement and the average true range scale together).",
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap — "
            "a ``null`` prior close drops the close-based true-range terms and shrinks the ATR, so on "
            "a gap the ratio can exceed the nominal ``[0, 100]`` bound.",
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
            "for ``window >= 2`` the whole series so far must be flat (both the ATR and the smoothed "
            "movement zero, since the ATR is an infinite-memory Wilder RMA), so the result is a ``0 / "
            "0``, i.e. ``NaN``; a merely local flat patch after earlier movement leaves the ATR "
            "small-but-positive and the DI finite, while at ``window == 1`` there is no memory, so a "
            "single bar with zero range and zero gap already triggers it.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The plus directional indicator for each row, the same length as the inputs, in ``[0, "
    "100]`` on complete bars. The first ``window - 1`` values are ``null`` (warm-up).",
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
            partition=("AAPL",) * 5 + ("NVDA",) * 5,
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5),
                "close": (None, 10.5, 11.5, 11.0, float("nan"), 12.0, 13.5, 13.0),
            },
            intro="A leading ``null`` ``close`` (absorbed by the ATR's true-range maximum) and a later "
            "``NaN`` (which latches) make the handling visible:",
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={
                "high": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
                "low": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
                "close": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
            },
            intro="**Degenerate denominator** — a fully flat window makes the average true range zero, so "
            "the smoothed movement over it is the indeterminate ``0/0`` and the result is ``NaN`` "
            "after warm-up:",
            params={"window": 3},
        ),
    ),
)
