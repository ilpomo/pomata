"""
Declaration for ``pomata.indicators.aroon_oscillator`` — the aroon up-minus-down line, window-nulling, scale-
invariant.
"""

import polars as pl

from pomata.indicators import aroon, aroon_oscillator
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_aroon_oscillator
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape


def _aroon_oscillator_component() -> pl.Expr:
    """aroon_oscillator recomposed as the aroon up line minus its down line, at the canonical window=5."""
    bands = aroon(pl.col("high"), pl.col("low"), 5)
    return bands.struct.field("up") - bands.struct.field("down")


AROON_OSCILLATOR = suite_indicators(
    factory=aroon_oscillator,
    inputs=("high", "low"),
    params={"window": 5},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW,
    oracle=reference_aroon_oscillator,
    scaling=(ScaleAxis(roles=("high", "low"), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0),
            "low": (9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0),
        },
        output=(None, None, None, 66.6667, 33.3333, 33.3333, 100.0, 33.3333),
        params={"window": 3},
    ),
    recomposition=_aroon_oscillator_component,
    pins=(
        Pin(
            label="window_one_boundary",
            inputs={
                "high": (10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0),
                "low": (9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0),
            },
            params_override={"window": 1},
            expected=(None, 100.0, 100.0, -100.0, 100.0, -100.0, 100.0, -100.0, 100.0, -100.0),
            reason="window=1 collapses the Aroon lookback to the last two bars, so a strictly alternating series "
            "saturates the oscillator at plus or minus 100 from the second row on",
        ),
    ),
    reference='Chande, T. S. (1995). "The Time Price Oscillator." *Technical Analysis of Stocks & '
    "Commodities*, 13(9), 369-374.",
    reference_url="https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/aroon",
    see_also=(
        ("aroon", "The two-line indicator this collapses into one."),
        ("donchian_channels", "The rolling high/low extremes the lines are built from."),
        ("williams_r", "Another windowed high-low range oscillator."),
    ),
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "window == 1",
            "the look-back collapses to the last two bars, so each Aroon line is ``0`` or ``100`` and "
            "the oscillator takes only ``-100``, ``0``, or ``+100``.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The oscillator for each row, the same length as the inputs, in ``[-100, 100]``. The "
    "first ``window`` rows are ``null`` (warm-up), inherited from :func:`aroon`.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Look-back length; the extremes are sought over the last ``window + 1`` bars. Must be ``>= 1``.",
    },
    intro_basic="Basic usage on high-low bars:",
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
    intro_missing="A ``null`` (which nulls the oscillator) and a ``NaN`` (which propagates) in ``high`` "
    "make the handling visible:",
)
