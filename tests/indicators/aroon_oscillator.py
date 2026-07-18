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
)
