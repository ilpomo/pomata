"""Spec for ``pomata.indicators.aroon_oscillator`` — the aroon up-minus-down line, window-nulling, scale-invariant."""

import polars as pl
from tests.indicators.oracles import aroon_oscillator_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import aroon, aroon_oscillator


def _aroon_oscillator_component() -> pl.Expr:
    """aroon_oscillator recomposed as the aroon up line minus its down line, at the canonical window=5."""
    bands = aroon(pl.col("high"), pl.col("low"), 5)
    return bands.struct.field("up") - bands.struct.field("down")


AROON_OSCILLATOR = Spec(
    factory=aroon_oscillator,
    inputs=("high", "low"),
    params={"window": 5},
    shape=Shape.SERIES,
    warmup=5,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=aroon_oscillator_reference,
    # The difference of two position-based lines, scale-INVARIANT, degree 0.
    scale=(ScaleAxis(roles=("high", "low"), degree=0),),
    component_expr=_aroon_oscillator_component,
    golden_params={"window": 3},
    golden_input={
        "high": (10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0),
        "low": (9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0),
    },
    golden_output=(None, None, None, 66.6667, 33.3333, 33.3333, 100.0, 33.3333),
    pins=(
        SpecPin(
            label="window_one_boundary",
            inputs={
                "high": (10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0),
                "low": (9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0),
            },
            params_override={"window": 1},
            expected=(None, 100.0, 100.0, -100.0, 100.0, -100.0, 100.0, -100.0, 100.0, -100.0),
            reason="the window=1 degenerate branch, never reached by the fixed-window generic rungs",
        ),
    ),
)
