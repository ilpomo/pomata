"""Spec for ``pomata.indicators.dx`` — Wilder's directional movement index, gap-bridging, scale-invariant."""

import math

from tests.indicators.oracles import dx_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import dx

DX = Spec(
    factory=dx,
    inputs=("high", "low", "close"),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=13,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=dx_reference,
    # The normalized spread of the two directional indicators, bounded in [0, 100] and scale-INVARIANT, degree 0
    #
    scale=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    golden_params={"window": 2},
    golden_input={
        "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0),
        "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0),
        "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5),
    },
    golden_output=(None, 100.0, 100.0, 20.0, 76.4706, 20.0, 72.6027),
    pins=(
        SpecPin(
            label="flat_window_is_nan",
            inputs={"high": (10.0,) * 8, "low": (10.0,) * 8, "close": (10.0,) * 8},
            params_override={"window": 3},
            expected=(None, None, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan),
            reason="a fully flat window has no movement either way, so both directional indicators are NaN and the "
            "indeterminate 0/0 spread propagates",
        ),
    ),
)
