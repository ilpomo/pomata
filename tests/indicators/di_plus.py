"""Spec for ``pomata.indicators.di_plus`` — Wilder's positive directional indicator, gap-bridging, scale-invariant."""

import math

from tests.indicators.oracles import di_plus_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import di_plus

DI_PLUS = Spec(
    factory=di_plus,
    inputs=("high", "low", "close"),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=13,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=di_plus_reference,
    # A percentage ratio of smoothed movements to the average true range, bounded in [0, 100] and scale-INVARIANT,
    # degree 0
    scale=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    golden_params={"window": 2},
    golden_input={
        "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0),
        "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0),
        "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5),
    },
    golden_output=(None, 40.0, 54.5455, 31.5789, 58.8235, 36.1446, 59.7156),
    pins=(
        SpecPin(
            label="flat_window_is_nan",
            inputs={"high": (10.0,) * 8, "low": (10.0,) * 8, "close": (10.0,) * 8},
            params_override={"window": 3},
            expected=(None, None, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan),
            reason="a fully flat window makes the average true range zero, so the smoothed movement over it is the "
            "indeterminate 0/0",
        ),
    ),
)
