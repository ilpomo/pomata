"""Spec for ``pomata.indicators.adx`` — Wilder's average directional index, gap-bridging, scale-invariant."""

import math

from tests.indicators.oracles import adx_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import adx

ADX = Spec(
    factory=adx,
    inputs=("high", "low", "close"),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=26,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=adx_reference,
    # A Wilder rma of the directional movement index, bounded in [0, 100] and scale-INVARIANT, degree 0
    # (tests/indicators/test_adx.py::TestAdxProperties::test_scale_invariance).
    scale=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    golden_params={"window": 2},
    golden_input={
        "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5),
        "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
        "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0),
    },
    golden_output=(None, None, 100.0, 60.0, 68.2353, 44.1176, 58.3602, 39.1801, 55.4486, 37.7243),
    pins=(
        SpecPin(
            label="flat_window_is_nan",
            inputs={"high": (10.0,) * 8, "low": (10.0,) * 8, "close": (10.0,) * 8},
            params_override={"window": 3},
            expected=(None, None, None, None, math.nan, math.nan, math.nan, math.nan),
            reason="a fully flat window makes the underlying dx the indeterminate 0/0 (both directional indicators are "
            "zero), which then poisons the Wilder smoothing recursion (tests/indicators/test_adx.py"
            "::TestAdxEdge::test_flat_window_is_nan)",
        ),
    ),
)
