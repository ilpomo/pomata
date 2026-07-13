"""Spec for ``pomata.indicators.vwap`` — the cumulative volume-weighted average price, gap-bridging, degree-1."""

import math

from tests.indicators.oracles import vwap_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import vwap

VWAP = Spec(
    factory=vwap,
    inputs=("high", "low", "close", "volume"),
    params={},
    shape=Shape.SERIES,
    warmup=None,
    oracle=vwap_reference,
    # The running volume-weighted typical price: homogeneous of degree 1 in the price legs and invariant to a common
    # rescaling of volume, degree 0.
    scale=(
        ScaleAxis(roles=("high", "low", "close"), degree=1),
        ScaleAxis(roles=("volume",), degree=0),
    ),
    golden_input={
        "high": (2.0, 4.0, 6.0),
        "low": (0.0, 2.0, 4.0),
        "close": (1.0, 3.0, 5.0),
        "volume": (10.0, 20.0, 30.0),
    },
    golden_output=(1.0, 2.3333, 3.6667),
    pins=(
        SpecPin(
            label="zero_leading_volume_is_nan_then_recovers",
            inputs={
                "high": (10.0, 11.0, 12.0),
                "low": (8.0, 9.0, 10.0),
                "close": (9.0, 10.0, 11.0),
                "volume": (0.0, 100.0, 100.0),
            },
            expected=(math.nan, 10.0, 10.5),
            reason="a zero cumulative volume at the first bar is the 0/0 degenerate (NaN); once volume accrues the "
            "running average recovers",
        ),
    ),
)
