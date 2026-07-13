"""Spec for ``pomata.indicators.midprice`` — the rolling high/low midprice of a bar series, window-nulling, degree-1."""

from tests.indicators.oracles import midprice_reference
from tests.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import midprice

MIDPRICE = Spec(
    factory=midprice,
    inputs=("high", "low"),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=13,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=midprice_reference,
    # The mean of the window's highest high and lowest low, homogeneous of degree 1
    scale=(ScaleAxis(roles=("high", "low"), degree=1),),
    golden_params={"window": 3},
    golden_input={
        "high": (11.0, 12.0, 13.0, 12.5, 14.0),
        "low": (9.0, 10.0, 11.0, 11.0, 12.0),
    },
    golden_output=(None, None, 11.0, 11.5, 12.5),
)
