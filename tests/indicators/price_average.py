"""Spec for ``pomata.indicators.price_average`` — the OHLC mean, elementwise, propagating, degree-1 homogeneous."""

import math

from tests.indicators.oracles import price_average_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import price_average

PRICE_AVERAGE = Spec(
    factory=price_average,
    inputs=("open", "high", "low", "close"),
    params={},
    shape=Shape.SERIES,
    warmup=None,
    oracle=price_average_reference,
    # The mean of four price legs scales linearly with them together, degree 1 (tests/indicators/test_price_average.py
    # ::TestPriceAverageProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("open", "high", "low", "close"), degree=1),),
    golden_input={
        "open": (10.0, 11.0, 12.0, 11.5, 13.0),
        "high": (11.0, 12.0, 13.0, 12.5, 14.0),
        "low": (9.0, 10.0, 11.0, 11.0, 12.0),
        "close": (10.0, 11.5, 12.5, 11.5, 13.5),
    },
    golden_output=(10.0, 11.125, 12.125, 11.625, 13.125),
    pins=(
        SpecPin(
            label="null_takes_precedence_over_nan",
            inputs={"open": (10.0, None), "high": (11.0, math.nan), "low": (9.0, 10.0), "close": (10.0, 11.5)},
            expected=(10.0, None),
            reason="a null in open and a NaN in high on the same row yields null — null wins over NaN "
            "(test_price_average.py::TestPriceAverageEdge::test_null_takes_precedence_over_nan)",
        ),
    ),
)
