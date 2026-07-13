"""Spec for ``pomata.indicators.price_typical`` — the HLC mean, elementwise, propagating, degree-1 homogeneous."""

import math

from tests_new.indicators.oracles import price_typical_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import price_typical

PRICE_TYPICAL = Spec(
    factory=price_typical,
    inputs=("high", "low", "close"),
    params={},
    shape=Shape.SERIES,
    warmup=None,
    oracle=price_typical_reference,
    # The mean of high/low/close scales linearly with them together, degree 1 (tests/indicators/test_price_typical.py
    # ::TestPriceTypicalProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("high", "low", "close"), degree=1),),
    golden_input={
        "high": (11.0, 12.0, 13.0, 12.5, 14.0),
        "low": (9.0, 10.0, 11.0, 11.0, 12.0),
        "close": (10.0, 11.5, 12.5, 11.5, 13.5),
    },
    golden_output=(10.0, 11.1667, 12.1667, 11.6667, 13.1667),
    pins=(
        SpecPin(
            label="null_precedence_null_high_nan_low",
            inputs={"high": (11.0, None), "low": (9.0, math.nan), "close": (10.0, 11.5)},
            expected=(10.0, None),
            reason="a null in high combined with a NaN in low on the same row yields null — null wins over NaN "
            "(test_price_typical.py::TestPriceTypicalEdge::test_null_takes_precedence_over_nan)",
        ),
    ),
)
