"""Spec for ``pomata.indicators.price_weighted_close`` — the close-weighted HLC mean, elementwise, degree-1."""

import math

from tests.indicators.oracles import price_weighted_close_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import price_weighted_close

PRICE_WEIGHTED_CLOSE = Spec(
    factory=price_weighted_close,
    inputs=("high", "low", "close"),
    params={},
    shape=Shape.SERIES,
    warmup=None,
    oracle=price_weighted_close_reference,
    # (high + low + 2*close) / 4 scales linearly with the three legs together, degree 1
    # (tests/indicators/test_price_weighted_close.py::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("high", "low", "close"), degree=1),),
    golden_input={
        "high": (11.0, 12.0, 13.0, 12.5, 14.0),
        "low": (9.0, 10.0, 11.0, 11.0, 12.0),
        "close": (10.0, 11.5, 12.5, 11.5, 13.5),
    },
    golden_output=(10.0, 11.25, 12.25, 11.625, 13.25),
    pins=(
        SpecPin(
            label="null_propagates",
            inputs={"high": (11.0, None, 13.0), "low": (9.0, 10.0, 11.0), "close": (10.0, 11.5, 12.5)},
            expected=(10.0, None, 12.25),
            reason="a null in exactly one input role nulls that row only "
            "(test_price_weighted_close.py::test_null_propagates)",
        ),
        SpecPin(
            label="nan_propagates",
            inputs={"high": (11.0, math.nan, 13.0), "low": (9.0, 10.0, 11.0), "close": (10.0, 11.5, 12.5)},
            expected=(10.0, math.nan, 12.25),
            reason="a NaN in exactly one input role nans that row only "
            "(test_price_weighted_close.py::test_nan_propagates)",
        ),
        SpecPin(
            label="null_takes_precedence_over_nan",
            inputs={"high": (11.0, None), "low": (9.0, math.nan), "close": (10.0, 11.5)},
            expected=(10.0, None),
            reason="a row carrying both a null (high) and a NaN (low) yields null — null wins over NaN "
            "(test_price_weighted_close.py::test_null_takes_precedence_over_nan)",
        ),
    ),
)
