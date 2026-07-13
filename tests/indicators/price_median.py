"""Spec for ``pomata.indicators.price_median`` — the high-low midpoint, elementwise, propagating, degree-1."""

import math

from tests.indicators.oracles import price_median_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import price_median

PRICE_MEDIAN = Spec(
    factory=price_median,
    inputs=("high", "low"),
    params={},
    shape=Shape.SERIES,
    warmup=None,
    oracle=price_median_reference,
    # The midpoint (high + low) / 2 scales linearly with both inputs together, degree 1
    #
    scale=(ScaleAxis(roles=("high", "low"), degree=1),),
    golden_input={"high": (11.0, 12.0, 13.0, 12.5, 14.0), "low": (9.0, 10.0, 11.0, 11.0, 12.0)},
    golden_output=(10.0, 11.0, 12.0, 11.75, 13.0),
    pins=(
        SpecPin(
            label="null_precedence_null_high_nan_low",
            inputs={"high": (11.0, None), "low": (9.0, math.nan)},
            expected=(10.0, None),
            reason="a null in high against a NaN in low on the same row yields null — null wins over NaN ",
        ),
    ),
)
