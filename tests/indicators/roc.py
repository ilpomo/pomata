"""Spec for ``pomata.indicators.roc`` — the windowed rate of change in percent, propagating, scale-invariant."""

import math

from tests.indicators.oracles import roc_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import roc

ROC = Spec(
    factory=roc,
    inputs=("expr",),
    params={"window": 3},
    shape=Shape.SERIES,
    warmup=3,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=roc_reference,
    # A ratio minus one, scale-INVARIANT, degree 0.
    scale=(ScaleAxis(roles=("expr",), degree=0),),
    golden_params={"window": 2},
    golden_input={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)},
    golden_output=(None, None, 200.0, 100.0, 66.6667),
    pins=(
        SpecPin(
            label="single_row_window_one",
            inputs={"expr": (42.0,)},
            params_override={"window": 1},
            expected=(None,),
            reason="a one-element series with window=1 stays undefined",
        ),
        SpecPin(
            label="window_one_is_single_period_return",
            inputs={"expr": (2.0, 4.0, 6.0)},
            params_override={"window": 1},
            expected=(None, 100.0, 50.0),
            reason="the one-period simple return in percent, the documented window=1 case",
        ),
        SpecPin(
            label="all_nan",
            inputs={"expr": (math.nan, math.nan, math.nan)},
            params_override={"window": 1},
            expected=(None, math.nan, math.nan),
            reason="an all-NaN series: warm-up stays null, then NaN propagates once both endpoints exist",
        ),
        SpecPin(
            label="constant_series_is_zero",
            inputs={"expr": (5.0, 5.0, 5.0, 5.0)},
            params_override={"window": 1},
            expected=(None, 0.0, 0.0, 0.0),
            reason="ROC of a constant non-zero series is exactly 0 once warmed up",
        ),
        SpecPin(
            label="zero_lagged_nonzero_change_is_signed_inf",
            inputs={"expr": (0.0, 5.0, 0.0, -5.0)},
            params_override={"window": 1},
            expected=(None, math.inf, -100.0, -math.inf),
            reason="a non-zero change over a zero lagged value is +/-inf, sign tracking the change direction",
        ),
        SpecPin(
            label="zero_lagged_mixed_change",
            inputs={"expr": (0.0, 0.0, 5.0)},
            params_override={"window": 1},
            expected=(None, math.nan, math.inf),
            reason="a zero change over zero is NaN (0/0), a non-zero change over zero is +inf",
        ),
    ),
)
