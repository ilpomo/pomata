"""Spec for ``pomata.indicators.stochastic_fast`` — the fast stochastic struct (k, d), window-nulling."""

import math

from tests_new.indicators.oracles import stochastic_fast_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import stochastic_fast

STOCHASTIC_FAST = Spec(
    factory=stochastic_fast,
    inputs=("high", "low", "close"),
    params={"window_k": 14, "window_d": 3},
    shape=Shape.STRUCT,
    fields=("k", "d"),
    warmup={"k": 13, "d": 15},
    lands_on="close",
    raises=(
        ({"window_k": 0}, r"window_k must be >= 1"),
        ({"window_d": 0}, r"window_d must be >= 1"),
    ),
    oracle=stochastic_fast_reference,
    # Both lines are bounded ratios of price ranges, scale-INVARIANT, degree 0 (tests/indicators/test_stochastic_fast.py
    # ::test_scale_invariance).
    scale=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    golden_params={"window_k": 5, "window_d": 3},
    golden_input={
        "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5),
        "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
        "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0),
    },
    golden_output={
        "k": (None, None, None, None, 87.5, 66.6667, 85.7143, 71.4286, 85.7143, 71.4286),
        "d": (None, None, None, None, None, None, 79.9603, 74.6032, 80.9524, 76.1905),
    },
    pins=(
        SpecPin(
            label="flat_window_is_nan",
            inputs={"high": (10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0), "close": (10.0, 10.0, 10.0)},
            params_override={"window_k": 2, "window_d": 1},
            expected={"k": (None, math.nan, math.nan), "d": (None, math.nan, math.nan)},
            reason="a flat window makes the raw %K's 0/0 division NaN, which the %D pass carries through "
            "(test_stochastic_fast.py::test_flat_window_is_nan)",
        ),
    ),
)
