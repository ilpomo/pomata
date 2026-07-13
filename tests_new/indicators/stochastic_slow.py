"""Spec for ``pomata.indicators.stochastic_slow`` — the slowed stochastic struct (k, d), window-nulling, invariant."""

import math

from tests_new.indicators.oracles import stochastic_slow_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import stochastic_slow

STOCHASTIC_SLOW = Spec(
    factory=stochastic_slow,
    inputs=("high", "low", "close"),
    params={"window_k": 14, "window_slowing": 3, "window_d": 3},
    shape=Shape.STRUCT,
    fields=("k", "d"),
    warmup={"k": 15, "d": 17},
    lands_on="close",
    raises=(
        ({"window_k": 0}, r"window_k must be >= 1"),
        ({"window_slowing": 0}, r"window_slowing must be >= 1"),
        ({"window_d": 0}, r"window_d must be >= 1"),
    ),
    oracle=stochastic_slow_reference,
    # Both lines are bounded ratios of price ranges, scale-INVARIANT, degree 0 (tests/indicators/test_stochastic_slow.py
    # ::test_scale_invariance).
    scale=(ScaleAxis(roles=("high", "low", "close"), degree={"k": 0, "d": 0}),),
    golden_params={"window_k": 5, "window_slowing": 3, "window_d": 3},
    golden_input={
        "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5),
        "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
        "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0),
    },
    golden_output={
        "k": (None, None, None, None, None, None, 79.9603, 74.6032, 80.9524, 76.1905),
        "d": (None, None, None, None, None, None, None, None, 78.5053, 77.2487),
    },
    pins=(
        SpecPin(
            label="flat_range_is_nan",
            inputs={"high": (10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0), "close": (10.0, 10.0, 10.0)},
            params_override={"window_k": 2, "window_slowing": 1, "window_d": 1},
            expected={"k": (None, math.nan, math.nan), "d": (None, math.nan, math.nan)},
            reason="a flat look-back makes the raw %K's 0/0 division NaN, passed through by the slowing and %D SMAs "
            "(test_stochastic_slow.py::TestStochasticSlowEdge::test_flat_window_is_nan)",
        ),
    ),
)
