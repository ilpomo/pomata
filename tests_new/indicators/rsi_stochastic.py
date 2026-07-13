"""Spec for ``pomata.indicators.rsi_stochastic`` — the stochastic of RSI struct (k, d), gap-bridging."""

import math

from tests_new.indicators.oracles import rsi_stochastic_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import rsi_stochastic

RSI_STOCHASTIC = Spec(
    factory=rsi_stochastic,
    inputs=("wave",),
    params={"window_rsi": 14, "window_k": 14, "window_d": 3},
    shape=Shape.STRUCT,
    fields=("k", "d"),
    warmup={"k": 27, "d": 29},
    raises=(
        ({"window_rsi": 0}, r"window_rsi must be >= 1"),
        ({"window_k": 0}, r"window_k must be >= 1"),
        ({"window_d": 0}, r"window_d must be >= 1"),
    ),
    oracle=rsi_stochastic_reference,
    # The underlying RSI is scale-invariant, so both lines inherit it verbatim, degree 0
    # (tests/indicators/test_rsi_stochastic.py::test_scale_invariance).
    scale=(ScaleAxis(roles=("wave",), degree={"k": 0, "d": 0}),),
    golden_params={"window_rsi": 3, "window_k": 3, "window_d": 2},
    golden_input={"wave": (50.0, 51.0, 50.5, 52.0, 51.5, 53.0, 52.0, 54.0, 53.5, 55.0)},
    golden_output={
        "k": (None, None, None, None, None, 94.7368, 0.0, 81.5861, 44.2237, 100.0),
        "d": (None, None, None, None, None, None, 47.3684, 40.793, 62.9049, 72.1118),
    },
    pins=(
        SpecPin(
            label="flat_rsi_window_is_nan",
            inputs={"wave": (10.0, 11.0, 12.0, 13.0, 14.0)},
            params_override={"window_rsi": 2, "window_k": 2, "window_d": 1},
            expected={
                "k": (None, None, None, math.nan, math.nan),
                "d": (None, None, None, math.nan, math.nan),
            },
            reason="a monotone run gives an exactly-flat RSI, so the %K channel normalization is the 0/0 degenerate "
            "NaN (test_rsi_stochastic.py::TestRsiStochasticEdge::test_flat_window_is_nan)",
        ),
    ),
)
