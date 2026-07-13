"""Spec for ``pomata.metrics.total_return_rolling`` — the growth over a trailing window, scale-invariant."""

import math

from tests.metrics.oracles import total_return_rolling_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import total_return_rolling

TOTAL_RETURN_ROLLING = Spec(
    factory=total_return_rolling,
    inputs=("equity_curve",),
    params={"window": 3},
    shape=Shape.SERIES,
    warmup=2,
    raises=(({"window": 1}, r"window must be >= 2"),),
    oracle=total_return_rolling_reference,
    # The window's endpoint ratio E_t / E_{t-window+1} is unchanged by a uniform rescale (verified numerically; the
    # old suite carries no scale test).
    scale=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden_input={"equity_curve": (1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25)},
    golden_output=(None, None, 0.05, 0.0909, 0.0952, 0.0833, 0.087),
    pins=(
        SpecPin(
            label="window_equals_length",
            inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2, 1.15)},
            expected=(None, None, None, None, 0.1499999999999999),
            reason="when the window equals the series length only the last row is defined "
            "(tests/metrics/test_total_return_rolling.py::test_window_equals_length)",
            params_override={"window": 5},
        ),
        SpecPin(
            label="endpoint_null_is_null",
            inputs={"equity_curve": (1.0, 1.1, None)},
            expected=(None, None, None),
            reason="a null at a window endpoint yields null: the result depends on both endpoints "
            "(tests/metrics/test_total_return_rolling.py::test_endpoint_null_is_null)",
        ),
        SpecPin(
            label="interior_null_is_spanned",
            inputs={"equity_curve": (1.0, None, 1.2)},
            expected=(None, None, 0.19999999999999996),
            reason="an interior null has zero effect on a fully-defined window; only the two endpoints determine it "
            "(tests/metrics/test_total_return_rolling.py::test_interior_null_is_spanned)",
        ),
        SpecPin(
            label="endpoint_nan_propagates",
            inputs={"equity_curve": (1.0, 1.1, math.nan)},
            expected=(None, None, math.nan),
            reason="a NaN at a window endpoint propagates to NaN "
            "(tests/metrics/test_total_return_rolling.py::test_endpoint_nan_propagates)",
        ),
        SpecPin(
            label="matches_reference_representative_curve",
            inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)},
            expected=(
                None,
                None,
                None,
                0.19999999999999996,
                0.04545454545454519,
                0.23809523809523814,
                0.04166666666666674,
                0.21739130434782616,
            ),
            reason="a representative equity curve compared against the naive reference at the correctness tier "
            "(tests/metrics/test_total_return_rolling.py::TestTotalReturnRollingCorrectness::test_matches_reference)",
            params_override={"window": 4},
        ),
    ),
)
