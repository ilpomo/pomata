"""Spec for ``pomata.indicators.balance_of_power`` — the bar close-open ratio, elementwise, propagating, invariant."""

import math

from tests.indicators.oracles import balance_of_power_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import balance_of_power

BALANCE_OF_POWER = Spec(
    factory=balance_of_power,
    inputs=("open", "high", "low", "close"),
    params={},
    shape=Shape.SERIES,
    warmup=None,
    oracle=balance_of_power_reference,
    # A bounded ratio in [-1, 1], scale-INVARIANT, degree 0 (tests/indicators/test_balance_of_power.py
    # ::test_scale_invariance).
    scale=(ScaleAxis(roles=("open", "high", "low", "close"), degree=0),),
    lands_on="close",
    golden_input={
        "open": (10.0, 10.0, 10.0),
        "high": (12.0, 12.0, 12.0),
        "low": (8.0, 8.0, 8.0),
        "close": (11.0, 10.0, 9.0),
    },
    golden_output=(0.25, 0.0, -0.25),
    pins=(
        SpecPin(
            label="null_close_propagates",
            inputs={"open": (10.0, 10.0), "high": (12.0, 12.0), "low": (8.0, 8.0), "close": (11.0, None)},
            expected=(0.25, None),
            reason="a null in one input yields null for that row on a non-flat bar (test_balance_of_power.py"
            "::test_null_propagates); the shared flow rung nulls every role at once and cannot isolate one",
        ),
        SpecPin(
            label="null_takes_precedence_over_nan",
            inputs={"open": (10.0, None), "high": (12.0, math.nan), "low": (8.0, 8.0), "close": (11.0, 11.0)},
            expected=(0.25, None),
            reason="a row carrying both a null (open) and a NaN (high) yields null — null wins "
            "(test_balance_of_power.py::test_null_takes_precedence_over_nan)",
        ),
        SpecPin(
            label="nan_propagates",
            inputs={"open": (10.0, 10.0), "high": (12.0, math.nan), "low": (8.0, 8.0), "close": (11.0, 11.0)},
            expected=(0.25, math.nan),
            reason="a NaN in one input propagates to NaN for that row (test_balance_of_power.py::test_nan_propagates)",
        ),
        SpecPin(
            label="flat_bar_is_zero",
            inputs={"open": (10.0, 12.0), "high": (11.0, 12.0), "low": (9.0, 12.0), "close": (10.5, 11.0)},
            expected=(0.25, 0.0),
            reason="a flat bar (high == low, exact zero range) yields 0 by convention, over the bare 0/0 "
            "(test_balance_of_power.py::test_flat_bar_is_zero)",
        ),
    ),
)
