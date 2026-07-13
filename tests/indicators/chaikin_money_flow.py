"""Spec for ``pomata.indicators.chaikin_money_flow`` — the windowed money-flow ratio, window-nulling, invariant."""

import math

from tests.indicators.oracles import chaikin_money_flow_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import chaikin_money_flow

CHAIKIN_MONEY_FLOW = Spec(
    factory=chaikin_money_flow,
    inputs=("high", "low", "close", "volume"),
    params={"window": 20},
    shape=Shape.SERIES,
    warmup=19,
    lands_on="close",
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=chaikin_money_flow_reference,
    # The rolling money-flow volume over the rolling volume, a bounded ratio that is scale-INVARIANT in the price legs
    # and in volume independently, degree 0 (tests/indicators/test_chaikin_money_flow.py::test_price_scale_invariance /
    # test_volume_scale_invariance).
    scale=(
        ScaleAxis(roles=("high", "low", "close"), degree=0),
        ScaleAxis(roles=("volume",), degree=0),
    ),
    golden_params={"window": 3},
    golden_input={
        "high": (10.0, 12.0, 11.0, 13.0, 14.0),
        "low": (8.0, 9.0, 9.0, 10.0, 11.0),
        "close": (9.0, 11.0, 10.0, 12.0, 13.0),
        "volume": (100.0, 200.0, 150.0, 300.0, 250.0),
    },
    golden_output=(None, None, 0.1481, 0.2564, 0.2619),
    pins=(
        SpecPin(
            label="zero_total_volume_is_nan",
            inputs={
                "high": (10.0, 12.0, 11.0),
                "low": (8.0, 9.0, 9.0),
                "close": (9.0, 11.0, 10.0),
                "volume": (0.0, 0.0, 0.0),
            },
            params_override={"window": 2},
            expected=(None, math.nan, math.nan),
            reason="a window whose total volume is zero divides by zero, the IEEE-754 0/0 == NaN (tests/indicators/"
            "test_chaikin_money_flow.py::TestChaikinMoneyFlowEdge::test_zero_total_volume_is_nan)",
        ),
        SpecPin(
            label="zero_volume_after_large_volume_is_nan",
            inputs={
                "high": (12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0),
                "low": (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0),
                "close": (11.0, 12.5, 13.5, 14.0, 15.5, 16.0, 17.5),
                "volume": (1e16, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0),
            },
            params_override={"window": 3},
            expected=(None, None, 0.0, 0.25, 0.2, 0.0, math.nan),
            reason="an all-zero-volume window still yields NaN after a large 1e16 volume has slid out: the rolling sum "
            "retains a sub-ULP residual on exit, but the exact all-zero detection pins the final window to NaN "
            "(tests/indicators/test_chaikin_money_flow.py::TestChaikinMoneyFlowEdge"
            "::test_zero_volume_after_large_volume_is_nan)",
        ),
    ),
)
