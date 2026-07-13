"""Spec for ``pomata.indicators.chaikin_money_flow`` — the windowed money-flow ratio, window-nulling, invariant."""

from tests.indicators.oracles import chaikin_money_flow_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec

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
)
