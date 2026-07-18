"""
Declaration for ``pomata.indicators.midpoint`` — the rolling high/low midpoint of a series, window-nulling, degree-1.
"""

from pomata.indicators import midpoint
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_midpoint
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

MIDPOINT = suite_indicators(
    factory=midpoint,
    inputs=("expr",),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_midpoint,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={"expr": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)}, output=(None, None, 2.0, 3.0, 4.0, 5.0), params={"window": 3}
    ),
    pins=(
        Pin(
            label="window_one_is_identity",
            inputs={"expr": (1.0, 2.0, 3.0)},
            params_override={"window": 1},
            expected=(1.0, 2.0, 3.0),
            reason="window=1 makes the max and min the value itself, so the midpoint reproduces the input with no "
            "warm-up",
        ),
    ),
)
