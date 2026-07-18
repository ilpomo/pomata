"""
Declaration for ``pomata.indicators.hma`` — Hull's lag-reduced weighted mean, window-nulling, degree-1 homogeneous.
"""

from pomata.indicators import hma
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_hma
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

HMA = suite_indicators(
    factory=hma,
    inputs=("expr",),
    params={"window": 4},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW,
    oracle=reference_hma,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no Hull moving average.",
    raises=(
        ({"window": 1}, r"window must be >= 2"),
        ({"window": 0}, r"window must be >= 2"),
    ),
    golden=Golden(
        inputs={"expr": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)},
        output=(None, None, None, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
    ),
    pins=(
        Pin(
            label="golden_master_overshoot",
            inputs={"expr": (1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0)},
            expected=(None, None, None, None, 11.599999999999998, 11.5, 10.299999999999999, 10.0),
            reason="the lag correction 2*WMA(x,half) - WMA(x,window) over- and under-shoots the input range before the "
            "final smoothing settles",
        ),
        Pin(
            label="golden_master_round_half_up",
            inputs={"expr": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)},
            expected=(
                None,
                None,
                None,
                None,
                None,
                5.666666666666666,
                6.666666666666665,
                7.666666666666665,
                8.666666666666664,
                9.666666666666664,
                10.666666666666664,
                11.666666666666664,
            ),
            params_override={"window": 5},
            reason="the round-half-up period reduction at window=5: half-period = floor(5/2 + 0.5) = 3, not the "
            "banker-rounded 2",
        ),
    ),
)
