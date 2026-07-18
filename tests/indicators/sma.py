"""Declaration for ``pomata.indicators.sma`` — the simple rolling mean, window-nulling, degree-1 homogeneous."""

from pomata.indicators import sma
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_sma
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

SMA = suite_indicators(
    factory=sma,
    inputs=("expr",),
    params={"window": 3},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_sma,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0)}, output=(None, None, 4.0, 6.0, 8.0)),
    pins=(
        Pin(
            label="single_row_window_one_identity",
            inputs={"expr": (42.0,)},
            expected=(42.0,),
            params_override={"window": 1},
            reason="a one-element series with window=1 returns the value itself",
        ),
        Pin(
            label="single_row_window_exceeds_length",
            inputs={"expr": (42.0,)},
            expected=(None,),
            reason="a one-element series with window=3 is entirely warm-up",
        ),
        Pin(
            label="window_equals_length",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(None, None, 2.0),
            reason="a window equal to the series length emits exactly one defined value, the whole-series mean",
        ),
        Pin(
            label="window_one_is_identity",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(1.0, 2.0, 3.0),
            params_override={"window": 1},
            reason="window=1 reproduces the input with no warm-up",
        ),
    ),
)
