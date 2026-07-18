"""
Declaration for ``pomata.indicators.absolute_price_oscillator`` — the EMA-difference oscillator, gap-bridging,
degree-1.
"""

from pomata.indicators import absolute_price_oscillator
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_absolute_price_oscillator
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape
from tests.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE

ABSOLUTE_PRICE_OSCILLATOR = suite_indicators(
    factory=absolute_price_oscillator,
    inputs=("expr",),
    params={"window_fast": 2, "window_slow": 3},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=2,
    oracle=reference_absolute_price_oscillator,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window_fast": 0}, r"window_fast must be >= 1"),
        ({"window_slow": 0}, r"window_slow must be >= 1"),
        ({"window_fast": 5, "window_slow": 3}, r"windows must be ordered window_fast <= window_slow"),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    golden=Golden(
        inputs={"expr": (10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0)},
        output=(None, None, 0.5, 0.1667, 0.3889, 0.463, 0.1543, 0.3848),
    ),
    pins=(
        Pin(
            label="equal_windows_are_zero",
            inputs={"expr": (10.0, 11.0, 12.0)},
            expected=(None, 0.0, 0.0),
            params_override={"window_fast": 2, "window_slow": 2},
            reason="equal fast/slow windows make the two EMAs identical so the oscillator cancels to exactly 0.0",
        ),
    ),
)
