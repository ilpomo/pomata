"""Declaration for ``pomata.indicators.trix`` — the triple-EMA rate of change, gap-bridging, NaN-latching, degree-0."""

import math

from pomata.indicators import trix
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_trix
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape
from tests.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE

TRIX = suite_indicators(
    factory=trix,
    inputs=("price",),
    params={"window": 2},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=4,
    oracle=reference_trix,
    scaling=(ScaleAxis(roles=("price",), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    golden=Golden(
        inputs={"price": (10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0)},
        output=(None, None, None, None, 5.4718, 7.4466, 2.989, 5.4253),
    ),
    pins=(
        Pin(
            label="window_one_identity_ema",
            inputs={"price": (100.0, 120.0, 90.0, 108.0)},
            expected=(None, 20.0, -25.0, 20.0),
            params_override={"window": 1},
            reason="window=1 makes every EMA pass the identity, degenerating TRIX to the one-period percentage ROC of "
            "the raw input",
        ),
        Pin(
            label="flat_zero_history_is_degenerate",
            inputs={"price": (0.0, 0.0, 0.0, 0.0, 0.0, 5.0)},
            expected=(None, None, None, None, math.nan, math.inf),
            reason="a zero-valued history drives the triple EMA to exactly zero, so the rate of change divides by that "
            "zero: a 0/0 while the EMA holds at zero, +/-inf the moment it moves off it",
        ),
    ),
)
