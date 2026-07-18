"""
Declaration for ``pomata.indicators.vortex`` — the plus/minus vortex movement pair, window-nulling, scale-invariant.
"""

import math

from pomata.indicators import vortex
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_vortex
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

VORTEX = suite_indicators(
    factory=vortex,
    inputs=("high", "low", "close"),
    params={"window": 14},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.STRUCT,
    fields=("plus", "minus"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"plus": 14, "minus": 14},
    oracle=reference_vortex,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree={"plus": 0, "minus": 0}),),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no Vortex indicator.",
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (2.0, 4.0, 6.0, 5.0, 7.0),
            "low": (1.0, 3.0, 4.0, 4.0, 5.0),
            "close": (1.5, 3.5, 5.0, 4.5, 6.0),
        },
        output={
            "plus": (None, None, 1.2, 1.1429, 1.1429),
            "minus": (None, None, 0.2, 0.5714, 0.5714),
        },
        params={"window": 2},
    ),
    pins=(
        Pin(
            label="window_one_single_bar_ratio",
            inputs={"high": (2.0, 4.0, 6.0), "low": (1.0, 3.0, 4.0), "close": (1.5, 3.5, 5.0)},
            params_override={"window": 1},
            expected={"plus": (None, 1.2, 1.2), "minus": (None, 0.4, 0.0)},
            reason="window=1 reduces each line to a single bar's vortex movement over its true range, the first bar "
            "warm-up (no prior bar)",
        ),
        Pin(
            label="flat_window_is_nan",
            inputs={"high": (10.0,) * 6, "low": (10.0,) * 6, "close": (10.0,) * 6},
            params_override={"window": 2},
            expected={
                "plus": (None, None, math.nan, math.nan, math.nan, math.nan),
                "minus": (None, None, math.nan, math.nan, math.nan, math.nan),
            },
            reason="a flat window has zero summed true range and zero summed movement, so both lines are the "
            "indeterminate 0/0 == NaN after warm-up",
        ),
    ),
)
