"""Declaration for ``pomata.indicators.fisher_transform`` — the Gaussianized channel struct (fisher, signal)."""

import math

from pomata.indicators import fisher_transform
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_fisher_transform
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

FISHER_TRANSFORM = suite_indicators(
    factory=fisher_transform,
    inputs=("high", "low"),
    params={"window": 10},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.STRUCT,
    fields=("fisher", "signal"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"fisher": 9, "signal": 10},
    oracle=reference_fisher_transform,
    scaling=(ScaleAxis(roles=("high", "low"), degree={"fisher": 0, "signal": 0}),),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no Fisher Transform.",
    raises=(({"window": 0}, r"window must be >= 1"),),
    flow_horizon=60,
    golden=Golden(
        inputs={"high": (2.0, 4.0, 3.0), "low": (0.0, 2.0, 1.0)},
        output={
            "fisher": (None, 0.3428, 0.0621),
            "signal": (None, None, 0.3428),
        },
        params={"window": 2},
    ),
    pins=(
        Pin(
            label="window_one_single_row_is_flat_nan",
            inputs={"high": (11.0,), "low": (9.0,)},
            params_override={"window": 1},
            expected={"fisher": (math.nan,), "signal": (None,)},
            reason="window=1 is flat by construction (max == min), so fisher is NaN from the first row while signal "
            "is still warm-up null",
        ),
        Pin(
            label="flat_window_is_nan",
            inputs={"high": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0)},
            params_override={"window": 3},
            expected={
                "fisher": (None, None, math.nan, math.nan, math.nan, math.nan),
                "signal": (None, None, None, math.nan, math.nan, math.nan),
            },
            reason="a constant series has max == min over every window: the channel normalization is 0/0 NaN, which "
            "bridges through the recursion",
        ),
    ),
)
