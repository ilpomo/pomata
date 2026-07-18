"""
Declaration for ``pomata.indicators.balance_of_power`` — the bar close-open ratio, elementwise, propagating,
invariant.
"""

import math

from pomata.indicators import balance_of_power
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_balance_of_power
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

BALANCE_OF_POWER = suite_indicators(
    factory=balance_of_power,
    inputs=("open", "high", "low", "close"),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.NONE,
    oracle=reference_balance_of_power,
    scaling=(ScaleAxis(roles=("open", "high", "low", "close"), degree=0),),
    talib=RelationTalib.MATCHES,
    golden=Golden(
        inputs={
            "open": (10.0, 10.0, 10.0),
            "high": (12.0, 12.0, 12.0),
            "low": (8.0, 8.0, 8.0),
            "close": (11.0, 10.0, 9.0),
        },
        output=(0.25, 0.0, -0.25),
    ),
    pins=(
        Pin(
            label="null_close_propagates",
            inputs={"open": (10.0, 10.0), "high": (12.0, 12.0), "low": (8.0, 8.0), "close": (11.0, None)},
            expected=(0.25, None),
            reason="a null in one input yields null for that row on a non-flat bar; the shared flow rung nulls "
            "every role at once and cannot isolate one",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"open": (10.0, None), "high": (12.0, math.nan), "low": (8.0, 8.0), "close": (11.0, 11.0)},
            expected=(0.25, None),
            reason="a row carrying both a null (open) and a NaN (high) yields null — null wins",
        ),
        Pin(
            label="nan_propagates",
            inputs={"open": (10.0, 10.0), "high": (12.0, math.nan), "low": (8.0, 8.0), "close": (11.0, 11.0)},
            expected=(0.25, math.nan),
            reason="a NaN in one input propagates to NaN for that row",
        ),
        Pin(
            label="flat_bar_is_zero",
            inputs={"open": (10.0, 12.0), "high": (11.0, 12.0), "low": (9.0, 12.0), "close": (10.5, 11.0)},
            expected=(0.25, 0.0),
            reason="a flat bar (high == low, exact zero range) yields 0 by convention, over the bare 0/0",
        ),
    ),
)
