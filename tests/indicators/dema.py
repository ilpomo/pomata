"""Declaration for ``pomata.indicators.dema`` — the double EMA lag-correction, gap-bridging, NaN-latching, degree-1."""

import math

from pomata.indicators import dema
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_dema
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

DEMA = suite_indicators(
    factory=dema,
    inputs=("expr",),
    params={"window": 3},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=4,
    oracle=reference_dema,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)},
        output=(None, None, 6.0, 8.0, 10.0, 12.0),
        params={"window": 2},
    ),
    pins=(
        Pin(
            label="single_row_window_one",
            inputs={"expr": (42.0,)},
            expected=(42.0,),
            params_override={"window": 1},
            reason="window=1 on one row is the identity",
        ),
        Pin(
            label="single_row_window_two",
            inputs={"expr": (42.0,)},
            expected=(None,),
            params_override={"window": 2},
            reason="a single row shorter than the warm-up yields null",
        ),
        Pin(
            label="null_bridged",
            inputs={"expr": (1.0, None, 3.0, 4.0)},
            expected=(None, None, None, 3.9999999999999996),
            params_override={"window": 2},
            reason="an early interior null extends the warm-up and the recursion bridges it",
        ),
        Pin(
            label="nan_latches",
            inputs={"expr": (1.0, math.nan, 3.0, 4.0, 5.0)},
            expected=(None, None, math.nan, math.nan, math.nan),
            params_override={"window": 2},
            reason="a NaN poisons the recursive state and latches for every subsequent value",
        ),
        Pin(
            label="window_one_is_identity",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(1.0, 2.0, 3.0),
            params_override={"window": 1},
            reason="window=1 degenerates each EMA pass to the identity",
        ),
        Pin(
            label="all_zero_series_is_zero",
            inputs={"expr": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
            expected=(None, None, None, None, 0.0, 0.0),
            params_override={"window": 3},
            reason="the degenerate all-zero recursion stays exactly at zero",
        ),
        Pin(
            label="window_fills_entire_series_with_warmup",
            inputs={"expr": (1.0, 2.0)},
            expected=(None, None),
            params_override={"window": 2},
            reason="a series no longer than the warm-up emits nothing",
        ),
        Pin(
            label="interior_null_bridged",
            inputs={"expr": (2.0, 4.0, None, 8.0, 10.0, 12.0)},
            expected=(None, None, None, 9.428571428571427, 10.412698412698411, 12.116402116402115),
            params_override={"window": 2},
            reason="an interior null nulls its own row while the recursion bridges the gap",
        ),
        Pin(
            label="golden_master_adjusted",
            inputs={"expr": (3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0)},
            expected=(
                None,
                None,
                None,
                None,
                4.042089093701996,
                7.846915855948113,
                3.832696745373009,
                5.383328263901351,
            ),
            params_override={"adjust": True},
            reason="the adjust=True golden branch at the canonical window",
        ),
        Pin(
            label="constant_series",
            inputs={"expr": (5.0, 5.0, 5.0, 5.0, 5.0, 5.0)},
            expected=(None, None, None, None, 5.0, 5.0),
            params_override={"window": 3},
            reason="DEMA of a constant equals that constant once warmed up",
        ),
    ),
)
