"""
Declaration for ``pomata.indicators.t3`` — Tillson's six-EMA smoother, gap-bridging, NaN-latching, degree-1
homogeneous.
"""

import math

from pomata.indicators import t3
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_t3
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

T3 = suite_indicators(
    factory=t3,
    inputs=("expr",),
    params={"window": 3, "volume_factor": 0.7},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=12,
    oracle=reference_t3,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"volume_factor": math.nan}, r"volume_factor must be a finite number"),
        ({"volume_factor": math.inf}, r"volume_factor must be a finite number"),
        ({"volume_factor": -math.inf}, r"volume_factor must be a finite number"),
    ),
    golden=Golden(
        inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)},
        output=(None, None, None, None, None, None, 13.1, 15.1, 17.1, 19.1),
        params={"window": 2},
    ),
    pins=(
        Pin(
            label="window_one_is_identity",
            inputs={"expr": (1.0, 2.0, 3.0)},
            expected=(1.0, 2.0, 3.0),
            params_override={"window": 1},
            reason="window=1 makes every EMA the identity and the coefficients sum to 1",
        ),
        Pin(
            label="single_row_window_one",
            inputs={"expr": (42.0,)},
            expected=(42.0,),
            params_override={"window": 1},
            reason="a one-row series with window=1 is the identity",
        ),
        Pin(
            label="single_row_window_two",
            inputs={"expr": (42.0,)},
            expected=(None,),
            params_override={"window": 2},
            reason="a one-row series with window=2 cannot complete any EMA pass",
        ),
        Pin(
            label="window_exceeds_series_length",
            inputs={"expr": (1.0, 2.0)},
            expected=(None, None),
            params_override={"window": 2},
            reason="a series shorter than the window warm-up yields an entirely null output",
        ),
        Pin(
            label="all_zero_series_is_zero",
            inputs={"expr": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
            expected=(None, None, None, None, None, None, 0.0, 0.0),
            params_override={"window": 2},
            reason="the degenerate all-zero window stays exactly at zero",
        ),
        Pin(
            label="null_bridged",
            inputs={"expr": (1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)},
            expected=(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                7.565358126301888,
                8.557239987129336,
                9.552943668252915,
            ),
            params_override={"window": 2},
            reason="an early null extends the warm-up; the value resumes once all six EMA passes re-seed",
        ),
        Pin(
            label="nan_latches",
            inputs={"expr": (1.0, math.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)},
            expected=(None, None, None, None, None, None, math.nan, math.nan, math.nan, math.nan),
            params_override={"window": 2},
            reason="a NaN poisons the recursion and latches as exactly NaN for every subsequent value",
        ),
        Pin(
            label="interior_null_bridged",
            inputs={"expr": (2.0, 4.0, None, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)},
            expected=(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                15.14916985651142,
                17.11616757027457,
                19.104042047403723,
            ),
            params_override={"window": 2},
            reason="an interior null nulls its position while the recursion bridges the gap",
        ),
        Pin(
            label="constant_series",
            inputs={"expr": (5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0)},
            expected=(None, None, None, None, None, None, 5.0, 5.0, 5.0, 5.0),
            params_override={"window": 2},
            reason="T3 of a constant equals that constant once warmed up",
        ),
        Pin(
            label="golden_master_adjusted",
            inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)},
            expected=(
                None,
                None,
                None,
                None,
                None,
                None,
                13.065715164239307,
                15.090278304462984,
                17.0981497650465,
                19.10020550466045,
            ),
            params_override={"window": 2, "adjust": True},
            reason="the frozen adjusted-mode golden master",
        ),
    ),
)
