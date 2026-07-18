"""Declaration for ``pomata.indicators.percentage_price_oscillator`` — the normalized EMA-difference, gap-bridging."""

import math

from pomata.indicators import percentage_price_oscillator
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_percentage_price_oscillator
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape
from tests_new.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE

PERCENTAGE_PRICE_OSCILLATOR = suite_indicators(
    factory=percentage_price_oscillator,
    inputs=("price",),
    params={"window_fast": 12, "window_slow": 26},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=25,
    oracle=reference_percentage_price_oscillator,
    scaling=(ScaleAxis(roles=("price",), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window_fast": 0}, r"window_fast must be >= 1"),
        ({"window_slow": 0}, r"window_slow must be >= 1"),
        ({"window_fast": 5, "window_slow": 3}, r"windows must be ordered window_fast <= window_slow"),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    golden=Golden(
        inputs={"price": (10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0)},
        output=(None, None, 4.5455, 1.5152, 3.2407, 3.5613, 1.1871, 2.7484),
        params={"window_fast": 2, "window_slow": 3},
    ),
    pins=(
        Pin(
            label="equal_windows_are_zero",
            inputs={"price": (10.0, 11.0, 12.0)},
            params_override={"window_fast": 2, "window_slow": 2},
            expected=(None, 0.0, 0.0),
            reason="equal fast/slow windows produce identical EMAs so the oscillator cancels to exactly 0 ",
        ),
        Pin(
            label="zero_slow_ema_is_nan",
            inputs={"price": (0.0, 0.0, 0.0, 0.0)},
            params_override={"window_fast": 2, "window_slow": 3},
            expected=(None, None, math.nan, math.nan),
            reason="an all-zero series drives both EMAs to exactly 0.0, so the 0/0 boundary surfaces as NaN ",
        ),
        Pin(
            label="nonzero_gap_zero_slow_ema_is_inf",
            inputs={"price": (1.0, 1.0, -2.0)},
            params_override={"window_fast": 2, "window_slow": 3},
            expected=(None, None, -math.inf),
            reason="a window summing to zero seeds the slow EMA at exactly 0.0 while the fast EMA stays non-zero, so "
            "the non-zero gap over the zero slow EMA is +/-inf — the infinity beside the 0/0 NaN pin",
        ),
    ),
)
