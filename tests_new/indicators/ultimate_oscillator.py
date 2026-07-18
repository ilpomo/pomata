"""
Declaration for ``pomata.indicators.ultimate_oscillator`` — Williams' three-window oscillator, window-nulling,
invariant.
"""

import math

from pomata.indicators import ultimate_oscillator
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_ultimate_oscillator
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

_HLC_HIGH = (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5)

_HLC_LOW = (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5)

_HLC_CLOSE = (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0)

ULTIMATE_OSCILLATOR = suite_indicators(
    factory=ultimate_oscillator,
    inputs=("high", "low", "close"),
    params={"window_short": 7, "window_medium": 14, "window_long": 28},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=27,
    oracle=reference_ultimate_oscillator,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window_short": 0}, r"window_short must be >= 1"),
        ({"window_medium": 0}, r"window_medium must be >= 1"),
        ({"window_long": 0}, r"window_long must be >= 1"),
        (
            {"window_short": 14, "window_medium": 7},
            r"windows must be ordered window_short <= window_medium <= window_long",
        ),
        (
            {"window_medium": 28, "window_long": 14},
            r"windows must be ordered window_short <= window_medium <= window_long",
        ),
    ),
    golden=Golden(
        inputs={"high": _HLC_HIGH, "low": _HLC_LOW, "close": _HLC_CLOSE},
        output=(None, None, None, 60.7143, 66.6667, 65.0433, 67.619, 65.4762, 67.619, 65.4762),
        params={"window_short": 2, "window_medium": 3, "window_long": 4},
    ),
    pins=(
        Pin(
            label="window_all_one_equal_windows_accepted",
            inputs={"high": _HLC_HIGH, "low": _HLC_LOW, "close": _HLC_CLOSE},
            params_override={"window_short": 1, "window_medium": 1, "window_long": 1},
            expected=(
                50.0,
                66.66666666666667,
                66.66666666666667,
                50.0,
                75.0,
                50.0,
                75.0,
                50.0,
                75.0,
                50.0,
            ),
            reason="equal windows are accepted (not raised) and the minimum window=1 is fully defined from row 0",
        ),
        Pin(
            label="flat_window_is_nan",
            inputs={"high": (10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0), "close": (10.0, 10.0, 10.0)},
            params_override={"window_short": 1, "window_medium": 1, "window_long": 2},
            expected=(None, math.nan, math.nan),
            reason="the 0/0 degenerate on a flat well-formed series, detected via residual-free rolling maxima",
        ),
        Pin(
            label="flat_window_is_nan_at_large_magnitude",
            inputs={"high": (1e9, 1e9, 1e9), "low": (1e9, 1e9, 1e9), "close": (1e9, 1e9, 1e9)},
            params_override={"window_short": 1, "window_medium": 1, "window_long": 2},
            expected=(None, math.nan, math.nan),
            reason="the exact-flat guard is residual-free at scale, yielding NaN rather than a falsely-saturated value",
        ),
        Pin(
            label="flat_range_missing_low_is_inf",
            inputs={"high": (10.0, 8.0), "low": (10.0, None), "close": (10.0, 12.0)},
            params_override={"window_short": 1, "window_medium": 1, "window_long": 2},
            expected=(None, math.inf),
            reason="a missing low sends the true range to zero through the prior-close fallback while the buying "
            "pressure stays positive, so the quotient is +/-inf — the infinity beside the 0/0 NaN pin",
        ),
    ),
)
