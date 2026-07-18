"""Declaration for ``pomata.indicators.awesome_oscillator`` — the SMA-of-median difference, window-nulling, degree-1."""

from pomata.indicators import awesome_oscillator
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_awesome_oscillator
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape
from tests_new.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE

AWESOME_OSCILLATOR = suite_indicators(
    factory=awesome_oscillator,
    inputs=("high", "low"),
    params={"window_fast": 5, "window_slow": 34},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=33,
    oracle=reference_awesome_oscillator,
    scaling=(ScaleAxis(roles=("high", "low"), degree=1),),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no Awesome Oscillator.",
    raises=(
        ({"window_fast": 0}, r"window_fast must be >= 1"),
        ({"window_slow": 0}, r"window_slow must be >= 1"),
        ({"window_fast": 5, "window_slow": 3}, r"windows must be ordered window_fast <= window_slow"),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    golden=Golden(
        inputs={"high": (2.0, 4.0, 6.0, 8.0, 10.0), "low": (0.0, 2.0, 4.0, 6.0, 8.0)},
        output=(None, None, 1.0, 1.0, 1.0),
        params={"window_fast": 2, "window_slow": 3},
    ),
    pins=(
        Pin(
            label="single_row_equal_windows",
            inputs={"high": (2.0,), "low": (0.0,)},
            params_override={"window_fast": 1, "window_slow": 1},
            expected=(0.0,),
            reason="window_fast == window_slow == 1 on one bar gives 0",
        ),
        Pin(
            label="single_row_warmup",
            inputs={"high": (2.0,), "low": (0.0,)},
            params_override={"window_fast": 1, "window_slow": 3},
            expected=(None,),
            reason="a slow window of 3 on one bar is still warm-up",
        ),
        Pin(
            label="equal_windows_is_zero",
            inputs={"high": (2.0, 4.0, 6.0, 8.0), "low": (0.0, 2.0, 4.0, 6.0)},
            params_override={"window_fast": 2, "window_slow": 2},
            expected=(None, 0.0, 0.0, 0.0),
            reason="equal windows give an identically-zero oscillator where defined",
        ),
        Pin(
            label="flat_series_is_zero",
            inputs={"high": (5.0, 5.0, 5.0, 5.0, 5.0), "low": (5.0, 5.0, 5.0, 5.0, 5.0)},
            params_override={"window_fast": 2, "window_slow": 3},
            expected=(None, None, 0.0, 0.0, 0.0),
            reason="over a constant median both averages equal it, so AO is 0",
        ),
    ),
)
