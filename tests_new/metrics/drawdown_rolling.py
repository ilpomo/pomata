"""
Declaration for ``pomata.metrics.drawdown_rolling`` — the decline from the trailing-window peak, scale-invariant.
"""

from pomata.metrics import drawdown_rolling
from tests_new.metrics.drawdown import DRAWDOWN
from tests_new.metrics.enums import BehaviorNan, BehaviorNull
from tests_new.metrics.harness import suite_metrics
from tests_new.metrics.oracles import reference_drawdown_rolling
from tests_new.support.declaration import Golden, Pin, ScaleAxis

DRAWDOWN_ROLLING = suite_metrics(
    factory=drawdown_rolling,
    inputs=("equity_curve",),
    params={"window": 3},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    rolling_of=DRAWDOWN,
    window="window",
    warmup=2,
    oracle=reference_drawdown_rolling,
    scaling=(ScaleAxis(roles=("equity_curve",), degree=0),),
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25)},
        output=(None, None, -0.0455, 0.0, -0.0417, 0.0, -0.0385),
    ),
    pins=(
        Pin(
            label="window_equals_length",
            inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2)},
            expected=(None, None, None, 0.0),
            reason="when the window exactly equals the series length only the last row is defined ",
            params_override={"window": 4},
        ),
        Pin(
            label="window_peak_is_zero",
            inputs={"equity_curve": (1.0, 1.1, 1.2, 1.3)},
            expected=(None, None, 0.0, 0.0),
            reason="at a monotonically rising window's peak the drawdown is exactly 0 ",
        ),
    ),
)
