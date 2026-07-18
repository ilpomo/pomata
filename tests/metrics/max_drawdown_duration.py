"""
Declaration for ``pomata.metrics.max_drawdown_duration`` — reducing, the longest underwater run, scale-invariant.
"""

from pomata.metrics import max_drawdown_duration
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_max_drawdown_duration
from tests.support.declaration import Golden, Pin, ScaleAxis

MAX_DRAWDOWN_DURATION = suite_metrics(
    factory=max_drawdown_duration,
    inputs=("equity_curve",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    oracle=reference_max_drawdown_duration,
    scaling=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden=Golden(inputs={"equity_curve": (1.0, 0.9, 0.8, 0.85, 1.1, 1.05)}, output=(3.0,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a one-element series is never underwater, so the duration is 0 ",
        ),
        Pin(
            label="no_drawdown_is_zero",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(0.0,),
            reason="a monotonically rising curve is never underwater, so the duration is 0 ",
        ),
    ),
)
