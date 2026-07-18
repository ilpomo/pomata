"""Declaration for ``pomata.metrics.max_drawdown`` — reducing, the deepest peak-to-trough decline, scale-invariant."""

from pomata.metrics import max_drawdown
from tests_new.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests_new.metrics.harness import suite_metrics
from tests_new.metrics.oracles import reference_max_drawdown
from tests_new.support.declaration import Golden, Pin, ScaleAxis

MAX_DRAWDOWN = suite_metrics(
    factory=max_drawdown,
    inputs=("equity_curve",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    oracle=reference_max_drawdown,
    scaling=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden=Golden(inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2, 0.9, 1.0)}, output=(-0.25,)),
    pins=(
        Pin(
            label="single_row_is_zero",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a one-element series is at its own peak, so the maximum drawdown is 0",
        ),
        Pin(
            label="monotonic_rise_is_zero",
            inputs={"equity_curve": (1.0, 1.1, 1.2, 1.3)},
            expected=(0.0,),
            reason="a never-declining curve has zero drawdown",
        ),
    ),
)
