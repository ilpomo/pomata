"""Declaration for ``pomata.metrics.pain_index`` — reducing, the mean absolute drawdown, scale-invariant."""

from pomata.metrics import pain_index
from tests_new.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests_new.metrics.harness import suite_metrics
from tests_new.metrics.oracles import reference_pain_index
from tests_new.support.declaration import Golden, Pin, ScaleAxis

PAIN_INDEX = suite_metrics(
    factory=pain_index,
    inputs=("equity_curve",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    oracle=reference_pain_index,
    scaling=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden=Golden(inputs={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)}, output=(0.0179,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a one-element series is at its own peak, so the pain index is exactly 0 ",
        ),
        Pin(
            label="no_drawdown_is_zero",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(0.0,),
            reason="a monotonically rising curve is never below its running peak, so the mean drawdown is 0 ",
        ),
    ),
)
