"""Declaration for ``pomata.metrics.ulcer_index`` — reducing, the RMS drawdown, scale-invariant."""

from pomata.metrics import ulcer_index
from tests_new.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests_new.metrics.harness import suite_metrics
from tests_new.metrics.oracles import reference_ulcer_index
from tests_new.support.declaration import Golden, Pin, ScaleAxis

ULCER_INDEX = suite_metrics(
    factory=ulcer_index,
    inputs=("equity_curve",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    oracle=reference_ulcer_index,
    scaling=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden=Golden(inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2, 0.9, 1.0)}, output=(0.1241,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a one-element series has no drawdown, so the Ulcer Index is 0",
        ),
        Pin(
            label="monotonic_rise_is_zero",
            inputs={"equity_curve": (1.0, 1.1, 1.2, 1.3)},
            expected=(0.0,),
            reason="a never-declining curve has all-zero drawdowns, so the Ulcer Index is exactly 0",
        ),
    ),
)
