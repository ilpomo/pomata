"""Declaration for ``pomata.metrics.total_return`` — reducing, the final growth factor minus one, scale-exempt."""

from pomata.metrics import total_return
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_total_return
from tests.support.declaration import Golden, Pin, ScaleExempt

TOTAL_RETURN = suite_metrics(
    factory=total_return,
    inputs=("equity_curve",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    oracle=reference_total_return,
    scaling=ScaleExempt(
        reason="a growth-factor series normalized to a unit start (the result is the final value minus "
        "one), neither scale-homogeneous nor scale-invariant"
    ),
    golden=Golden(inputs={"equity_curve": (1.1, 1.045, 1.254, 1.3794)}, output=(0.3794,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"equity_curve": (1.21,)},
            expected=(0.21,),
            reason="a one-element series resolves to the final growth minus one ",
        ),
    ),
)
