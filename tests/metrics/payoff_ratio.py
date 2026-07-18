"""
Declaration for ``pomata.metrics.payoff_ratio`` — reducing, average win over average loss magnitude, scale-
invariant.
"""

from pomata.metrics import payoff_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_payoff_ratio
from tests.support.declaration import Golden, Pin, ScaleAxis

PAYOFF_RATIO = suite_metrics(
    factory=payoff_ratio,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_payoff_ratio,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    golden=Golden(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, output=(1.0833,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(None,),
            reason="a one-element series leaves one side empty, so the ratio is null ",
        ),
        Pin(
            label="no_losses_is_null",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(None,),
            reason="an all-positive series has no losing side, so the ratio is null ",
        ),
        Pin(
            label="no_gains_is_null",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(None,),
            reason="an all-negative series has no winning side, so the ratio is null ",
        ),
    ),
)
