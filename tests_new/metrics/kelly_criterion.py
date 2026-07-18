"""Declaration for ``pomata.metrics.kelly_criterion`` — reducing, the optimal betting fraction, scale-invariant."""

from pomata.metrics import kelly_criterion
from tests_new.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests_new.metrics.harness import suite_metrics
from tests_new.metrics.oracles import reference_kelly_criterion
from tests_new.support.declaration import Golden, Pin, ScaleAxis

KELLY_CRITERION = suite_metrics(
    factory=kelly_criterion,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_kelly_criterion,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    golden=Golden(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, output=(0.1758,)),
    pins=(
        Pin(
            label="single_row_one_sided",
            inputs={"returns": (0.02,)},
            expected=(None,),
            reason="a one-element series is one-sided, so the payoff ratio is undefined and the fraction is null",
        ),
        Pin(
            label="no_losses_is_null",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(None,),
            reason="an all-positive series has an undefined payoff ratio, so the fraction is null ",
        ),
        Pin(
            label="no_wins_is_null",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(None,),
            reason="an all-negative series has an undefined payoff ratio, so the fraction is null ",
        ),
    ),
)
