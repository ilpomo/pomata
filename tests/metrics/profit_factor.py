"""Declaration for ``pomata.metrics.profit_factor`` — reducing, gross gains over gross losses, scale-invariant."""

import math

from pomata.metrics import profit_factor
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_profit_factor
from tests.support.declaration import Golden, Pin, ScaleAxis

PROFIT_FACTOR = suite_metrics(
    factory=profit_factor,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_profit_factor,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    golden=Golden(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, output=(1.4444,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(math.inf,),
            reason="a single gain has zero gross loss, so the factor is +inf",
        ),
        Pin(
            label="no_losses_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(math.inf,),
            reason="an all-positive series has no losses, so the ratio is +inf ",
        ),
        Pin(
            label="no_gains_is_zero",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(0.0,),
            reason="an all-negative series has no gains, so the ratio is 0 ",
        ),
        Pin(
            label="all_zero_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series has zero gains and losses, so the ratio is 0/0, i.e. NaN ",
        ),
    ),
)
