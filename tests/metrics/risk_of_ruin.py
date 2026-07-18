"""Declaration for ``pomata.metrics.risk_of_ruin`` — reducing, the gambler's-ruin probability from the win rate."""

from pomata.metrics import risk_of_ruin
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_risk_of_ruin
from tests.support.declaration import Golden, Pin, ScaleAxis

RISK_OF_RUIN = suite_metrics(
    factory=risk_of_ruin,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_risk_of_ruin,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    golden=Golden(inputs={"returns": (0.02, -0.01, 0.03, -0.02)}, output=(1.0,)),
    pins=(
        Pin(label="single_row", inputs={"returns": (0.05,)}, expected=(0.0,), reason="a single win (p=1) gives ruin 0"),
        Pin(
            label="all_wins_is_zero",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(0.0,),
            reason="an all-winning series (p=1) has no ruin risk",
        ),
        Pin(
            label="all_losses_is_one",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(1.0,),
            reason="an all-losing series (p=0) is certain ruin",
        ),
        Pin(
            label="all_zero_is_null",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(None,),
            reason="a series of exact-zero returns has no decisive bars, so the win rate and ruin are null ",
        ),
    ),
)
