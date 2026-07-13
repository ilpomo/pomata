"""Spec for ``pomata.metrics.risk_of_ruin`` — reducing, the gambler's-ruin probability from the win rate,
scale-invariant.
"""

from tests.metrics.oracles import risk_of_ruin_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import risk_of_ruin

RISK_OF_RUIN = Spec(
    factory=risk_of_ruin,
    inputs=("returns",),
    params={},
    shape=Shape.REDUCING,
    oracle=risk_of_ruin_reference,
    # Built on a pure sign-count win rate, unchanged by a positive rescale
    #
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.02, -0.01, 0.03, -0.02)},
    golden_output=(1.0,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(0.0,),
            reason="a single win (p=1) gives ruin 0",
        ),
        SpecPin(
            label="all_wins_is_zero",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(0.0,),
            reason="an all-winning series (p=1) has no ruin risk",
        ),
        SpecPin(
            label="all_losses_is_one",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(1.0,),
            reason="an all-losing series (p=0) is certain ruin",
        ),
        SpecPin(
            label="all_zero_is_null",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(None,),
            reason="a series of exact-zero returns has no decisive bars, so the win rate and ruin are null ",
        ),
    ),
)
