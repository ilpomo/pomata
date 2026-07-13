"""Spec for ``pomata.metrics.win_rate`` — reducing, the fraction of decisive returns that are gains, scale-invariant."""

from tests.metrics.oracles import win_rate_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import win_rate

WIN_RATE = Spec(
    factory=win_rate,
    inputs=("returns",),
    params={},
    shape=Shape.REDUCING,
    oracle=win_rate_reference,
    # Signs are unchanged by a positive rescale
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)},
    golden_output=(0.5714,),
    pins=(
        SpecPin(
            label="single_row_positive_win",
            inputs={"returns": (0.05,)},
            expected=(1.0,),
            reason="one positive decisive return gives a rate of 1",
        ),
        SpecPin(
            label="all_positive_is_one",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(1.0,),
            reason="an all-positive series wins every decisive period",
        ),
        SpecPin(
            label="all_negative_is_zero",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(0.0,),
            reason="an all-negative series wins no decisive period",
        ),
        SpecPin(
            label="all_zero_is_null",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(None,),
            reason="an all-zero series has no decisive returns, so the rate is null ",
        ),
        SpecPin(
            label="zero_excluded_from_denominator",
            inputs={"returns": (0.01, 0.0, 0.02)},
            expected=(1.0,),
            reason="an exact-zero return is excluded from the denominator ",
        ),
    ),
)
