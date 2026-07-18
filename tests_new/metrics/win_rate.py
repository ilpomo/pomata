"""
Declaration for ``pomata.metrics.win_rate`` — reducing, the fraction of decisive returns that are gains, scale-
invariant.
"""

from pomata.metrics import win_rate
from tests_new.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests_new.metrics.harness import suite_metrics
from tests_new.metrics.oracles import reference_win_rate
from tests_new.support.declaration import Golden, Pin, ScaleAxis

WIN_RATE = suite_metrics(
    factory=win_rate,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.COLLAPSES,
    oracle=reference_win_rate,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    golden=Golden(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, output=(0.5714,)),
    pins=(
        Pin(
            label="single_row_positive_win",
            inputs={"returns": (0.05,)},
            expected=(1.0,),
            reason="one positive decisive return gives a rate of 1",
        ),
        Pin(
            label="all_positive_is_one",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(1.0,),
            reason="an all-positive series wins every decisive period",
        ),
        Pin(
            label="all_negative_is_zero",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(0.0,),
            reason="an all-negative series wins no decisive period",
        ),
        Pin(
            label="all_zero_is_null",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(None,),
            reason="an all-zero series has no decisive returns, so the rate is null ",
        ),
        Pin(
            label="zero_excluded_from_denominator",
            inputs={"returns": (0.01, 0.0, 0.02)},
            expected=(1.0,),
            reason="an exact-zero return is excluded from the denominator ",
        ),
    ),
)
