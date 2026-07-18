"""Declaration for ``pomata.metrics.gain_to_pain_ratio`` — reducing, net return over total loss, scale-invariant."""

import math

from pomata.metrics import gain_to_pain_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_gain_to_pain_ratio
from tests.support.declaration import Golden, Pin, ScaleAxis

GAIN_TO_PAIN_RATIO = suite_metrics(
    factory=gain_to_pain_ratio,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_gain_to_pain_ratio,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    golden=Golden(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, output=(0.4444,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.02,)},
            expected=(math.inf,),
            reason="a one-element positive series has no loss, so the ratio is +inf ",
        ),
        Pin(
            label="no_losses_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(math.inf,),
            reason="an all-positive series has no loss, so the ratio is +inf ",
        ),
        Pin(
            label="all_negative_is_minus_one",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(-1.0,),
            reason="an all-negative series has net loss equal to its total loss, so the ratio is -1 ",
        ),
        Pin(
            label="all_zero_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series has zero total loss and zero net return, so the ratio is a 0/0, i.e. "
            "NaN — the degenerate-denominator NaN beside the +inf pin",
        ),
    ),
)
