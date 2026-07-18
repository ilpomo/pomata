"""
Declaration for ``pomata.metrics.common_sense_ratio`` — reducing, profit factor times tail ratio, scale-invariant.
"""

import math

import polars as pl

from pomata.metrics import common_sense_ratio, profit_factor, tail_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_common_sense_ratio
from tests.support.declaration import Golden, Pin, ScaleAxis

COMMON_SENSE_RATIO = suite_metrics(
    factory=common_sense_ratio,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_common_sense_ratio,
    recomposition=lambda: profit_factor(pl.col("returns")) * tail_ratio(pl.col("returns")),
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    golden=Golden(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, output=(2.1081,)),
    pins=(
        Pin(
            label="single_row_all_loss",
            inputs={"returns": (-0.02,)},
            expected=(0.0,),
            reason="a one-element loss has profit factor 0 and tail ratio 1, so the product is 0 ",
        ),
        Pin(
            label="no_losses_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(math.inf,),
            reason="an all-positive series has an infinite profit factor and a finite positive tail ratio, so +inf",
        ),
        Pin(
            label="all_zero_is_nan",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(math.nan,),
            reason="an all-zero series drives both the profit factor and the tail ratio to a 0/0, so the "
            "product is NaN — the degenerate-denominator NaN beside the +inf pin",
        ),
    ),
)
