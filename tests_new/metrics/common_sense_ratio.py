"""Spec for ``pomata.metrics.common_sense_ratio`` — reducing, profit factor times tail ratio, scale-invariant."""

import math

import polars as pl
from tests_new.metrics.oracles import common_sense_ratio_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import common_sense_ratio, profit_factor, tail_ratio

COMMON_SENSE_RATIO = Spec(
    factory=common_sense_ratio,
    inputs=("returns",),
    params={},
    shape=Shape.REDUCING,
    oracle=common_sense_ratio_reference,
    # A product of two scale-invariant factors (test_common_sense_ratio.py::test_scale_invariance).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)},
    golden_output=(2.1081,),
    component_expr=lambda: profit_factor(pl.col("returns")) * tail_ratio(pl.col("returns")),
    pins=(
        SpecPin(
            label="single_row_all_loss",
            inputs={"returns": (-0.02,)},
            expected=(0.0,),
            reason="a one-element loss has profit factor 0 and tail ratio 1, so the product is 0 "
            "(test_common_sense_ratio.py::test_single_row)",
        ),
        SpecPin(
            label="no_losses_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(math.inf,),
            reason="an all-positive series has an infinite profit factor and a finite positive tail ratio, so +inf "
            "(test_common_sense_ratio.py::test_no_losses_is_inf)",
        ),
    ),
)
