"""Spec for ``pomata.indicators.dm_plus`` — Wilder's smoothed positive directional movement, gap-bridging, degree-1."""

import math

from tests.indicators.oracles import dm_plus_reference
from tests.support.spec import Deviant, ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import dm_plus

DM_PLUS = Spec(
    factory=dm_plus,
    inputs=("high", "low"),
    params={"window": 2},
    shape=Shape.SERIES,
    warmup=1,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=dm_plus_reference,
    # A price-difference smoothed by a Wilder rma, homogeneous of degree 1.
    scale=(ScaleAxis(roles=("high", "low"), degree=1),),
    all_null=Deviant(
        expected=(None,) + (0.0,) * 11,
        reason="a null high or low leaves the raw +DM at 0 (the up-move guard is unsatisfied), which the Wilder rma "
        "smooths to 0 past the one-row warm-up",
    ),
    flow_deviation="the up-move guard turns a fully-missing bar into 0 movement, so a full-bar null / NaN is absorbed "
    "and the rma recurrence continues at 0 (never a null trace or a latch), while a single-column NaN on the high leg "
    "still latches — one shared policy cannot hold both; pinned below and covered by the missing-data property tier",
    golden_input={
        "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0),
        "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0),
    },
    golden_output=(None, 0.5, 0.75, 0.375, 0.9375, 0.4688, 0.9844),
    pins=(
        SpecPin(
            label="interior_null_bridged",
            inputs={
                "high": (10.0, 11.0, 12.0, None, 13.0, 13.5, 14.0, 13.5),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5),
            },
            expected=(None, 0.5, 0.75, 0.375, 0.1875, 0.09375, 0.296875, 0.1484375),
            reason="an interior null makes the raw +DM 0 there and the rma carries its state across the gap (no null "
            "trace)",
        ),
        SpecPin(
            label="nan_on_high_leg_latches",
            inputs={
                "high": (10.0, 11.0, 12.0, 12.5, 13.0, math.nan, 14.0, 13.5),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5),
            },
            expected=(None, 0.5, 0.75, 0.375, 0.4375, math.nan, math.nan, math.nan),
            reason="a NaN on the high leg (the up-move driver) poisons the raw +DM and latches the rma forever",
        ),
    ),
)
