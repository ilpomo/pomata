"""Spec for ``pomata.indicators.dm_minus`` — Wilder's smoothed negative directional movement, gap-bridging, degree-1."""

import math

from tests.indicators.oracles import dm_minus_reference
from tests_new.support.spec import Deviant, ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import dm_minus

DM_MINUS = Spec(
    factory=dm_minus,
    inputs=("high", "low"),
    params={"window": 2},
    shape=Shape.SERIES,
    warmup=1,
    lands_on="low",
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=dm_minus_reference,
    # A price-difference smoothed by a Wilder rma, homogeneous of degree 1 (tests/indicators/test_dm_minus.py
    # ::TestDmMinusProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("high", "low"), degree=1),),
    all_null=Deviant(
        expected=(None,) + (0.0,) * 11,
        reason="a null high or low leaves the raw -DM at 0 (the down-move guard is unsatisfied), which the Wilder rma "
        "smooths to 0 past the one-row warm-up (tests/indicators/test_dm_minus.py::TestDmMinusEdge::test_all_null)",
    ),
    flow_deviation="the down-move guard turns a fully-missing bar into 0 movement, so a full-bar null / NaN is "
    "absorbed and the rma recurrence continues at 0 (never a null trace or a latch), while a single-column NaN on the "
    "low leg still latches — one shared policy cannot hold both; pinned below and covered by the missing-data tier",
    golden_input={
        "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0),
        "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0),
    },
    golden_output=(None, 0.0, 0.0, 0.25, 0.125, 0.3125, 0.1562),
    pins=(
        SpecPin(
            label="interior_null_bridged",
            inputs={
                "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
                "low": (9.0, 10.0, 11.0, None, 12.0, 12.5, 13.0, 12.5),
            },
            expected=(None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25),
            reason="an interior null makes the raw -DM 0 there and the rma carries its state across the gap (no null "
            "trace) (test_dm_minus.py::TestDmMinusEdge::test_null_bridged)",
        ),
        SpecPin(
            label="nan_on_low_leg_latches",
            inputs={
                "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
                "low": (9.0, 10.0, 11.0, 11.5, 12.0, math.nan, 13.0, 12.5),
            },
            expected=(None, 0.0, 0.0, 0.0, 0.0, math.nan, math.nan, math.nan),
            reason="a NaN on the low leg (the down-move driver) poisons the raw -DM and latches the rma forever "
            "(test_dm_minus.py::TestDmMinusEdge::test_nan_latches)",
        ),
    ),
)
