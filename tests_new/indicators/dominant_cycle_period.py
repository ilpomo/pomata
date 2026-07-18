"""
Declaration for ``pomata.indicators.dominant_cycle_period`` — Ehlers' Hilbert dominant-cycle length, latching,
invariant.
"""

import math

from pomata.indicators import dominant_cycle_period
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_dominant_cycle_period
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(40))

DOMINANT_CYCLE_PERIOD = suite_indicators(
    factory=dominant_cycle_period,
    inputs=("expr",),
    params={},
    null=BehaviorNull.LATCHES,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=32,
    oracle=reference_dominant_cycle_period,
    scaling=(ScaleAxis(roles=("expr",), degree=0),),
    talib=RelationTalib.MATCHES,
    golden=Golden(
        inputs={"expr": _SAMPLE},
        output=(None,) * 32 + (19.0186, 19.3994, 19.7391, 20.051, 20.3271, 20.5471, 20.6936, 20.7611),
    ),
    pins=(
        Pin(
            label="reference_flat_run_matches",
            inputs={"expr": (100.0,) * 48},
            expected=(None,) * 32
            + (
                24.7067,
                25.7598,
                26.659,
                27.4199,
                28.0591,
                28.5931,
                29.0372,
                29.4052,
                29.7092,
                29.9599,
                30.166,
                30.3353,
                30.474,
                30.5877,
                30.6807,
                30.7567,
            ),
            reason="the cycle-pipeline degenerate (a flat run): the period clamp and the AND-gated phase update damp "
            "the phasor residual before it reaches this output, so impl and oracle agree there (measured worst "
            "relative deviation ~3.4e-15) and no conditioning filter is declared — the corner stays witnessed by "
            "this fixed case, rounded because the degenerate pipeline settles on libm-dependent fixed points that "
            "differ across OS math libraries",
            round_to=4,
        ),
    ),
)
