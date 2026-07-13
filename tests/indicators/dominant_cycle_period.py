"""Spec for ``pomata.indicators.dominant_cycle_period`` — Ehlers' Hilbert dominant-cycle length, latching, invariant."""

import math

from tests.indicators.oracles import dominant_cycle_period_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import dominant_cycle_period

# A clean 20-bar-period carrier: 40 bars leave 8 emitted values past the 32-bar settling warm-up.
_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(40))

DOMINANT_CYCLE_PERIOD = Spec(
    factory=dominant_cycle_period,
    inputs=("expr",),
    params={},
    shape=Shape.SERIES,
    warmup=32,
    oracle=dominant_cycle_period_reference,
    # A cycle length in bars, clamped to [6, 50]: scale-INVARIANT, degree 0.
    scale=(ScaleAxis(roles=("expr",), degree=0),),
    golden_input={"expr": _SAMPLE},
    golden_output=(None,) * 32 + (19.0186, 19.3994, 19.7391, 20.051, 20.3271, 20.5471, 20.6936, 20.7611),
    pins=(
        SpecPin(
            label="flat_run_matches_reference",
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
