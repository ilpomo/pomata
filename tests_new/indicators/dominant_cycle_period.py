"""Spec for ``pomata.indicators.dominant_cycle_period`` — Ehlers' Hilbert dominant-cycle length, latching, invariant."""

import math

from tests_new.indicators.oracles import dominant_cycle_period_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import dominant_cycle_period

# A clean 20-bar-period carrier: 40 bars leave 8 emitted values past the 32-bar settling warm-up (the old golden).
_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(40))

DOMINANT_CYCLE_PERIOD = Spec(
    factory=dominant_cycle_period,
    inputs=("expr",),
    params={},
    shape=Shape.SERIES,
    warmup=32,
    oracle=dominant_cycle_period_reference,
    # A cycle length in bars, clamped to [6, 50]: scale-INVARIANT, degree 0 (tests/indicators/
    # test_dominant_cycle_period.py::TestDominantCyclePeriodProperties::test_scale_invariance).
    scale=(ScaleAxis(roles=("expr",), degree=0),),
    golden_input={"expr": _SAMPLE},
    golden_output=(None,) * 32 + (19.0186, 19.3994, 19.7391, 20.051, 20.3271, 20.5471, 20.6936, 20.7611),
    pins=(
        SpecPin(
            label="flat_run_matches_reference",
            inputs={"expr": (100.0,) * 48},
            expected=(None,) * 32
            + (
                24.70674364190461,
                25.759785209283244,
                26.659016921991494,
                27.419872243335252,
                28.059073468742767,
                28.593066571367846,
                29.037173224588177,
                29.405188650757665,
                29.709249339159193,
                29.95985763720833,
                30.165991597698344,
                30.335255509873,
                30.474044091510358,
                30.587704598460938,
                30.680688264307367,
                30.75668694251152,
            ),
            reason="the cycle-pipeline degenerate (a flat run) the old suite filtered out of the property tiers: the "
            "period clamp and the AND-gated phase update damp the phasor residual before it reaches this output, so "
            "impl and oracle agree there (measured worst relative deviation ~3.4e-15) and no conditioning filter is "
            "declared — the corner stays witnessed by this fixed case instead",
        ),
    ),
)
