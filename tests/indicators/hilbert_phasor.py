"""Spec for ``pomata.indicators.hilbert_phasor`` — Ehlers' in-phase/quadrature phasor struct, latching, degree-1."""

import math

from tests.indicators.oracles import hilbert_phasor_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import hilbert_phasor

# A clean 20-bar-period carrier: 40 bars leave 8 emitted values past the 32-bar settling warm-up (the old golden).
_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(40))

HILBERT_PHASOR = Spec(
    factory=hilbert_phasor,
    inputs=("expr",),
    params={},
    shape=Shape.STRUCT,
    fields=("in_phase", "quadrature"),
    warmup={"in_phase": 32, "quadrature": 32},
    oracle=hilbert_phasor_reference,
    # Both components carry the price's units, homogeneous of degree 1 (tests/indicators/test_hilbert_phasor.py
    # ::TestHilbertPhasorProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("expr",), degree={"in_phase": 1, "quadrature": 1}),),
    golden_input={"expr": _SAMPLE},
    golden_output={
        "in_phase": (None,) * 32 + (-0.0296, -2.9751, -5.7341, -7.9735, -9.445, -10.0056, -9.5949, -8.227),
        "quadrature": (None,) * 32 + (-9.5596, -9.5537, -8.4419, -6.3428, -3.5083, -0.2378, 3.1502, 6.3025),
    },
    pins=(
        SpecPin(
            label="flat_run_collapses_to_cancellation_residual",
            inputs={"expr": (100.0,) * 40},
            expected={
                "in_phase": (None,) * 32 + (0.0,) * 8,
                "quadrature": (None,) * 32 + (0.0,) * 8,
            },
            reason="the cycle-pipeline degenerate (a flat run) the old suite filtered out of the property tiers: this "
            "struct exports the raw FIR components, so both collapse to the ~1e-14 cancellation residual itself, "
            "which the property tiers' absolute band absorbs (measured worst deviation ~3.4e-13) — no conditioning "
            "filter is declared and the corner stays witnessed by this fixed case instead: at the declared "
            "rounding the residual reads as an exact 0.0 on every platform, which IS the fact worth pinning",
            round_to=4,
        ),
    ),
)
