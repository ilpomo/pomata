"""Spec for ``pomata.indicators.hilbert_trendline`` — Ehlers' instantaneous trendline, latching, degree-1."""

import math

from tests.indicators.oracles import hilbert_trendline_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import hilbert_trendline

# A 0.5/bar trend plus a clean 20-bar cycle: 80 bars leave 17 emitted values past the 63-bar warm-up (the old golden).
_SAMPLE = tuple(100.0 + 0.5 * index + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(80))

HILBERT_TRENDLINE = Spec(
    factory=hilbert_trendline,
    inputs=("expr",),
    params={},
    shape=Shape.SERIES,
    warmup=63,
    oracle=hilbert_trendline_reference,
    # The trendline rides the price scale, homogeneous of degree 1 (tests/indicators/test_hilbert_trendline.py
    # ::TestHilbertTrendlineProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("expr",), degree=1),),
    golden_input={"expr": _SAMPLE},
    golden_output=(None,) * 63
    + (
        126.2134,
        126.7457,
        127.253,
        127.75,
        128.25,
        128.75,
        129.35,
        129.825,
        130.3,
        130.775,
        131.25,
        131.75,
        131.9595,
        132.251,
        132.6398,
        133.1343,
        133.7348,
    ),
    pins=(
        SpecPin(
            label="flat_run_is_the_constant",
            inputs={"expr": (100.0,) * 80},
            expected=(None,) * 63 + (100.0,) * 17,
            reason="the cycle-pipeline degenerate (a flat run) the old suite filtered out of the property tiers: the "
            "trendline reads a downstream EWMA of the price, so the phasor's cancellation residual never reaches "
            "this output — impl and oracle are bit-identical here (measured deviation exactly 0.0; the old filter's "
            "copy-pasted branch-flip reason did not hold for this indicator) — no conditioning filter is declared "
            "and the corner stays witnessed by this fixed case instead",
        ),
    ),
)
