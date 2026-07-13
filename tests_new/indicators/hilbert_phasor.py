"""Spec for ``pomata.indicators.hilbert_phasor`` — Ehlers' in-phase/quadrature phasor struct, latching, degree-1."""

import math

import polars as pl
from tests_new.indicators.oracles import hilbert_phasor_reference
from tests_new.support import spans_even_lag_repeat
from tests_new.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import hilbert_phasor

# A clean 20-bar-period carrier: 40 bars leave 8 emitted values past the 32-bar settling warm-up (the old golden).
_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(40))


def _no_even_lag_repeat(frame: pl.DataFrame) -> bool:
    """Exclude the shared cycle-pipeline degenerate: an even-lag repeat drives the phasor to a cancellation residual."""
    finite = [value for value in frame.to_series(0).to_list() if value is not None and math.isfinite(value)]
    return not spans_even_lag_repeat(finite)


HILBERT_PHASOR = Spec(
    factory=hilbert_phasor,
    inputs=("expr",),
    params={},
    shape=Shape.STRUCT,
    fields=("in_phase", "quadrature"),
    warmup=32,
    oracle=hilbert_phasor_reference,
    conditioning=_no_even_lag_repeat,
    # Both components carry the price's units, homogeneous of degree 1 (tests/indicators/test_hilbert_phasor.py
    # ::TestHilbertPhasorProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("expr",), degree=1),),
    golden_input={"expr": _SAMPLE},
    golden_output={
        "in_phase": (None,) * 32 + (-0.0296, -2.9751, -5.7341, -7.9735, -9.445, -10.0056, -9.5949, -8.227),
        "quadrature": (None,) * 32 + (-9.5596, -9.5537, -8.4419, -6.3428, -3.5083, -0.2378, 3.1502, 6.3025),
    },
)
