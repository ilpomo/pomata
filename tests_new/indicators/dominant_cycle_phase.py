"""Spec for ``pomata.indicators.dominant_cycle_phase`` — Ehlers' Hilbert dominant-cycle phase, latching, invariant."""

import math

import polars as pl
from tests_new.indicators.oracles import dominant_cycle_phase_reference
from tests_new.support import spans_even_lag_repeat
from tests_new.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import dominant_cycle_phase

# A clean 20-bar-period carrier: 80 bars leave 17 emitted values past the 63-bar settling warm-up (the old golden).
_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(80))


def _no_even_lag_repeat(frame: pl.DataFrame) -> bool:
    """Exclude the shared cycle-pipeline degenerate: an even-lag repeat flips the phase branch a residual apart."""
    finite = [value for value in frame.to_series(0).to_list() if value is not None and math.isfinite(value)]
    return not spans_even_lag_repeat(finite)


DOMINANT_CYCLE_PHASE = Spec(
    factory=dominant_cycle_phase,
    inputs=("expr",),
    params={},
    shape=Shape.SERIES,
    warmup=63,
    oracle=dominant_cycle_phase_reference,
    conditioning=_no_even_lag_repeat,
    # A phase in degrees: scale-INVARIANT, degree 0 (tests/indicators/test_dominant_cycle_phase.py
    # ::TestDominantCyclePhaseProperties::test_scale_invariance).
    scale=(ScaleAxis(roles=("expr",), degree=0),),
    golden_input={"expr": _SAMPLE},
    golden_output=(None,) * 63
    + (
        54.1853,
        72.1855,
        90.1782,
        108.1678,
        126.1594,
        144.1573,
        162.1633,
        180.1763,
        198.1917,
        216.204,
        234.2083,
        252.2035,
        270.1915,
        288.177,
        306.1651,
        -35.84,
        -17.8363,
    ),
)
