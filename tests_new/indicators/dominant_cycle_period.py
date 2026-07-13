"""Spec for ``pomata.indicators.dominant_cycle_period`` — Ehlers' Hilbert dominant-cycle length, latching, invariant."""

import math

import polars as pl
from tests.indicators.oracles import dominant_cycle_period_reference
from tests.support import spans_even_lag_repeat
from tests_new.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import dominant_cycle_period

# A clean 20-bar-period carrier: 40 bars leave 8 emitted values past the 32-bar settling warm-up (the old golden).
_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(40))


def _no_even_lag_repeat(frame: pl.DataFrame) -> bool:
    """Exclude the shared cycle-pipeline degenerate: an even-lag repeat drives the in-phase branch a residual apart."""
    finite = [value for value in frame.to_series(0).to_list() if value is not None and math.isfinite(value)]
    return not spans_even_lag_repeat(finite)


DOMINANT_CYCLE_PERIOD = Spec(
    factory=dominant_cycle_period,
    inputs=("expr",),
    params={},
    shape=Shape.SERIES,
    warmup=32,
    oracle=dominant_cycle_period_reference,
    conditioning=_no_even_lag_repeat,
    # A cycle length in bars, clamped to [6, 50]: scale-INVARIANT, degree 0 (tests/indicators/
    # test_dominant_cycle_period.py::TestDominantCyclePeriodProperties::test_scale_invariance).
    scale=(ScaleAxis(roles=("expr",), degree=0),),
    golden_input={"expr": _SAMPLE},
    golden_output=(None,) * 32 + (19.0186, 19.3994, 19.7391, 20.051, 20.3271, 20.5471, 20.6936, 20.7611),
)
