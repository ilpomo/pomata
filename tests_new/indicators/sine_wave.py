"""Spec for ``pomata.indicators.sine_wave`` — Ehlers' sine / lead-sine struct, latching, scale-invariant."""

import math

import polars as pl
from tests.indicators.oracles import sine_wave_reference
from tests.support import spans_even_lag_repeat
from tests_new.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import sine_wave

# A clean 20-bar-period carrier: 80 bars leave 17 emitted values past the 63-bar settling warm-up (the old golden).
_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(80))


def _no_even_lag_repeat(frame: pl.DataFrame) -> bool:
    """Exclude the shared cycle-pipeline degenerate: an even-lag repeat flips the phase that fixes both lines."""
    finite = [value for value in frame.to_series(0).to_list() if value is not None and math.isfinite(value)]
    return not spans_even_lag_repeat(finite)


SINE_WAVE = Spec(
    factory=sine_wave,
    inputs=("expr",),
    params={},
    shape=Shape.STRUCT,
    fields=("sine", "lead_sine"),
    warmup=63,
    oracle=sine_wave_reference,
    conditioning=_no_even_lag_repeat,
    # Both lines are the sine of a phase, bounded in [-1, 1] and scale-INVARIANT, degree 0 (tests/indicators/
    # test_sine_wave.py::TestSineWaveProperties::test_scale_invariance).
    scale=(ScaleAxis(roles=("expr",), degree=0),),
    golden_input={"expr": _SAMPLE},
    golden_output={
        "sine": (None,) * 63
        + (
            0.8109,
            0.9521,
            1.0,
            0.9501,
            0.8074,
            0.5856,
            0.3063,
            -0.0031,
            -0.3122,
            -0.5907,
            -0.8111,
            -0.9521,
            -1.0,
            -0.9501,
            -0.8073,
            -0.5855,
            -0.3063,
        ),
        "lead_sine": (None,) * 63
        + (
            0.9872,
            0.8895,
            0.7049,
            0.4514,
            0.1537,
            -0.1591,
            -0.4565,
            -0.7093,
            -0.8925,
            -0.9882,
            -0.9871,
            -0.8894,
            -0.7047,
            -0.4512,
            -0.1536,
            0.1592,
            0.4565,
        ),
    },
)
