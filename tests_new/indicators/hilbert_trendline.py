"""Spec for ``pomata.indicators.hilbert_trendline`` — Ehlers' instantaneous trendline, latching, degree-1."""

import math

import polars as pl
from tests_new.indicators.oracles import hilbert_trendline_reference
from tests_new.support import spans_even_lag_repeat
from tests_new.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import hilbert_trendline

# A 0.5/bar trend plus a clean 20-bar cycle: 80 bars leave 17 emitted values past the 63-bar warm-up (the old golden).
_SAMPLE = tuple(100.0 + 0.5 * index + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(80))


def _no_even_lag_repeat(frame: pl.DataFrame) -> bool:
    """Exclude the shared cycle-pipeline degenerate: an even-lag repeat drives the phasor to a cancellation residual."""
    finite = [value for value in frame.to_series(0).to_list() if value is not None and math.isfinite(value)]
    return not spans_even_lag_repeat(finite)


HILBERT_TRENDLINE = Spec(
    factory=hilbert_trendline,
    inputs=("expr",),
    params={},
    shape=Shape.SERIES,
    warmup=63,
    oracle=hilbert_trendline_reference,
    conditioning=_no_even_lag_repeat,
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
)
