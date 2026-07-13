"""Spec for ``pomata.indicators.trend_mode`` — Ehlers' trend / cycle flag, latching, scale-invariant."""

import math

import polars as pl
from tests_new.indicators.oracles import trend_mode_reference
from tests_new.support import spans_even_lag_repeat
from tests_new.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import trend_mode

# A clean 20-bar-period carrier: 80 bars leave 17 emitted flags past the 63-bar settling warm-up (the old golden).
_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(80))


def _no_even_lag_repeat(frame: pl.DataFrame) -> bool:
    """Exclude the shared cycle-pipeline degenerate: an even-lag repeat flips the phase and so the emitted flag."""
    finite = [value for value in frame.to_series(0).to_list() if value is not None and math.isfinite(value)]
    return not spans_even_lag_repeat(finite)


TREND_MODE = Spec(
    factory=trend_mode,
    inputs=("expr",),
    params={},
    shape=Shape.SERIES,
    warmup=63,
    oracle=trend_mode_reference,
    conditioning=_no_even_lag_repeat,
    # A 0/1 trend-vs-cycle flag: scale-INVARIANT, degree 0 (tests/indicators/test_trend_mode.py
    # ::TestTrendModeProperties::test_scale_invariance).
    scale=(ScaleAxis(roles=("expr",), degree=0),),
    golden_input={"expr": _SAMPLE},
    golden_output=(None,) * 63 + (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
)
