"""Spec for ``pomata.indicators.trend_mode`` — Ehlers' trend / cycle flag, latching, scale-invariant."""

import math

from tests_new.indicators.oracles import trend_mode_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import trend_mode

# A clean 20-bar-period carrier: 80 bars leave 17 emitted flags past the 63-bar settling warm-up (the old golden).
_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(80))

TREND_MODE = Spec(
    factory=trend_mode,
    inputs=("expr",),
    params={},
    shape=Shape.SERIES,
    warmup=63,
    oracle=trend_mode_reference,
    # A 0/1 trend-vs-cycle flag: scale-INVARIANT, degree 0 (tests/indicators/test_trend_mode.py
    # ::TestTrendModeProperties::test_scale_invariance).
    scale=(ScaleAxis(roles=("expr",), degree=0),),
    golden_input={"expr": _SAMPLE},
    golden_output=(None,) * 63 + (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    pins=(
        SpecPin(
            label="flat_run_flags_trend",
            inputs={"expr": (100.0,) * 80},
            expected=(None,) * 63 + (1.0,) * 17,
            reason="the cycle-pipeline degenerate (a flat run) the old suite filtered out of the property tiers: the "
            "trend/cycle vote needs ~317+ rows of sustained degeneracy before the flag ever flips against the "
            "oracle, far past the property tiers' frame cap (~91 rows), so impl and oracle agree on every reachable "
            "input (0 mismatches across the probed eps-ladder and length sweep) — no conditioning filter is "
            "declared and the corner stays witnessed by this fixed case instead",
        ),
    ),
)
