"""Declaration for ``pomata.indicators.trend_mode`` — Ehlers' trend / cycle flag, latching, scale-invariant."""

import math

from pomata.indicators import trend_mode
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_trend_mode
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(80))

TREND_MODE = suite_indicators(
    factory=trend_mode,
    inputs=("expr",),
    params={},
    null=BehaviorNull.LATCHES,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=63,
    oracle=reference_trend_mode,
    scaling=(ScaleAxis(roles=("expr",), degree=0),),
    talib=RelationTalib.MATCHES,
    golden=Golden(
        inputs={"expr": _SAMPLE},
        output=(None,) * 63 + (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ),
    pins=(
        Pin(
            label="flat_run_flags_trend",
            inputs={"expr": (100.0,) * 80},
            expected=(None,) * 63 + (1.0,) * 17,
            reason="the cycle-pipeline degenerate (a flat run): the trend/cycle vote needs ~317+ rows of sustained "
            "degeneracy before the flag ever flips against the oracle, far past the property tiers' frame cap "
            "(~91 rows), so impl and oracle agree on every reachable input (0 mismatches across the probed "
            "eps-ladder and length sweep) — no conditioning filter is declared and the corner stays witnessed by "
            "this fixed case instead",
        ),
    ),
)
