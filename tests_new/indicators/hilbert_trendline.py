"""Declaration for ``pomata.indicators.hilbert_trendline`` — Ehlers' instantaneous trendline, latching, degree-1."""

import math

from pomata.indicators import hilbert_trendline
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_hilbert_trendline
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

_SAMPLE = tuple(100.0 + 0.5 * index + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(80))

HILBERT_TRENDLINE = suite_indicators(
    factory=hilbert_trendline,
    inputs=("expr",),
    params={},
    null=BehaviorNull.LATCHES,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=63,
    oracle=reference_hilbert_trendline,
    scaling=(ScaleAxis(roles=("expr",), degree=1),),
    talib=RelationTalib.MATCHES,
    golden=Golden(
        inputs={"expr": _SAMPLE},
        output=(None,) * 63
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
    ),
    pins=(
        Pin(
            label="flat_run_is_the_constant",
            inputs={"expr": (100.0,) * 80},
            expected=(None,) * 63 + (100.0,) * 17,
            reason="the cycle-pipeline degenerate (a flat run): the trendline reads a downstream EWMA of the price, "
            "so the phasor's cancellation residual never reaches this output — impl and oracle are bit-identical "
            "here (measured deviation exactly 0.0) — no conditioning filter is declared and the corner stays "
            "witnessed by this fixed case",
        ),
    ),
)
