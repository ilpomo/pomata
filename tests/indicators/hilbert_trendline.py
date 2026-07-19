"""Declaration for ``pomata.indicators.hilbert_trendline`` — Ehlers' instantaneous trendline, latching, degree-1."""

import math

from pomata.indicators import hilbert_trendline
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_hilbert_trendline
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

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
    reference="Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*. Wiley.",
    see_also=(
        ("trend_mode", "Uses the price's deviation from this trendline."),
        ("dominant_cycle_period", "The cycle length this averages the price over."),
        ("mama", "The adaptive average from the same pipeline."),
    ),
    opener_override="The fixed FIR smoothing and quadrature stages are computed independently, but the "
    "adaptive dominant-cycle period feeds back into its own measurement and the stages built "
    "on it, so the reference oracle replays Ehlers' pipeline and confirms its internal "
    "consistency rather than independence; the independent witness is the set of frozen "
    "golden masters, plus TA-Lib parity on the converged tail (the differential tier compares "
    "the whole cycle cluster — every HT_* counterpart plus MAMA — against the C reference). "
    "Where measurable the oracle agrees to ten significant figures (a ``1e-10`` band) on any "
    "finite input within a sane dynamic range — a flat or period-two (even-lag) series "
    "included, though there the reading itself is physically meaningless (the Hilbert "
    "quadrature is a pure cancellation residual: there is no cycle to measure). The "
    "documentation's *Correctness* page gives the method and the float-conditioning limit "
    "beyond it.",
    bullets=(
        ("Null", "a ``null`` price latches ``null`` for every row from there."),
        ("NaN", "a ``NaN`` price latches ``null`` for every row from there, as any non-finite value does."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The instantaneous trendline for each row, the same length as ``expr``, on the price "
    "scale. The first ``63`` rows are ``null`` (warm-up).",
    example_imports=("import math",),
    intro_basic="Spanning a whole cycle, the trendline cancels the swing and tracks the mean level (here ``100``):",
    examples=(
        Example(
            verbatim=(
                ">>> frame = pl.select(close=100.0 + (2 * math.pi * pl.int_range(200) / 20).sin())",
                ">>> "
                'round(frame.select(hilbert_trendline=hilbert_trendline(pl.col("close")))["hilbert_trendline"][-1], '
                "2)",
                "100.0",
            )
        ),
    ),
)
