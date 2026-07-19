"""
Declaration for ``pomata.indicators.dominant_cycle_period`` — Ehlers' Hilbert dominant-cycle length, latching,
invariant.
"""

import math

from pomata.indicators import dominant_cycle_period
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_dominant_cycle_period
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(40))

DOMINANT_CYCLE_PERIOD = suite_indicators(
    factory=dominant_cycle_period,
    inputs=("expr",),
    params={},
    null=BehaviorNull.LATCHES,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=32,
    oracle=reference_dominant_cycle_period,
    scaling=(ScaleAxis(roles=("expr",), degree=0),),
    talib=RelationTalib.MATCHES,
    golden=Golden(
        inputs={"expr": _SAMPLE},
        output=(None,) * 32 + (19.0186, 19.3994, 19.7391, 20.051, 20.3271, 20.5471, 20.6936, 20.7611),
    ),
    pins=(
        Pin(
            label="reference_flat_run_matches",
            inputs={"expr": (100.0,) * 48},
            expected=(None,) * 32
            + (
                24.7067,
                25.7598,
                26.659,
                27.4199,
                28.0591,
                28.5931,
                29.0372,
                29.4052,
                29.7092,
                29.9599,
                30.166,
                30.3353,
                30.474,
                30.5877,
                30.6807,
                30.7567,
            ),
            reason="the cycle-pipeline degenerate (a flat run): the period clamp and the AND-gated phase update damp "
            "the phasor residual before it reaches this output, so impl and oracle agree there (measured worst "
            "relative deviation ~3.4e-15) and no conditioning filter is declared — the corner stays witnessed by "
            "this fixed case, rounded because the degenerate pipeline settles on libm-dependent fixed points that "
            "differ across OS math libraries",
            round_to=4,
        ),
    ),
    reference="Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*. Wiley.",
    see_also=(
        ("dominant_cycle_phase", "The phase of the same dominant cycle."),
        ("hilbert_phasor", "The phasor the period is measured from."),
        ("hilbert_trendline", "Averages the price over one cycle of this period."),
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
    returns_body="The dominant-cycle period for each row, the same length as ``expr``, settling into ``[6, "
    "50]`` (the raw estimate is clamped there before the reported double-smoothing, so every "
    "emitted row already lies inside the band; the earliest rows start near its middle while "
    "the smoothers converge). The first ``32`` rows are ``null`` (the warm-up the recursive "
    "smoothers need to settle).",
    example_imports=("import math",),
    intro_basic="The dominant cycle of a clean period-20 sine, read at the last bar (close to its true "
    "length of ``20`` bars):",
    examples=(
        Example(
            verbatim=(
                ">>> frame = pl.select(close=100.0 + (2 * math.pi * pl.int_range(200) / 20).sin())",
                '>>> expr = dominant_cycle_period(pl.col("close"))',
                '>>> round(frame.select(dominant_cycle_period=expr)["dominant_cycle_period"][-1], 2)',
                "20.03",
            )
        ),
    ),
)
