"""
Declaration for ``pomata.indicators.hilbert_phasor`` — Ehlers' in-phase/quadrature phasor struct, latching, degree-1.
"""

import math

from pomata.indicators import hilbert_phasor
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_hilbert_phasor
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(40))

HILBERT_PHASOR = suite_indicators(
    factory=hilbert_phasor,
    inputs=("expr",),
    params={},
    null=BehaviorNull.LATCHES,
    nan=BehaviorNan.LATCHES,
    shape=Shape.STRUCT,
    fields=("in_phase", "quadrature"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"in_phase": 32, "quadrature": 32},
    oracle=reference_hilbert_phasor,
    scaling=(ScaleAxis(roles=("expr",), degree={"in_phase": 1, "quadrature": 1}),),
    talib=RelationTalib.MATCHES,
    golden=Golden(
        inputs={"expr": _SAMPLE},
        output={
            "in_phase": (None,) * 32 + (-0.0296, -2.9751, -5.7341, -7.9735, -9.445, -10.0056, -9.5949, -8.227),
            "quadrature": (None,) * 32 + (-9.5596, -9.5537, -8.4419, -6.3428, -3.5083, -0.2378, 3.1502, 6.3025),
        },
    ),
    pins=(
        Pin(
            label="flat_run_collapses_to_cancellation_residual",
            inputs={"expr": (100.0,) * 40},
            expected={
                "in_phase": (None,) * 32 + (0.0,) * 8,
                "quadrature": (None,) * 32 + (0.0,) * 8,
            },
            reason="the cycle-pipeline degenerate (a flat run): this struct exports the raw FIR components, so both "
            "collapse to the ~1e-14 cancellation residual itself, which the property tiers' absolute band absorbs "
            "(measured worst deviation ~3.4e-13) — no conditioning filter is declared and the corner stays witnessed "
            "by this fixed case instead: at the declared rounding the residual reads as an exact 0.0 on every "
            "platform, which IS the fact worth pinning",
            round_to=4,
        ),
    ),
    reference="Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*. Wiley.",
    see_also=(
        ("dominant_cycle_period", "Measured from this phasor by the homodyne discriminator."),
        ("mama", "Adapts on the rate of change of this phasor's phase."),
        ("dominant_cycle_phase", "The companion dominant-cycle phase."),
    ),
    opener_override="The fixed FIR smoothing and quadrature stages are computed independently, but the "
    "adaptive dominant-cycle period feeds back into its own measurement and the stages built "
    "on it, so the reference oracle replays Ehlers' pipeline and confirms its internal "
    "consistency rather than independence; the independent witness is the set of frozen "
    "golden masters, plus TA-Lib parity on the converged tail (the differential tier compares "
    "the whole cycle cluster — every HT_* counterpart plus MAMA — against the C reference). "
    "Where measurable the oracle agrees to ten significant figures (a ``1e-10`` band) on any "
    "finite input within a sane dynamic range, except on a flat or period-two (even-lag) "
    "series, where the Hilbert quadrature is a pure cancellation residual and the measurement "
    "is ill-conditioned (there is no cycle to measure). The documentation's *Correctness* "
    "page gives the method and the float-conditioning limit beyond it.",
    bullets=(
        ("Null", "a ``null`` price latches ``null`` for every row from there."),
        ("NaN", "a ``NaN`` price latches ``null`` for every row from there, as any non-finite value does."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A struct ``pl.Expr`` with two ``Float64`` fields, the same length as ``expr``:"
    "\n\n"
    "- ``in_phase`` — the in-phase (real) component of the phasor. - ``quadrature`` — the "
    "quadrature (imaginary) component of the phasor."
    "\n\n"
    "The first ``32`` rows are ``null`` (warm-up). Read one line with "
    '``.struct.field("in_phase")`` or split both with ``.struct.unnest()``.',
    example_imports=("import math",),
    intro_basic="The in-phase and quadrature components on a clean period-20 sine, at the last bar:",
    examples=(
        Example(
            verbatim=(
                ">>> frame = pl.select(close=100.0 + (2 * math.pi * pl.int_range(200) / 20).sin())",
                '>>> phasor = frame.select(hilbert_phasor=hilbert_phasor(pl.col("close"))).unnest("hilbert_phasor")',
                '>>> round(phasor["in_phase"][-1], 2), round(phasor["quadrature"][-1], 2)',
                "(-0.8, 0.61)",
            )
        ),
    ),
)
