"""Declaration for ``pomata.indicators.trend_mode`` — Ehlers' trend / cycle flag, latching, scale-invariant."""

import math

from pomata.indicators import trend_mode
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_trend_mode
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

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
    reference="Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*. Wiley.",
    see_also=(
        ("hilbert_trendline", "The trendline the mode compares the price against."),
        ("sine_wave", "The sine-wave crossings the mode combines."),
        ("dominant_cycle_phase", "The phase rate the mode also uses."),
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
    note_postscript="The underlying phase branch guards an *exact* zero of the cosine projection (saturating to "
    "``±90`` as that projection vanishes), rather than the inventor's fixed ``0.001`` absolute cutoff; this is the "
    "continuous limit and keeps the phase invariant under a lossless rescale of the price, whereas a fixed threshold "
    "would be scale-dependent.",
    returns_body="The mode flag (``1.0`` trend / ``0.0`` cycle) for each row, the same length as ``expr``. "
    "The first ``63`` rows are ``null`` (warm-up). The days-in-trend count the flag "
    "thresholds has no canonical seed, so for a few dozen rows past the warm-up the flag can "
    "differ from a differently-seeded implementation before that state converges; the TA-Lib "
    "parity below holds on the converged tail.",
    example_imports=("import math",),
    intro_basic="A small pure cycle -- its swing below ~1.5% of price -- stays in cycle mode, so the flag "
    "stays ``0`` over a clean low-amplitude period-20 sine:",
)
