"""
Declaration for ``pomata.indicators.dominant_cycle_phase`` — Ehlers' Hilbert dominant-cycle phase, latching,
invariant.
"""

import math

import polars as pl

from pomata.indicators import dominant_cycle_phase
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_dominant_cycle_phase
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape
from tests.support.strategies import spans_even_lag_run

_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(80))


def _no_sustained_even_lag_run(frame: pl.DataFrame) -> bool:
    """
    Reject a SUSTAINED even-lag run — the regime where the phase branch genuinely flips a residual apart. Measured
    boundary (impl vs oracle, hand-graduated flat and alternation families): an exact flat run deviates ~3.6e-2 and
    an exact alternation turns dangerous only once its relative amplitude shrinks below ~1e-2 — both are full-length
    even-lag runs, rejected here — while an ISOLATED even-lag tie inside a well-spread series never crosses the
    property band (worst ~4.5e-14). The near-degenerate band the run predicate cannot see (a NEAR-flat drift with
    relative spread below ~4e-6, or a NEAR alternation at tiny amplitude, deviating past 1e-10 without any exact
    tie) is unreachable by the property tiers' independent per-element draws, whose relative spread is O(1) by
    construction. Only the finite bars can reach the degenerate, so filtering them keeps the missing-data tier from
    rejecting on interior null / NaN.
    """
    finite = [value for value in frame.to_series(0).to_list() if value is not None and math.isfinite(value)]
    return not spans_even_lag_run(finite)


DOMINANT_CYCLE_PHASE = suite_indicators(
    factory=dominant_cycle_phase,
    inputs=("expr",),
    params={},
    null=BehaviorNull.LATCHES,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=63,
    oracle=reference_dominant_cycle_phase,
    scaling=(ScaleAxis(roles=("expr",), degree=0),),
    talib=RelationTalib.MATCHES,
    conditioning=_no_sustained_even_lag_run,
    golden=Golden(
        inputs={"expr": _SAMPLE},
        output=(None,) * 63
        + (
            54.1853,
            72.1855,
            90.1782,
            108.1678,
            126.1594,
            144.1573,
            162.1633,
            180.1763,
            198.1917,
            216.204,
            234.2083,
            252.2035,
            270.1915,
            288.177,
            306.1651,
            -35.84,
            -17.8363,
        ),
    ),
    pins=(
        Pin(
            label="flat_run_settles_on_the_zero_fixed_point",
            inputs={"expr": (0.0,) * 80},
            expected=(None,) * 63
            + (
                150.0003,
                150.0002,
                150.0002,
                150.0001,
                150.0001,
                150.0001,
                150.0001,
                150.0001,
                150.0,
                150.0,
                150.0,
                150.0,
                150.0,
                150.0,
                150.0,
                150.0,
                150.0,
            ),
            reason="the regime the conditioning filter excludes, witnessed at its one platform-stable point: on a "
            "NONZERO flat run the phase is a cancellation residual whose sign — and so the whole atan branch — "
            "differs across OS math libraries (measured ~3.8e-2 from the oracle on one platform, other values on "
            "another), which is exactly the hazard the filter names; on the all-zero run the branch is pinned by "
            "IEEE (atan2(0, 0) = 0) and the lanes settle on constant-driven fixed points, exact at the declared "
            "rounding on every platform",
            covers_conditioning=True,
            round_to=4,
        ),
    ),
    reference="Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*. Wiley.",
    see_also=(
        ("dominant_cycle_period", "The length of the same dominant cycle."),
        ("sine_wave", "The sine of this phase."),
        ("mama", "The adaptive average driven by the same pipeline's phasor-phase rate."),
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
            "Stability",
            "on a constant (flat) price, or any sustained even-lag run, the discrete transform's "
            "projections are pure cancellation residuals, so the phase is numerically arbitrary — "
            "there is no cycle to measure. The phase branch guards an *exact* zero of the cosine "
            "projection (saturating to ``±90`` as that projection vanishes), rather than the "
            "inventor's fixed ``0.001`` absolute cutoff; this is the continuous limit and keeps the "
            "phase invariant under a lossless rescale of the price, whereas a fixed threshold would "
            "be scale-dependent.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The dominant-cycle phase in degrees for each row, the same length as ``expr``. The first "
    "``63`` rows are ``null`` (the warm-up: the smoothers' settling plus the dominant-cycle "
    "look-back).",
    example_imports=("import math",),
    intro_basic="The dominant-cycle phase of a clean period-20 sine, read at the last bar (in degrees): "
    ">>> import math",
)
