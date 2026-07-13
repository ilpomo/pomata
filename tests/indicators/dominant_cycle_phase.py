"""Spec for ``pomata.indicators.dominant_cycle_phase`` — Ehlers' Hilbert dominant-cycle phase, latching, invariant."""

import math

import polars as pl
from tests.indicators.oracles import dominant_cycle_phase_reference
from tests.support import spans_even_lag_run
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import dominant_cycle_phase

# A clean 20-bar-period carrier: 80 bars leave 17 emitted values past the 63-bar settling warm-up (the old golden).
_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(80))


def _no_sustained_even_lag_run(frame: pl.DataFrame) -> bool:
    """
    Reject a SUSTAINED even-lag run — the regime where the phase branch genuinely flips a residual apart. Measured
    boundary (impl vs oracle, hand-graduated flat and alternation families): an exact flat run deviates ~3.6e-2 and
    an exact alternation turns dangerous only once its relative amplitude shrinks below ~1e-2 — both are full-length
    even-lag runs, rejected here — while an ISOLATED even-lag tie inside a well-spread series (the bulk of what the
    old single-pair predicate rejected, ~27% of all draws) never crosses the property band (worst ~4.5e-14). The
    near-degenerate band the run predicate cannot see (a NEAR-flat drift with relative spread below ~4e-6, or a NEAR
    alternation at tiny amplitude, deviating past 1e-10 without any exact tie) is unreachable by the property tiers'
    independent per-element draws, whose relative spread is O(1) by construction. Only the finite bars can reach the
    degenerate, so filtering them keeps the missing-data tier from rejecting on interior null / NaN.
    """
    finite = [value for value in frame.to_series(0).to_list() if value is not None and math.isfinite(value)]
    return not spans_even_lag_run(finite)


DOMINANT_CYCLE_PHASE = Spec(
    factory=dominant_cycle_phase,
    inputs=("expr",),
    params={},
    shape=Shape.SERIES,
    warmup=63,
    oracle=dominant_cycle_phase_reference,
    conditioning=_no_sustained_even_lag_run,
    # A phase in degrees: scale-INVARIANT, degree 0 (tests/indicators/test_dominant_cycle_phase.py
    # ::TestDominantCyclePhaseProperties::test_scale_invariance).
    scale=(ScaleAxis(roles=("expr",), degree=0),),
    golden_input={"expr": _SAMPLE},
    golden_output=(None,) * 63
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
    pins=(
        SpecPin(
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
)
