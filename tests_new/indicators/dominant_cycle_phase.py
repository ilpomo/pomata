"""Spec for ``pomata.indicators.dominant_cycle_phase`` — Ehlers' Hilbert dominant-cycle phase, latching, invariant."""

import math

import polars as pl
from tests_new.indicators.oracles import dominant_cycle_phase_reference
from tests_new.support import spans_even_lag_run
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

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
            label="flat_run_diverges_from_reference",
            inputs={"expr": (100.0,) * 80},
            expected=(None,) * 63
            + (
                245.48502471988814,
                245.4841739964856,
                245.48348376605713,
                245.4829238543553,
                245.4824697392075,
                245.48210149592745,
                245.48180293873128,
                245.48156092185275,
                245.4813647707648,
                245.4812058193823,
                245.4810770335804,
                245.48097270499812,
                245.48088820206146,
                245.48081976757982,
                245.4807643542402,
                245.48071949093415,
                245.4806831741622,
            ),
            reason="the regime the conditioning filter excludes, witnessed once: on a whole-series flat run the "
            "impl's phase branch settles a residual apart from the naive oracle (impl ~245.49 vs oracle ~236.58 "
            "degrees, an intrinsic ~3.8e-2 transcription divergence, not a bug), so the lane is pinned to the "
            "implementation's deterministic output rather than the oracle",
            covers_conditioning=True,
        ),
    ),
)
