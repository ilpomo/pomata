"""Spec for ``pomata.indicators.sine_wave`` — Ehlers' sine / lead-sine struct, latching, scale-invariant."""

import math

import polars as pl
from tests.indicators.oracles import sine_wave_reference
from tests.support import spans_even_lag_run
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import sine_wave

# A clean 20-bar-period carrier: 80 bars leave 17 emitted values past the 63-bar settling warm-up (the old golden).
_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(80))


def _no_sustained_even_lag_run(frame: pl.DataFrame) -> bool:
    """
    Reject a SUSTAINED even-lag run — the regime where the phase branch that fixes both lines genuinely flips.
    Measured boundary (impl vs oracle, trailing flat / alternating runs on the golden carrier): real branch-flip
    disagreement starts probabilistically at ~9 structured bars and is the norm by 11-14 (a whole-series flat run
    deviates ~8.3e-2), while an ISOLATED even-lag tie — the bulk of what the old single-pair predicate rejected —
    stays within ~2.6e-10 on these unit-bounded lanes, inside the property tiers' absolute band. Only the finite
    bars can reach it, so filtering them keeps the missing-data tier from rejecting on interior null / NaN.
    """
    finite = [value for value in frame.to_series(0).to_list() if value is not None and math.isfinite(value)]
    return not spans_even_lag_run(finite)


SINE_WAVE = Spec(
    factory=sine_wave,
    inputs=("expr",),
    params={},
    shape=Shape.STRUCT,
    fields=("sine", "lead_sine"),
    warmup={"sine": 63, "lead_sine": 63},
    oracle=sine_wave_reference,
    conditioning=_no_sustained_even_lag_run,
    # Both lines are the sine of a phase, bounded in [-1, 1] and scale-INVARIANT, degree 0 (tests/indicators/
    # test_sine_wave.py::TestSineWaveProperties::test_scale_invariance).
    scale=(ScaleAxis(roles=("expr",), degree={"sine": 0, "lead_sine": 0}),),
    golden_input={"expr": _SAMPLE},
    golden_output={
        "sine": (None,) * 63
        + (
            0.8109,
            0.9521,
            1.0,
            0.9501,
            0.8074,
            0.5856,
            0.3063,
            -0.0031,
            -0.3122,
            -0.5907,
            -0.8111,
            -0.9521,
            -1.0,
            -0.9501,
            -0.8073,
            -0.5855,
            -0.3063,
        ),
        "lead_sine": (None,) * 63
        + (
            0.9872,
            0.8895,
            0.7049,
            0.4514,
            0.1537,
            -0.1591,
            -0.4565,
            -0.7093,
            -0.8925,
            -0.9882,
            -0.9871,
            -0.8894,
            -0.7047,
            -0.4512,
            -0.1536,
            0.1592,
            0.4565,
        ),
    },
    pins=(
        SpecPin(
            label="sustained_flat_run_settles_on_the_zero_fixed_point",
            inputs={"expr": (0.0,) * 80},
            expected={
                "sine": (None,) * 63 + (0.5,) * 17,
                "lead_sine": (None,) * 63 + (-0.2588,) * 17,
            },
            reason="the regime the conditioning filter excludes, witnessed at its one platform-stable point: on a "
            "NONZERO flat run the phase is a cancellation residual whose sign — and so the whole phasor branch — "
            "differs across OS math libraries (measured ~9e-2 from the oracle on one platform, other values on "
            "another), which is exactly the hazard the filter names; on the all-zero run the branch is pinned by "
            "IEEE (atan2(0, 0) = 0) and both lanes settle on constant-driven fixed points, exact at the declared "
            "rounding on every platform and inside the documented [-1, 1] bound",
            covers_conditioning=True,
            round_to=4,
        ),
    ),
)
