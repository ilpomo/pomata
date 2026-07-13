"""Spec for ``pomata.indicators.sine_wave`` — Ehlers' sine / lead-sine struct, latching, scale-invariant."""

import math

import polars as pl
from tests_new.indicators.oracles import sine_wave_reference
from tests_new.support import spans_even_lag_run
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

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
            label="sustained_flat_run_diverges_from_reference",
            inputs={"expr": (100.0,) * 80},
            expected={
                "sine": (None,) * 63
                + (
                    -0.9098528522762199,
                    -0.9098466913108698,
                    -0.9098416924932389,
                    -0.9098376373787095,
                    -0.9098343484232236,
                    -0.9098316813577921,
                    -0.909829518978248,
                    -0.9098277660887951,
                    -0.9098263453860005,
                    -0.9098251941091461,
                    -0.909824261314937,
                    -0.909823505660742,
                    -0.9098228936019193,
                    -0.9098223979262363,
                    -0.9098219965626348,
                    -0.9098216716132133,
                    -0.9098214085667864,
                ),
                "lead_sine": (None,) * 63
                + (
                    -0.936763690237157,
                    -0.9367688863514293,
                    -0.9367731020425325,
                    -0.9367765216916077,
                    -0.9367792951247279,
                    -0.936781544066418,
                    -0.936783367391654,
                    -0.9367848453996032,
                    -0.9367860432908455,
                    -0.9367870139960882,
                    -0.9367878004769019,
                    -0.936788437596679,
                    -0.9367889536417882,
                    -0.9367893715580732,
                    -0.936789709955756,
                    -0.9367899839259475,
                    -0.9367902057039424,
                ),
            },
            reason="the regime the conditioning filter excludes, witnessed once: on a whole-series flat run the "
            "impl's phase branch flips against the naive oracle (impl sine ~-0.9099 vs oracle ~-0.8346, an intrinsic "
            "~9e-2 transcription divergence, not a bug), so the lanes are pinned to the implementation's "
            "deterministic output rather than the oracle; both stay inside the documented [-1, 1] bound",
            covers_conditioning=True,
        ),
    ),
)
