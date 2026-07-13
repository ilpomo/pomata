"""Spec for ``pomata.indicators.mama`` — the adaptive-alpha cycle pair, latching, golden-anchored, oracle-adapted."""

import math

import polars as pl
from tests_new.indicators.oracles import mama_reference
from tests_new.support import spans_even_lag_run
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import mama

# A clean 20-bar-period carrier: 40 bars leave 8 emitted values past the 32-bar settling warm-up.
_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(40))

# The degenerate core the conditioning filter excludes: a whole-series period-two alternation (see the pin below).
_ALTERNATION = tuple(100.0 if index % 2 == 0 else 105.0 for index in range(40))


def _mama_oracle(spec: Spec, frame: pl.DataFrame) -> object:
    """Reach the oracle by position only: its kwargs (``fast_limit`` / ``slow_limit``) are not the factory's mirror."""
    return spec.oracle(frame[spec.inputs[0]].to_list())


def _no_sustained_even_lag_run(frame: pl.DataFrame) -> bool:
    """
    Reject a SUSTAINED even-lag run: Ehlers' six-tap quadrature filter reads the four-bar smooth at even lags, so a
    sustained flat run or period-two alternation drives the in-phase component to a cancellation residual the impl's
    explicit FIR and the oracle's compensated sum round apart, flipping the phasor branch. Measured boundary (impl vs
    oracle on the golden carrier): a trailing alternation agrees exactly through 13 bars and jumps to a 1.23e-2
    relative deviation at 14, while an ISOLATED even-lag tie — the bulk of what the old single-pair predicate
    rejected — never deviates past pure FP noise (~1e-14), so only the sustained run is excluded. Only the finite bars
    can reach it, so filtering them keeps the missing-data tier from rejecting on interior null / NaN.
    """
    finite = [value for value in frame.to_series(0).to_list() if value is not None and math.isfinite(value)]
    return not spans_even_lag_run(finite)


MAMA = Spec(
    factory=mama,
    inputs=("expr",),
    params={"limit_fast": 0.5, "limit_slow": 0.05},
    shape=Shape.STRUCT,
    fields=("mama", "fama"),
    warmup={"mama": 32, "fama": 32},
    raises=(
        ({"limit_fast": 0.0}, r"limit_fast must be in"),
        ({"limit_slow": 0.0}, r"limit_slow must be in"),
        ({"limit_fast": 3.0}, r"limit_fast must be in"),
        ({"limit_slow": 2.5}, r"limit_slow must be in"),
        ({"limit_fast": 0.05, "limit_slow": 0.5}, r"limit_fast must be >= limit_slow"),
    ),
    oracle=mama_reference,
    oracle_adapter=_mama_oracle,
    conditioning=_no_sustained_even_lag_run,
    # Both lines ride the price scale, so they scale linearly with the bars (tests/indicators/test_mama.py:312
    # test_scale_homogeneity).
    scale=(ScaleAxis(roles=("expr",), degree={"mama": 1, "fama": 1}),),
    golden_input={"expr": _SAMPLE},
    golden_output={
        "mama": (None,) * 32 + (97.9767, 97.6734, 97.3142, 96.9485, 96.6255, 96.3897, 96.2764, 96.308),
        "fama": (None,) * 32 + (99.6954, 99.6448, 99.5866, 99.5206, 99.4482, 99.3718, 99.2944, 99.2197),
    },
    pins=(
        SpecPin(
            label="sustained_alternation_diverges_from_reference",
            inputs={"expr": _ALTERNATION},
            expected={
                "mama": (None,) * 32
                + (
                    100.23930838404078,
                    100.47734296483874,
                    100.23867148241936,
                    100.47673790829839,
                    100.2383689541492,
                    100.57840016802486,
                    100.28920008401244,
                    100.79457866237591,
                ),
                "fama": (None,) * 32
                + (
                    100.41529482367099,
                    100.4168460272002,
                    100.37230239100498,
                    100.37491327893731,
                    100.3407771977403,
                    100.34926160426534,
                    100.3342462242021,
                    100.3589386498334,
                ),
            },
            reason="the regime the conditioning filter excludes, witnessed once: on a whole-series period-two "
            "alternation the impl's phasor branch flips against the naive oracle (impl mama ~100.79 vs oracle "
            "~103.33 at the last row, an intrinsic ~2.5e-2 transcription divergence, not a bug), so the lanes are "
            "pinned to the implementation's deterministic output rather than the oracle",
            covers_conditioning=True,
        ),
    ),
)
