"""Declaration for ``pomata.indicators.mama`` — the adaptive-alpha cycle pair, latching, golden-anchored."""

import math

import polars as pl

from pomata.indicators import mama
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_mama
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape
from tests_new.support.strategies import spans_even_lag_run

# A clean 20-bar-period carrier: 40 bars leave 8 emitted values past the 32-bar settling warm-up.
_SAMPLE = tuple(100.0 + 10.0 * math.sin(2 * math.pi * index / 20) for index in range(40))

# The degenerate core the conditioning filter excludes: a whole-series period-two alternation (see the pin below).
_ALTERNATION = tuple(100.0 if index % 2 == 0 else 105.0 for index in range(40))


def _no_sustained_even_lag_run(frame: pl.DataFrame) -> bool:
    """
    Reject a SUSTAINED even-lag run: Ehlers' six-tap quadrature filter reads the four-bar smooth at even lags, so a
    sustained flat run or period-two alternation drives the in-phase component to a cancellation residual the impl's
    explicit FIR and the oracle's compensated sum round apart, flipping the phasor branch. Measured boundary (impl vs
    oracle on the golden carrier): a trailing alternation agrees exactly through 13 bars and jumps to a 1.23e-2
    relative deviation at 14, while an ISOLATED even-lag tie never deviates past pure FP noise (~1e-14), so only the
    sustained run is excluded. Only the finite bars can reach it, so filtering them keeps the missing-data tier from
    rejecting on interior null / NaN.
    """
    finite = [value for value in frame.to_series(0).to_list() if value is not None and math.isfinite(value)]
    return not spans_even_lag_run(finite)


MAMA = suite_indicators(
    factory=mama,
    inputs=("expr",),
    params={"limit_fast": 0.5, "limit_slow": 0.05},
    null=BehaviorNull.LATCHES,
    nan=BehaviorNan.LATCHES,
    shape=Shape.STRUCT,
    fields=("mama", "fama"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"mama": 32, "fama": 32},
    oracle=reference_mama,
    scaling=(ScaleAxis(roles=("expr",), degree={"mama": 1, "fama": 1}),),
    talib=RelationTalib.MATCHES,
    conditioning=_no_sustained_even_lag_run,
    raises=(
        ({"limit_fast": 0.0}, r"limit_fast must be in"),
        ({"limit_slow": 0.0}, r"limit_slow must be in"),
        ({"limit_fast": 3.0}, r"limit_fast must be in"),
        ({"limit_slow": 2.5}, r"limit_slow must be in"),
        ({"limit_fast": 0.05, "limit_slow": 0.5}, r"limit_fast must be >= limit_slow"),
    ),
    golden=Golden(
        inputs={"expr": _SAMPLE},
        output={
            "mama": (None,) * 32 + (97.9767, 97.6734, 97.3142, 96.9485, 96.6255, 96.3897, 96.2764, 96.308),
            "fama": (None,) * 32 + (99.6954, 99.6448, 99.5866, 99.5206, 99.4482, 99.3718, 99.2944, 99.2197),
        },
    ),
    pins=(
        Pin(
            label="sustained_alternation_diverges_from_reference",
            inputs={"expr": _ALTERNATION},
            expected={
                "mama": (None,) * 32 + (100.2393, 100.4773, 100.2387, 100.4767, 100.2384, 100.5784, 100.2892, 100.7946),
                "fama": (None,) * 32 + (100.4153, 100.4168, 100.3723, 100.3749, 100.3408, 100.3493, 100.3342, 100.3589),
            },
            reason="the regime the conditioning filter excludes, witnessed once: on a whole-series period-two "
            "alternation the impl's phasor branch flips against the naive oracle (impl mama ~100.79 vs oracle "
            "~103.33 at the last row, an intrinsic ~2.5e-2 transcription divergence, not a bug), so the lanes are "
            "pinned to the implementation's output rather than the oracle — rounded, because the degenerate "
            "pipeline settles on libm-dependent fixed points that differ across OS math libraries",
            covers_conditioning=True,
            round_to=4,
        ),
    ),
)
