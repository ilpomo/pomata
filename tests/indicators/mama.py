"""Declaration for ``pomata.indicators.mama`` — the adaptive-alpha cycle pair, latching, golden-anchored."""

import math

import polars as pl

from pomata.indicators import mama
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_mama
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape
from tests.support.strategies import spans_even_lag_run

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
    reference='Ehlers, J. F. "MAMA — The Mother of Adaptive Moving Averages." MESA Software.',
    see_also=(
        ("hilbert_phasor", "The phasor whose phase rate sets the smoothing constant."),
        ("kama", "Another adaptive moving average, adapting on the efficiency ratio."),
        ("dominant_cycle_phase", "The dominant-cycle phase from the same pipeline."),
    ),
    notes=(
        (
            "Seeding",
            "Both lines are seeded at the price prefix — ``MAMA`` and ``FAMA`` start from the price "
            "and the recurrence runs from there. Ehlers' original presentation instead "
            "zero-initializes both lines, so the two report different values across the warm-up "
            "region before the exponential weighting washes the seed out; pomata's price seed is the "
            "saner choice for a price-level average. Port warm-up-sensitive logic accordingly.",
        ),
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
            "on a sustained even-lag run (a flat price or a period-two alternation) Ehlers' six-tap "
            "quadrature filter reads the four-bar smooth at even lags, so the in-phase component "
            "collapses to a cancellation residual and the phasor branch — and with it the adaptive "
            "smoothing constant — turns numerically arbitrary; there is no cycle to adapt to.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A struct ``pl.Expr`` with two ``Float64`` fields, the same length as ``expr``:"
    "\n\n"
    "- ``mama`` — the MESA adaptive moving average. - ``fama`` — the Following Adaptive "
    "Moving Average, the slower signal-line pass."
    "\n\n"
    "The first ``32`` rows are ``null`` (warm-up). Read one line with "
    '``.struct.field("mama")`` or split both with ``.struct.unnest()``.',
    raises_prose="ValueError: If ``limit_fast`` or ``limit_slow`` is outside ``(0, 1]`` — the smoothing "
    "constant is a weight, so a limit above ``1`` makes ``1 - alpha`` negative and the "
    "recurrence diverges — or if ``limit_fast < limit_slow``, which would pin the adaptive "
    "smoothing constant at ``limit_slow`` and make ``limit_fast`` a false upper bound.",
    args_prose={
        "limit_fast": "Upper bound on the smoothing constant (a fast cycle; canonical default ``0.5``). Must be "
        "in ``(0, 1]`` and ``>= limit_slow``.",
        "limit_slow": "Lower bound on the smoothing constant (a slow cycle; canonical default ``0.05``). Must "
        "be in ``(0, 1]``.",
    },
    example_imports=("import math",),
    intro_basic="Both adaptive lines track the level of a clean period-20 cycle (here ``100``), at the last bar:",
    examples=(
        Example(
            verbatim=(
                ">>> frame = pl.select(close=100.0 + (2 * math.pi * pl.int_range(200) / 20).sin())",
                '>>> lines = frame.select(mama=mama(pl.col("close"))).unnest("mama")',
                '>>> round(lines["mama"][-1], 2), round(lines["fama"][-1], 2)',
                "(99.67, 99.96)",
            )
        ),
    ),
)
