"""
The indicators family dialect: the closed vocabularies an indicator declaration answers with.

The richest of the three families. Every axis here is a finite, closed set — a contributor picks a member, never
invents one — so a declaration cannot drift into free-form prose and the generated failure messages read the same
label the declaration stated. Beyond the shared null / NaN / shape / scaling axes, the indicators family adds how a
windowed or recursive output warms up, how its recursion seeds (Phase-B metadata, see :class:`Seeding`), and its
relation to the TA-Lib reference (:class:`RelationTalib`), which drives the differential tier and absorbs the old
``talib_coverage`` registry.
"""

import enum


class BehaviorNull(enum.Enum):
    """What an interior ``null`` does to an indicator output (the indicators dialect of ``NullPolicy``)."""

    BRIDGED = "bridged"  # a recursion steps over it (state carries), so later rows recover
    IN_WINDOW_IS_NULL = "in_window_is_null"  # nulls every window that overlaps it, then recovers
    PROPAGATES = (
        "propagates"  # nulls the rows it reaches (its own, a one-bar lag, or a contracting recursion), recovers
    )
    LATCHES = "latches"  # a cycle recursion carries the gap forward forever (every later row stays missing)
    ABSORBED = "absorbed"  # a null candidate is dropped from the pointwise computation; no output row is nulled for it


class BehaviorNan(enum.Enum):
    """What an interior ``NaN`` does to an indicator output (the indicators dialect of ``pomata._policy.NanPolicy``)."""

    PROPAGATES = "propagates"  # nans the rows it reaches, then recovers (a pointwise or fixed-lag map)
    LATCHES = "latches"  # a recursion carries it forward forever (every later row stays missing — a NaN or a null)


class Warmup(enum.Enum):
    """The warm-up an indicator owes; the harness resolves it to a leading-null ``int`` / mapping / ``None``."""

    NONE = "none"  # no warm-up: a windowless pointwise transform is defined at the first row
    WINDOW_MINUS_ONE = "window_minus_one"  # a single rolling window emits its first value at row ``window - 1``
    WINDOW = "window"  # a single rolling window whose first value lands one row later (a differenced or lagged window)
    EXPR = "expr"  # a composite warm-up (stacked smoothers, a Wilder seed) stated as an explicit ``int``
    PER_FIELD = "per_field"  # a struct's warm-up, stated as an explicit per-field mapping


class Seeding(enum.Enum):
    """
    How a recursion seeds its first state — Phase-B metadata for the generated docstring, NOT load-bearing yet.

    No rung reads this axis today; it is recorded so the cutover's docstring generator can state each recursion's
    initialization in prose. Its correctness is not proven by the suite, so a declaration sets it only where the
    seeding is documented (the EWM means) and leaves ``NONE`` elsewhere rather than guessing.
    """

    SMA_SEED = "sma_seed"  # seeded by the simple average of the first window (the canonical EMA start)
    FIRST_VALUE = "first_value"  # seeded by the first observation
    RMA_SEED = "rma_seed"  # Wilder's smoothing seed (the first window's simple average, decayed at ``1 / window``)
    NONE = "none"  # no recursion to seed, or the seeding is not recorded


class RelationTalib(enum.Enum):
    """
    An indicator's relation to its TA-Lib twin — the partition the differential tier reads, absorbing the old
    ``talib_coverage`` registry.

    ``MATCHES`` is compared against TA-Lib bar for bar; ``DOCUMENTED_DIVERGENCE`` and ``NO_EQUIVALENT`` are accounted
    for but not compared, each carrying its reason string on the declaration (``talib_reason``).
    """

    MATCHES = "matches"  # a TA-Lib twin the implementation reproduces (at the reference band, whole series or tail)
    DOCUMENTED_DIVERGENCE = "documented_divergence"  # a TA-Lib twin the implementation deliberately diverges from
    NO_EQUIVALENT = "no_equivalent"  # no TA-Lib twin exists to compare against
