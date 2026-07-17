"""
The canonical edge-case taxonomy: which Note bullets a function's declarations demand, and their canonical wording.

Every public docstring closes its Note with an ``**Edge-case behavior:**`` list. This module is the single home of
that list's contract: the class vocabulary (one label per concept, fixed order), the presence rules (a class appears
exactly when the machine-readable declarations activate it — the policy registry, the spec's pins, its conditioning
filter, its shape), and the canonical wording each class carries. The ``Null`` and ``NaN`` bullets have a full
sentence template per policy value (``BULLET_CANON``): a byte-identical skeleton with one free slot for the
function's own trigger, plus an optional trailing clause. A structural class freezes its terminal outcome clause
byte-for-byte (``OUTCOME_CANON``), and every degenerate outcome a Degenerate-denominator bullet asserts must be
witnessed by a pin (``asserted_outcome_kinds`` vs. ``degenerate_witness_kinds`` — the claim⇔pin link). The few
truthful bullets that resist the shared canon are pinned by name (``DEVIANT_BULLETS``). The docstring sweeps verify
the corpus against this module; ``expected_bullet`` prints the form a conforming bullet must carry, so authoring one
is copying that output and filling the slot.

The templates and clauses are distilled from the corpus's own majority phrasing, not invented: where the same
behavior was worded several ways, the most frequent form won and the variants converge on it.
"""

import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import cast

from tests.support.spec import Spec, SpecPin

from pomata._policy import POLICIES, NanPolicy, NullPolicy

__all__ = (
    "BULLET_CANON",
    "DEVIANT_BULLETS",
    "OUTCOME_CANON",
    "PARTITIONING_CANON",
    "PIN_CLASS_MARKERS",
    "ROLE_NOUNS",
    "EdgeClass",
    "asserted_outcome_kinds",
    "bullet_matches",
    "degenerate_witness_kinds",
    "expected_bullet",
    "outcome_clause",
    "required_classes",
    "scenario_witnesses",
)


class EdgeClass(Enum):
    """The canonical bullet classes, in the exact order they appear in every Edge-case list."""

    NULL = "Null"
    NAN = "NaN"
    DOMAIN = "Domain"
    INSUFFICIENT_SAMPLE = "Insufficient sample"
    DEGENERATE_DENOMINATOR = "Degenerate denominator"
    NON_FINITE_INPUT = "Non-finite input"
    STABILITY = "Stability"
    WINDOW_ONE = "window == 1"
    PARTITIONING = "Partitioning"


# The subject noun each input role contributes to a canonical sentence ("a ``null`` return is skipped", "a ``NaN``
# price propagates"). Keyed by the spec's first input role; extend only alongside a new role in the spec engine.
ROLE_NOUNS: Mapping[str, str] = {
    "asset_returns": "asset return",
    "benchmark": "benchmark return",
    "close": "close",
    "cost": "cost",
    "dividend_per_share": "per-share amount",
    "equity_curve": "equity",
    "expr": "value",
    "funding_rate": "rate",
    "high": "price",
    "low": "price",
    "open": "price",
    "pnl_gross": "gross P&L",
    "price": "price",
    "quantity": "quantity",
    "returns": "return",
    "returns_gross": "gross return",
    "volume": "volume",
    "wave": "value",
    "weight": "weight",
}


# Pin-label markers that activate a conditional class: a pin whose snake_case label contains one of the class's
# markers is that class's CANDIDATE witness; the pin's expected lanes must then show a matching degenerate outcome
# (null / NaN / an exact 0 / an infinity) — an ordinary finite outcome under a matching label is a definedness
# witness, not an edge case. An infinity in a pin's INPUTS activates the non-finite-input class regardless of label.
PIN_CLASS_MARKERS: Mapping[EdgeClass, tuple[str, ...]] = {
    EdgeClass.DOMAIN: (
        "wiped_out",
        "wipeout",
        "domain",
        "non_positive",
        "below_negative_one",
        "crossing_zero",
    ),
    EdgeClass.INSUFFICIENT_SAMPLE: (
        "single_row",
        "single_pair",
        "single_period",
        "single_observation",
        "fewer_than",
        "one_pair",
    ),
    EdgeClass.DEGENERATE_DENOMINATOR: (
        "constant_window",
        "constant_benchmark",
        "constant_is_nan",
        "constant_series_is",
        "constant_typical",
        "constant_range",
        "constant_portfolio",
        "constant_downside",
        "flat_window",
        "flat_series_is",
        "flat_bar",
        "flat_range",
        "flat_returns",
        "flat_rsi",
        "flat_path",
        "flat_tail",
        "flat_zero",
        "zero_volatility",
        "zero_excess",
        "zero_active",
        "zero_tracking",
        "zero_beta",
        "zero_left_tail",
        "zero_lagged",
        "zero_previous",
        "zero_slow",
        "zero_volume",
        "zero_total_volume",
        "all_zero_is",
        "all_zero_window",
        "all_zero_series_is_nan",
        "all_gain",
        "all_loss",
        "all_wins",
        "all_negative_is",
        "all_at_threshold",
        "no_drawdown",
        "no_decline",
        "no_losses",
        "no_gains",
        "no_downside",
        "no_wins",
        "no_activity_window",
        "monotonic_rise",
        "hh_equals_ll",
        "endpoint_exact_zero",
        "negative_zero",
    ),
    EdgeClass.WINDOW_ONE: ("window_one", "window_equals_one"),
}

# Classes a function's docstring carries although no pin outcome demands them — a deliberate, noteworthy defined
# behavior the maintainer wants stated: the parametric VaR collapses onto the mean on a zero dispersion, so its
# ``z * sigma`` term vanishes without any division and no pin can witness a degenerate denominator. Shrink-only.
ALSO_REQUIRED: frozenset[tuple[str, str]] = frozenset(
    {
        ("value_at_risk_parametric", "Degenerate denominator"),
    }
)


def _iter_scalars(value: object) -> Iterator[float | None]:
    """Every scalar reachable inside a pin's inputs or expected lanes, tuples and struct mappings flattened."""
    if value is None:
        yield None
    elif isinstance(value, float):
        yield value
    elif isinstance(value, tuple):
        for item in cast("tuple[object, ...]", value):
            yield from _iter_scalars(item)
    elif isinstance(value, Mapping):
        for item in cast("Mapping[object, object]", value).values():
            yield from _iter_scalars(item)


def _has_inf(value: object) -> bool:
    return any(scalar is not None and math.isinf(scalar) for scalar in _iter_scalars(value))


def _degenerate_outcome(pin_expected: object) -> bool:
    """Whether a pin's expected lanes show a degenerate outcome: a null, a ``NaN``, an exact ``0``, or an infinity."""
    return any(
        scalar is None or math.isnan(scalar) or math.isinf(scalar) or scalar == 0.0
        for scalar in _iter_scalars(pin_expected)
    )


def _matches(label: str, edge_class: EdgeClass) -> bool:
    return any(marker in label for marker in PIN_CLASS_MARKERS[edge_class])


def _pin_classes(spec: Spec) -> set[EdgeClass]:
    """The conditional classes a spec's pins activate: a label marker plus a matching degenerate outcome; a pin
    matched by the domain markers is a domain witness only, never a degenerate-denominator one.
    """
    active: set[EdgeClass] = set()
    for pin in spec.pins:
        if _has_inf(pin.inputs):
            active.add(EdgeClass.NON_FINITE_INPUT)
            continue
        if _matches(pin.label, EdgeClass.WINDOW_ONE):
            active.add(EdgeClass.WINDOW_ONE)
        if not _degenerate_outcome(pin.expected):
            continue
        if _matches(pin.label, EdgeClass.DOMAIN):
            active.add(EdgeClass.DOMAIN)
            continue
        if _matches(pin.label, EdgeClass.INSUFFICIENT_SAMPLE):
            active.add(EdgeClass.INSUFFICIENT_SAMPLE)
        if _matches(pin.label, EdgeClass.DEGENERATE_DENOMINATOR):
            active.add(EdgeClass.DEGENERATE_DENOMINATOR)
    return active


def required_classes(spec: Spec) -> tuple[EdgeClass, ...]:
    """
    The exact, ordered bullet classes ``spec``'s declarations demand: Null, NaN, and Partitioning always; each
    conditional class exactly when a pin (or the conditioning filter) witnesses it; plus the sanctioned
    ``ALSO_REQUIRED`` entries.
    """
    active = _pin_classes(spec)
    if spec.conditioning is not None:
        active.add(EdgeClass.STABILITY)
    active.update(EdgeClass(value) for name, value in ALSO_REQUIRED if name == spec.name)
    active.update((EdgeClass.NULL, EdgeClass.NAN, EdgeClass.PARTITIONING))
    return tuple(edge_class for edge_class in EdgeClass if edge_class in active)


@dataclass(frozen=True)
class CanonSentence:
    """A canonical Null / NaN sentence template. ``{noun}`` is the subject noun filled from the spec's first input
    role, ``{window}`` the lookback parameter's name, and ``{slot}`` — where present — the single free position a
    per-function clause may occupy.
    """

    template: str


# The canonical Null / NaN wording, one entry per key ``null_canon_key`` / ``nan_canon_key`` selects. Distilled from
# the corpus's majority phrasing: the skeleton is byte-identical across every function sharing the key, and ``{slot}``
# (where present) marks the one position a per-function clause may occupy. A bullet conforms when its text is that
# skeleton with the slot filled, optionally followed by one further clause a sanctioned connector appends (see
# ``bullet_matches``). The seven latching cycle functions all read a price series, so their two templates name
# ``price`` outright rather than the generic ``expr`` role noun.
BULLET_CANON: Mapping[str, CanonSentence] = {
    # -- Null, by NullPolicy ------------------------------------------------------------------------------------
    "null/skipped": CanonSentence("a ``null`` {noun} is skipped{slot}; an all-null (or empty) series yields ``null``."),
    "null/skipped_pairwise": CanonSentence(
        "an observation is used only where both legs are present; a ``null`` in either drops that pair."
    ),
    "null/absorbed": CanonSentence(
        "``null`` handling follows ``pl.max_horizontal``, which **skips** ``null`` candidates rather than "
        "propagating them{slot}."
    ),
    "null/propagates": CanonSentence(
        "a ``null`` {noun} makes that row ``null`` (``null`` takes precedence over ``NaN``)."
    ),
    "null/in_window_is_null": CanonSentence(
        "a window containing a ``null`` yields ``null`` (the window must hold ``{window}`` non-null values)."
    ),
    "null/bridged": CanonSentence(
        "a leading ``null`` run stays ``null`` until the first non-null seed; an interior ``null`` yields ``null`` "
        "at that position while the recursion continues across the gap."
    ),
    "null/latches": CanonSentence("a ``null`` price latches ``null`` for every row from there."),
    # The directional-movement pair (dm_plus / dm_minus): a null in either leg zeroes the affected raw movement
    # rather than propagating a miss — byte-identical across the pair, so no free slot.
    "null/movement_zeroed": CanonSentence(
        "a ``null`` in ``high`` or ``low`` makes the affected raw movement ``0`` for the rows whose difference it "
        "touches, so the raw movement carries no interior nulls and the only nulls emitted are the ``window - 1`` "
        "warm-up nulls from :func:`rma`."
    ),
    # -- NaN, by NanPolicy --------------------------------------------------------------------------------------
    "nan/poisons": CanonSentence("a ``NaN`` {noun} propagates, yielding ``NaN``."),
    "nan/poisons_pairwise": CanonSentence("a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``."),
    "nan/propagates_endpoint": CanonSentence("a ``NaN`` at either endpoint propagates, yielding ``NaN``."),
    "nan/propagates_window": CanonSentence("a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
    "nan/propagates_row": CanonSentence("a ``NaN`` {noun} yields ``NaN`` for that row."),
    "nan/latches": CanonSentence(
        "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null position."
    ),
    "nan/latches_null": CanonSentence(
        "a ``NaN`` price latches ``null`` for every row from there, as any non-finite value does."
    ),
    # The directional-movement pair (dm_plus / dm_minus): the own-side ``NaN`` latches while the opposing-side ``NaN``
    # zeroes the raw movement — the two legs mirror each other under the high/low and up/down swap, so ``{slot}`` spans
    # the mirrored span between the fixed opener and the shared close.
    "nan/movement_asymmetric": CanonSentence("a ``NaN`` in {slot} movement is silently dropped there."),
}


# The byte-exact terminal outcome clause each structural bullet closes on, frozen per class as the corpus carries it:
# a structural bullet must END, before at most one sanctioned trailing tail (a "reported, not clipped" coda or a
# clarifying parenthetical), with the clause ``outcome_clause`` extracts — and that clause must be one of these. The
# clause is where the class states its result (``a loud NaN`` for a domain violation, ``the ratio is +/-inf (or NaN
# ...)`` for a degenerate denominator, ...); the per-function head before it stays free. A reworded outcome reddens
# the freeze against the printed list. Re-derived from the corpus; grows or shrinks only with the docstrings.
OUTCOME_CANON: Mapping[EdgeClass, tuple[str, ...]] = {
    EdgeClass.DOMAIN: (
        "the result is a loud ``NaN``.",
        "the bar is ``-inf`` (a long) or ``+inf`` (a short), and a zero previous price makes the next bar ``+/-inf``.",
        "the result is a loud ``NaN`` — except with both prices negative, where the ratio is positive and the log is "
        "silently finite, an economically meaningless number the caller must screen for.",
    ),
    EdgeClass.INSUFFICIENT_SAMPLE: (
        "the result is ``null``.",
        "the result is a ``0 / 0``, i.e. ``NaN``.",
        "this is the defined geometric behavior, not an error.",
        "the result is exactly ``0``.",
        "the result is exactly ``0``, not ``null``.",
        "its (only) drawdown is exactly ``0``, not ``null``.",
        "the result is ``+inf``.",
        "the standardized moment is a ``0 / 0``, i.e. ``NaN``.",
    ),
    EdgeClass.DEGENERATE_DENOMINATOR: (
        "the result is a ``0 / 0``, i.e. ``NaN``.",
        "the result is a ``0 / 0``, i.e. ``NaN`` — the underlying :func:`dx` is the immediate ``0 / 0``, which then "
        "poisons the ADX recursion.",
        "the result is a ``0 / 0``, i.e. ``NaN`` — inherited from :func:`adx`, which then poisons the averaging of "
        "the current and ``window``-ago values.",
        "a zero-variance benchmark makes :func:`beta` ``NaN`` (a ``0 / 0``), which propagates here.",
        "a zero-variance window benchmark makes the slope ``NaN`` (a ``0 / 0``), which propagates here.",
        "the oscillator is exactly ``0``.",
        "a finite flat bar reads ``0`` even when ``open`` or ``close`` is ``null``, and only a ``null`` ``high`` or "
        "``low`` (which leaves the range itself ``null``) still yields ``null`` on a flat bar.",
        "all three bands collapse onto the constant — even at ``window == 1``, or just after a much larger value has "
        "left the window.",
        "the ratio is ``+/-inf`` (or ``NaN`` when the excess growth is also zero).",
        "a window starting exactly at zero blows the endpoint ratio to ``+inf``.",
        "the ratio is ``+/-inf`` (or ``NaN`` when the growth is also zero).",
        "the result is a ``0 / 0``, i.e. ``NaN``, detected exactly via the rolling extremes (its rolling maximum "
        "equals its rolling minimum) rather than the rounding noise a sub-ULP denominator residual would produce.",
        "a sub-ULP residual in the rolling-sum denominator cannot fake a finite reading.",
        "``+inf`` when there are no losses (the profit factor diverges) or a zero left tail (the tail ratio "
        "diverges), and ``NaN`` where a ``0 * inf`` arises.",
        "the result is ``0``.",
        "a single bar with zero range and zero gap already triggers it.",
        "the result is exactly ``0``.",
        "the ratio is ``+inf`` (or ``NaN`` when the net return is also zero).",
        "the result is ``+/-inf`` (or ``NaN`` when the mean active is also zero, the ``0 / 0``).",
        "the result is ``+/-inf`` (or ``NaN`` when the mean active is also zero).",
        "the smoothing constant sits at the slow bound and KAMA barely moves.",
        "the result is ``null``.",
        "the half-width vanishes and all three bands collapse onto the EMA.",
        "the standardized moment is a ``0 / 0``, i.e. ``NaN``.",
        "its :func:`sharpe_ratio` is infinite and the result is ``+/-inf``.",
        "the difference is exactly ``0``.",
        "the ratio is exactly ``0``.",
        "the ratio is ``+inf`` (or ``NaN`` when every return sits at the threshold, a ``0 / 0``).",
        "the result is a ``0 / 0``, i.e. ``NaN`` — a non-zero gap over a zero slow EMA is ``+/-inf``.",
        "the ratio is ``+inf`` (or ``NaN`` when an all-``0`` series also has zero gross gain, a ``0 / 0``).",
        "the ratio is ``+/-inf`` with the sign of the total return (or ``NaN`` when the total return is also zero).",
        "a zero price relative logs to ``-inf`` and a positive price over a zero previous price logs to ``+inf``.",
        "a zero change is a ``0 / 0``, i.e. ``NaN`` — while a non-zero change over it is ``+/-inf`` (the sign tracks "
        "the change), reported, not clipped, and a negative-zero ``-0.0`` previous price flips that sign but does not "
        "arise from real price data.",
        "with no edge (``p <= 0.5``) it is ``>= 1`` and the probability saturates at ``1`` (an all-losing ``p = 0`` "
        "divides by zero and clips to ``1``), while an all-winning ``p = 1`` gives ``0``.",
        "the result is a ``0 / 0``, i.e. ``NaN`` — a non-zero change over a zero lagged value is ``+/-inf``.",
        "the ratio is ``+/-inf`` (or ``NaN`` when the mean excess is also zero).",
        "the ratio is ``+/-inf`` (or ``NaN`` when the mean excess is also zero, the exact-zero rolling mean pinning "
        "the numerator so no slid-out residue rides above it).",
        "the result is exactly ``0`` — pinned explicitly, even where a much larger value has just left the window and "
        "the incremental rolling kernel would otherwise leave a cancellation residue.",
        "only an ``excess`` of zero on a drawdown-free curve gives ``+/-inf`` (or ``NaN`` when the excess growth is "
        "also zero).",
        "the result is a ``0 / 0``, i.e. ``NaN`` — off that level it is ``+/-inf`` instead.",
        "the result is a ``0 / 0``, i.e. ``NaN`` — off that level it is ``+/-inf`` instead, and either value then "
        "propagates through the slowing and %D averages.",
        "when the 5th-percentile return is exactly ``0`` against a non-zero 95th the ratio is ``+inf`` — reported, "
        "not clipped, following IEEE division.",
        "when a window's 5th-percentile return is exactly ``0`` against a non-zero 95th the ratio is ``+inf`` (or "
        "``NaN`` when the 95th is also ``0``).",
        "a zero-variance benchmark instead makes :func:`beta` ``NaN``, which propagates here.",
        "a zero-variance benchmark window instead makes the slope ``NaN``, which propagates here.",
        "the one-period rate of change is a ``0 / 0``, i.e. ``NaN`` while the EMA holds at zero, and ``+/-inf`` — "
        "reported, not clipped — the moment it moves off it.",
        "a finite buying pressure over an exactly-zero true range (the missing-``low`` fallback) is left to IEEE-754 "
        "as ``+/-inf``, and a near-flat range is reported, not clipped.",
        "``z * sigma`` vanishes and the result is the mean itself.",
        "a near-flat window (tiny ranges after a much larger one has slid out) is not silenced, since ``VI+`` is "
        "unbounded above and the streaming quotient cannot be clipped to a range, degrading in precision past a sane "
        "dynamic range.",
        "a sub-ULP rolling-sum residual cannot leak a spurious ``+/-inf`` instead.",
        "the result is a ``0 / 0``, i.e. ``NaN`` — a non-zero numerator over the zero range is ``+/-inf``.",
    ),
    EdgeClass.NON_FINITE_INPUT: (
        "a flat or long bar at an infinite ``price`` is ``0 * inf``, i.e. ``NaN``.",
        "a flat bar at an infinite ``price`` is ``0 * inf``, i.e. ``NaN``.",
        "a held bar at an infinite ``price`` is ``0 * inf``, i.e. ``NaN``.",
        "an ``inf`` quantity follows IEEE-754 through the arithmetic of the turnover difference, whose ``inf`` marks "
        "a trade and charges the flat ``fee``.",
        "an ``inf`` quantity follows IEEE-754 through the arithmetic of the turnover difference, an infinite move "
        "charging an ``inf`` cost.",
        "an ``inf`` weight follows IEEE-754 through the arithmetic of the turnover difference, an infinite move "
        "charging an ``inf`` cost.",
        "an ``inf`` return follows IEEE-754 through the arithmetic, and a canceling opposite infinity latches the "
        "running total to ``NaN``.",
        "the flow signs with ``quantity * dividend_per_share``.",
        "an ``inf`` return follows IEEE-754 through the arithmetic, where a later opposite-sign infinite factor flips "
        "the running product's sign.",
        "an ``inf`` quantity follows IEEE-754 through the arithmetic, and an infinite ``price`` propagates through "
        "the one-bar change.",
        "an ``inf`` quantity follows IEEE-754 through the arithmetic, where an infinite ``price`` contributes ``1 / "
        "inf = 0`` to the reciprocal change.",
        "an ``inf`` gross P&L follows IEEE-754 through the arithmetic.",
        "the return signs with ``weight * asset_returns``.",
        "an ``inf`` price follows IEEE-754 through the ratio and its logarithm, where two consecutive same-sign "
        "infinite prices divide to ``inf / inf = NaN``.",
        "an ``inf`` gross return follows IEEE-754 through the arithmetic.",
        "an ``inf`` price follows IEEE-754 through the ratio and the minus one, where two consecutive same-sign "
        "infinite prices divide to ``inf / inf = NaN``.",
        "an ``inf`` weight follows IEEE-754 through the arithmetic, where a single infinite ``weight`` carries "
        "``|inf| = inf`` forward.",
    ),
    EdgeClass.STABILITY: (
        "real return samples are far from the regime.",
        "real market windows are far from the regime.",
        "it stays in range but, past a sane dynamic range, degrades in precision.",
        "this is the continuous limit and keeps the phase invariant under a lossless rescale of the price, whereas a "
        "fixed threshold would be scale-dependent.",
        "that band is excluded from the property tiers, and the value is reported as computed.",
        "an exit more than one order of magnitude above the window's scale is recomputed exactly, while a smaller "
        "exit onto a window whose own spread has collapsed can amplify that residue through the near-zero variance "
        "into a wrong finite value.",
        "there is no cycle to adapt to.",
        "real return windows are far from the regime.",
        "this is the continuous limit and keeps the sine invariant under a lossless rescale of the price, whereas a "
        "fixed threshold would be scale-dependent.",
        "real market betas are far from the regime.",
        "that near-constant band is excluded from the property tiers, and the value is reported as computed.",
        "once a much larger value exits the window a near-constant remainder (relative spread below the conditioning "
        "floor) can diverge from a fresh two-pass computation — the excluded tail, reported as computed.",
    ),
    EdgeClass.WINDOW_ONE: (
        "each Aroon line is ``0`` or ``100`` and the oscillator takes only ``-100``, ``0``, or ``+100``.",
        "the ``max_horizontal``-reduced true range.",
        "every non-null result is ``NaN``.",
        "each row reports ``+100`` on an up move, ``-100`` on a down move, and ``NaN`` on no move.",
        "the expression reproduces the input.",
        "the bands are the bar's own ``high`` and ``low``, and the middle is its :func:`price_median`.",
        "the EMA reproduces the input.",
        "``max == min`` makes it flat by construction and ``fisher`` is ``NaN`` from the first row.",
        "the midpoint reproduces the input.",
        "the midprice reduces to the per-bar :func:`price_median`.",
        "``mom`` is the one-step first difference ``x_t - x_{t-1}``.",
        "each row reports ``100`` on an up move, ``0`` on a down move, and ``NaN`` on no move.",
        "the smoothing factor is ``1``, the warm-up vanishes, and the result reproduces the input.",
        "``roc`` is the one-period simple return in percent.",
        "the SMA reproduces the input.",
        "the result is ``0`` with the default ``ddof = 0``.",
        "a ``NaN`` self-heals once the true range is finite again.",
        "the expression reproduces the input up to a floating-point rounding.",
        "the TRIMA reproduces the input.",
        "TRIX is the one-period rate of change of ``expr``.",
        "the first row is ``null``.",
        "the VWMA reproduces the price to within a rounding ULP (``(p * v) / v`` is one float multiply-divide, not an "
        "identity copy — its siblings' bit-exact ``window == 1`` identity does not apply here).",
        ":math:`\\%R = -100\\,(H - C) / (H - L)`.",
        "the WMA reproduces the input.",
    ),
}

_STRUCTURAL_LABELS: frozenset[str] = frozenset(edge_class.value for edge_class in OUTCOME_CANON)

# One sanctioned trailing tail a structural bullet may carry after its outcome clause — a "reported, not clipped"
# coda or a clarifying parenthetical — stripped before the clause is read; then the clause is the segment after the
# last connector. The check and the ``OUTCOME_CANON`` freeze read the clause through this one function, so they agree.
_OUTCOME_TAILS: tuple[str, ...] = (
    r" — reported, not clipped(?: \([^()]*\))?\.$",
    r" — all reported, not clipped\.$",
    r" — never a plausible wrong number\.$",
    r"; (?:all )?reported, not clipped\.$",
    r"\s*\([^()]*\)\.$",
)
_OUTCOME_CONNECTORS: tuple[str, ...] = (", so ", "; ", ": ")


def outcome_clause(text: str) -> str:
    """The terminal outcome clause a structural bullet closes on: one sanctioned trailing tail stripped, then the
    segment after its last connector — the clause ``OUTCOME_CANON`` freezes byte-for-byte.
    """
    core = text
    for pattern in _OUTCOME_TAILS:
        match = re.search(pattern, core)
        if match:
            base = core[: match.start()].rstrip()
            core = base if base.endswith(".") else base + "."
            break
    cut = max(core.rfind(connector) for connector in _OUTCOME_CONNECTORS)
    if cut == -1:
        return core
    return next(core[cut + len(connector) :] for connector in _OUTCOME_CONNECTORS if core.rfind(connector) == cut)


# The degenerate outcome tokens a Degenerate-denominator bullet may assert, and a token's kind. A token a clause
# negates with "cannot" / "spurious" (a residue explicitly ruled OUT, e.g. vwma's guarded ``+/-inf``) is not an
# assertion and needs no pin; every asserted ``NaN`` or infinity does (the claim⇔pin link).
_OUTCOME_TOKENS: tuple[tuple[str, str], ...] = (
    ("``+/-inf``", "inf"),
    ("``+inf``", "inf"),
    ("``-inf``", "inf"),
    ("``NaN``", "nan"),
)
_OUTCOME_NEGATORS: tuple[str, ...] = ("cannot", "spurious")


def asserted_outcome_kinds(text: str) -> set[str]:
    """The degenerate outcome kinds (``nan`` / ``inf``) a structural bullet asserts, skipping a token its clause
    negates with "cannot" / "spurious" (a residue ruled out, not a produced value).
    """
    kinds: set[str] = set()
    for segment in re.split(r"[;—]", text):
        lowered = segment.lower()
        for token, kind in _OUTCOME_TOKENS:
            start = 0
            while (position := segment.find(token, start)) != -1:
                if not any(negator in lowered[:position] for negator in _OUTCOME_NEGATORS):
                    kinds.add(kind)
                start = position + len(token)
    return kinds


def _pin_kinds(expected: object) -> set[str]:
    """The outcome kinds (``null`` / ``nan`` / ``inf`` / ``zero`` / ``finite``) a pin's expected lanes carry."""
    kinds: set[str] = set()
    for scalar in _iter_scalars(expected):
        if scalar is None:
            kinds.add("null")
        elif math.isnan(scalar):
            kinds.add("nan")
        elif math.isinf(scalar):
            kinds.add("inf")
        elif scalar == 0.0:
            kinds.add("zero")
        else:
            kinds.add("finite")
    return kinds


def degenerate_witness_kinds(spec: Spec) -> set[str]:
    """The outcome kinds the spec's Degenerate-denominator pins witness: every pin whose label matches a
    degenerate-denominator marker contributes its expected kinds (a pin with an infinity in its INPUTS is a
    non-finite-input witness instead, so it is skipped).
    """
    kinds: set[str] = set()
    for pin in spec.pins:
        if _has_inf(pin.inputs):
            continue
        if _matches(pin.label, EdgeClass.DEGENERATE_DENOMINATOR):
            kinds |= _pin_kinds(pin.expected)
    return kinds


# The classes whose activation demands a demonstration scenario in the Examples block: Null, NaN, and Partitioning
# ride the three canonical usage scenarios every function opens with, and Stability states a precision regime with
# no byte-exact outcome a doctest could print.
_SCENARIO_CLASSES: tuple[EdgeClass, ...] = (
    EdgeClass.DOMAIN,
    EdgeClass.INSUFFICIENT_SAMPLE,
    EdgeClass.DEGENERATE_DENOMINATOR,
    EdgeClass.NON_FINITE_INPUT,
    EdgeClass.WINDOW_ONE,
)

# The fixed order the Degenerate-denominator kind collapse walks, so the demanded scenario sequence is deterministic.
_DEGENERATE_KIND_ORDER: tuple[str, ...] = ("null", "nan", "inf", "zero")


def _class_witnesses(spec: Spec, edge_class: EdgeClass) -> list[SpecPin]:
    """The pins that witness ``edge_class`` for an Examples demonstration, in spec order — the ``_pin_classes``
    triage with the domain-marked pins reserved to the Domain class.
    """
    witnesses: list[SpecPin] = []
    for pin in spec.pins:
        if _has_inf(pin.inputs):
            if edge_class is EdgeClass.NON_FINITE_INPUT:
                witnesses.append(pin)
            continue
        if edge_class is EdgeClass.WINDOW_ONE and _matches(pin.label, EdgeClass.WINDOW_ONE):
            witnesses.append(pin)
            continue
        if edge_class is EdgeClass.DOMAIN:
            if _degenerate_outcome(pin.expected) and _matches(pin.label, EdgeClass.DOMAIN):
                witnesses.append(pin)
            continue
        if _matches(pin.label, EdgeClass.DOMAIN):
            continue
        if edge_class is EdgeClass.INSUFFICIENT_SAMPLE and _matches(pin.label, EdgeClass.INSUFFICIENT_SAMPLE):
            witnesses.append(pin)
        if edge_class is EdgeClass.DEGENERATE_DENOMINATOR and _matches(pin.label, EdgeClass.DEGENERATE_DENOMINATOR):
            witnesses.append(pin)
    return witnesses


def scenario_witnesses(spec: Spec) -> tuple[tuple[EdgeClass, str, frozenset[str]], ...]:
    """
    The ordered edge scenarios ``spec`` demands of its Examples block, as ``(class, pin label, outcome kinds)``:
    for each conditional class the spec's pins activate, one demonstration per witnessing pin — collapsed, for the
    Degenerate-denominator class, to one scenario per distinct witnessed outcome kind (``null`` / ``nan`` / ``inf``
    / ``zero``), each carried by the first pin in spec order that shows it. A required class none of whose
    witnesses shows a degenerate outcome (the parametric VaR's mean collapse) demands one scenario from its first
    witness, with no asserted kind.
    """
    active = set(required_classes(spec))
    demanded: list[tuple[EdgeClass, str, frozenset[str]]] = []
    for edge_class in _SCENARIO_CLASSES:
        if edge_class not in active:
            continue
        pins = _class_witnesses(spec, edge_class)
        if edge_class is EdgeClass.DEGENERATE_DENOMINATOR:
            seen: set[str] = set()
            for pin in pins:
                new = [kind for kind in _DEGENERATE_KIND_ORDER if kind in _pin_kinds(pin.expected) - seen]
                if new:
                    seen.update(new)
                    demanded.append((edge_class, pin.label, frozenset(new)))
            if not seen and pins:
                demanded.append((edge_class, pins[0].label, frozenset()))
        elif pins:
            demanded.append((edge_class, pins[0].label, frozenset()))
    return tuple(demanded)


# The two canonical Partitioning sentences: a function with cross-row memory (a window, a recurrence, a shift, a
# cumulation) must be partitioned so each series is computed on its own history, while an elementwise transform is
# already correct because ``.over`` partitions it identically. A Partitioning bullet must be byte-equal to exactly
# one of these — no per-function slot, no trailing clause, no usage example.
PARTITIONING_CANON: frozenset[str] = frozenset(
    {
        "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        (
            "already correct on a multi-series panel: ``.over(...)`` partitions identically and is therefore "
            "optional here."
        ),
    }
)


# Bullets whose truthful wording resists the shared canon and is pinned as itself: ``ultimate_oscillator`` documents
# its asymmetric ``pl.min_horizontal`` / ``pl.max_horizontal`` fallback, which neither the row nor the windowed NaN
# canon can express, and its matching ``null`` wording rides the same fallback. Shrink-only.
DEVIANT_BULLETS: frozenset[tuple[str, str]] = frozenset(
    {
        ("ultimate_oscillator", "NaN"),
        ("ultimate_oscillator", "Null"),
    }
)


# The directional-movement pair (dm_plus / dm_minus): a null / NaN in the own vs. opposing side maps to a zeroed raw
# movement rather than a propagated miss, so both legs read their own two canon forms instead of the policy defaults.
ABSORBED_MOVEMENT: frozenset[str] = frozenset({"dm_minus", "dm_plus"})


def null_canon_key(spec: Spec) -> str:
    """The Null-bullet canon key ``spec``'s declarations select."""
    if spec.name in ABSORBED_MOVEMENT:
        return "null/movement_zeroed"
    policy = POLICIES[spec.name][0]
    if policy is NullPolicy.SKIPPED:
        return "null/skipped_pairwise" if len(spec.inputs) == 2 else "null/skipped"
    return f"null/{policy.name.lower()}"


# The two endpoint-ratio rolling metrics: only the window's two endpoints enter the computation, so the windowed
# NaN canon ("inside the window propagates") would overstate the contamination.
ENDPOINT_NAN: frozenset[str] = frozenset({"cagr_rolling", "total_return_rolling"})

# The path-dependent stop-and-reverse recurrence (parabolic_sar) reads a windowed high/low pair yet carries no
# ``window`` parameter, so its NaN wording is the windowed "inside the window propagates" form the row canon default
# would otherwise miss.
WINDOWED_NAN: frozenset[str] = frozenset({"parabolic_sar"})


def nan_canon_key(spec: Spec) -> str:
    """The NaN-bullet canon key ``spec``'s declarations select."""
    if spec.name in ABSORBED_MOVEMENT:
        return "nan/movement_asymmetric"
    policy = POLICIES[spec.name][1]
    if policy is NanPolicy.POISONS:
        return "nan/poisons_pairwise" if len(spec.inputs) == 2 else "nan/poisons"
    if policy is NanPolicy.LATCHES:
        return "nan/latches_null" if POLICIES[spec.name][0] is NullPolicy.LATCHES else "nan/latches"
    if spec.name in ENDPOINT_NAN:
        return "nan/propagates_endpoint"
    if spec.name in WINDOWED_NAN:
        return "nan/propagates_window"
    windowed = any(name.startswith("window") for name in (spec.params or {}))
    return "nan/propagates_window" if windowed else "nan/propagates_row"


_SLOT_SENTINEL = "\x01"
# A per-function clause may follow the canonical skeleton, introduced by one of the four connectors the corpus uses:
# an em-dash, an opening parenthesis, a semicolon, or a comma. The skeleton itself stays byte-fixed.
_TRAILING_CLAUSE = r"(?:,|;| —| \().*"


def _window_name(spec: Spec) -> str:
    """The spec's first ``window*`` parameter — the lookback the ``null/in_window_is_null`` canon names."""
    for name in spec.params:
        if name.startswith("window"):
            return name
    return "window"


def _render_canon(key: str, spec: Spec, slot: str = "<trigger>") -> str:
    """The canon skeleton for ``key``, the spec's noun and window filled and its free slot shown as ``slot``."""
    noun = ROLE_NOUNS[spec.inputs[0]]
    return BULLET_CANON[key].template.format(noun=noun, window=_window_name(spec), slot=slot)


def _canon_pattern(key: str, spec: Spec) -> str:
    """The regex a conforming bullet under ``key`` must match: the filled skeleton with its slot a free segment and
    an optional trailing clause before the final period.
    """
    noun = ROLE_NOUNS[spec.inputs[0]]
    rendered = BULLET_CANON[key].template.replace("{slot}", _SLOT_SENTINEL).format(noun=noun, window=_window_name(spec))
    body = rendered.removesuffix(".")
    if _SLOT_SENTINEL in body:
        before, _, after = body.partition(_SLOT_SENTINEL)
        skeleton = f"{re.escape(before)}(?P<free>.*?){re.escape(after)}"
    else:
        skeleton = re.escape(body)
    return rf"^{skeleton}(?:{_TRAILING_CLAUSE})?\.$"


def bullet_matches(spec: Spec, label: str, text: str) -> bool:
    """
    Whether one Note bullet's ``text`` is the canonical wording its ``label`` demands. A ``Null`` / ``NaN`` bullet
    must be the canon skeleton (``null_canon_key`` / ``nan_canon_key``) with its free slot filled and at most one
    sanctioned trailing clause; a structural bullet must close on a terminal outcome clause byte-equal to one of
    its class's ``OUTCOME_CANON`` entries; a ``Partitioning`` bullet must be byte-equal to a ``PARTITIONING_CANON``
    sentence.
    """
    if label == "Null":
        return re.match(_canon_pattern(null_canon_key(spec), spec), text) is not None
    if label == "NaN":
        return re.match(_canon_pattern(nan_canon_key(spec), spec), text) is not None
    if label in _STRUCTURAL_LABELS:
        return outcome_clause(text) in OUTCOME_CANON[EdgeClass(label)]
    if label == "Partitioning":
        return text in PARTITIONING_CANON
    return True


def expected_bullet(spec: Spec, label: str) -> str:
    """
    The canonical form ``label`` demands for ``spec``, as a failure message an author can copy. For ``Null`` / ``NaN``
    the skeleton with its free position rendered as ``<trigger>``; for a structural class the terminal outcome clauses
    one of which the bullet must close on; for ``Partitioning`` the canon sentences one of which must be reproduced
    verbatim; the empty string where phrasing is not enforced.
    """
    if label == "Null":
        return _render_canon(null_canon_key(spec), spec)
    if label == "NaN":
        return _render_canon(nan_canon_key(spec), spec)
    if label in _STRUCTURAL_LABELS:
        return "the bullet must close on one of, verbatim:\n  " + "\n  ".join(OUTCOME_CANON[EdgeClass(label)])
    if label == "Partitioning":
        return "one of, verbatim:\n  " + "\n  ".join(sorted(PARTITIONING_CANON))
    return ""
