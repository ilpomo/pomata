"""
The canonical edge-case taxonomy: which Note bullets a function's declarations demand, and their canonical wording.

Every public docstring closes its Note with an ``**Edge-case behavior:**`` list. This module is the single home of
that list's contract: the class vocabulary (one label per concept, fixed order), the presence rules (a class appears
exactly when the machine-readable declarations activate it — the policy registry, the spec's pins, its conditioning
filter, its shape), and the canonical wording each class carries. The ``Null`` and ``NaN`` bullets have a full
sentence template per policy value (``BULLET_CANON``): a byte-identical skeleton with one free slot for the
function's own trigger, plus an optional trailing clause. The structural classes are held only to naming a canonical
outcome fragment (``STRUCTURAL_OUTCOMES``). The few truthful bullets that resist the shared canon are pinned by name
(``DEVIANT_BULLETS``). The docstring sweeps verify the corpus against this module; ``expected_bullet`` prints the
form a conforming bullet must carry, so authoring one is copying that output and filling the slot.

The templates and fragments are distilled from the corpus's own majority phrasing, not invented: where the same
behavior was worded several ways, the most frequent form won and the variants converge on it.
"""

import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import cast

from tests.support.spec import Spec

from pomata._policy import POLICIES, NanPolicy, NullPolicy

__all__ = (
    "BULLET_CANON",
    "DEVIANT_BULLETS",
    "PIN_CLASS_MARKERS",
    "ROLE_NOUNS",
    "STRUCTURAL_OUTCOMES",
    "EdgeClass",
    "bullet_matches",
    "expected_bullet",
    "required_classes",
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

# Pins whose matching label is a naming accident, not an edge witness: the sign-convention pins (a signed zero is a
# convention, not a degeneracy) and the fixed-point accuracy pins (a constant input reproducing itself exercises no
# denominator). Shrink by renaming the pin; never grow without the written reason.
EXCLUDED_PINS: frozenset[tuple[str, str]] = frozenset(
    {
        ("pnl_gross", "short_on_flat_price_is_negative_zero"),
        ("pnl_gross_inverse", "short_flat_price_negative_zero"),
    }
)

# Classes a function's docstring carries although no pin outcome demands them — each a deliberate, noteworthy
# defined behavior the maintainer wants stated (the annualization power of a one-period cagr, kama's guarded
# efficiency ratio, the parametric VaR collapsing to the mean on zero dispersion). Shrink-only.
ALSO_REQUIRED: frozenset[tuple[str, str]] = frozenset(
    {
        ("cagr", "Insufficient sample"),
        ("kama", "Degenerate denominator"),
        ("keltner_channels", "Degenerate denominator"),
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
        if (spec.name, pin.label) in EXCLUDED_PINS:
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
}


# The canonical outcome fragments each structural class names, distilled from the corpus. A structural bullet is held
# only to containing one of these — its per-function trigger and elaboration are free — so the fixed vocabulary is
# the loud outcome the class is about: a loud ``NaN`` for a domain violation; a ``0 / 0`` / infinity / exact ``0`` /
# ``null`` / collapse-onto-the-defined-value for a degenerate denominator; IEEE-754 for a non-finite input; the
# conditioning limit (or "reported as computed") for a stability carve-out; the reduction a one-bar window becomes at
# ``window == 1``. Shrink-only: a narrower fragment set is a stronger contract.
STRUCTURAL_OUTCOMES: Mapping[str, tuple[str, ...]] = {
    "Domain": ("a loud ``NaN``", "economically meaningless"),
    "Insufficient sample": (
        "``null``",
        "``0 / 0``",
        "``+inf``",
        "an exact ``0``",
        "the result is ``0``",
        "``0`` (not",
    ),
    "Degenerate denominator": (
        "``0 / 0``",
        "``NaN``",
        "``+/-inf``",
        "``+inf``",
        "exactly ``0``",
        "an exact ``0``",
        "the result is ``0``",
        "``0`` (not",
        "``null``",
        "``1.0``",
        "collapse onto",
        "the mean itself",
        "gives ``0``",
    ),
    "Non-finite input": ("IEEE-754",),
    "Stability": (
        "float-conditioning limit",
        "conditioning floor",
        "numerically arbitrary",
        "degrades in precision",
        "reported as computed",
        "reported, not clipped",
    ),
    "window == 1": (
        "reproduces",
        "reduces to",
        "identity",
        "``NaN``",
        "``0``",
        "one-step",
        "one-period",
        "self-heals",
        "bar's own",
        "collapse",
    ),
}


# Bullets whose truthful wording resists the shared canon and is pinned as itself. The absorbed-movement pair
# (``dm_plus`` / ``dm_minus``) maps a ``null`` / ``NaN`` in the own vs. opposing side to a zeroed raw movement rather
# than a propagated miss; ``ultimate_oscillator`` documents its asymmetric ``pl.min_horizontal`` /
# ``pl.max_horizontal`` fallback; ``parabolic_sar`` reads a windowed high/low pair yet carries no ``window``
# parameter, so its NaN wording is the windowed form the row canon cannot express. Shrink-only.
DEVIANT_BULLETS: frozenset[tuple[str, str]] = frozenset(
    {
        ("dm_minus", "NaN"),
        ("dm_minus", "Null"),
        ("dm_plus", "NaN"),
        ("dm_plus", "Null"),
        ("parabolic_sar", "NaN"),
        ("ultimate_oscillator", "NaN"),
        ("ultimate_oscillator", "Null"),
    }
)


def null_canon_key(spec: Spec) -> str:
    """The Null-bullet canon key ``spec``'s declarations select."""
    policy = POLICIES[spec.name][0]
    if policy is NullPolicy.SKIPPED:
        return "null/skipped_pairwise" if len(spec.inputs) == 2 else "null/skipped"
    return f"null/{policy.name.lower()}"


# The two endpoint-ratio rolling metrics: only the window's two endpoints enter the computation, so the windowed
# NaN canon ("inside the window propagates") would overstate the contamination.
ENDPOINT_NAN: frozenset[str] = frozenset({"cagr_rolling", "total_return_rolling"})


def nan_canon_key(spec: Spec) -> str:
    """The NaN-bullet canon key ``spec``'s declarations select."""
    policy = POLICIES[spec.name][1]
    if policy is NanPolicy.POISONS:
        return "nan/poisons_pairwise" if len(spec.inputs) == 2 else "nan/poisons"
    if policy is NanPolicy.LATCHES:
        return "nan/latches_null" if POLICIES[spec.name][0] is NullPolicy.LATCHES else "nan/latches"
    if spec.name in ENDPOINT_NAN:
        return "nan/propagates_endpoint"
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
    sanctioned trailing clause; a structural bullet is held only to naming a canonical outcome fragment for its
    class. ``Partitioning`` is not phrase-checked here — its presence and closing position are the ordering sweep's.
    """
    if label == "Null":
        return re.match(_canon_pattern(null_canon_key(spec), spec), text) is not None
    if label == "NaN":
        return re.match(_canon_pattern(nan_canon_key(spec), spec), text) is not None
    if label in STRUCTURAL_OUTCOMES:
        return any(fragment in text for fragment in STRUCTURAL_OUTCOMES[label])
    return True


def expected_bullet(spec: Spec, label: str) -> str:
    """
    The canonical form ``label`` demands for ``spec``, as a failure message an author can copy. For ``Null`` / ``NaN``
    the skeleton with its free position rendered as ``<trigger>``; for a structural class the outcome fragments one of
    which must appear; the empty string where phrasing is not enforced.
    """
    if label == "Null":
        return _render_canon(null_canon_key(spec), spec)
    if label == "NaN":
        return _render_canon(nan_canon_key(spec), spec)
    if label in STRUCTURAL_OUTCOMES:
        return "text naming one of: " + ", ".join(STRUCTURAL_OUTCOMES[label])
    return ""
