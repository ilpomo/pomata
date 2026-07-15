"""
The canonical edge-case taxonomy: which Note bullets a function's declarations demand, and their canonical wording.

Every public docstring closes its Note with an ``**Edge-case behavior:**`` list. This module is the single home of
that list's contract: the class vocabulary (one label per concept, fixed order), the presence rules (a class appears
exactly when the machine-readable declarations activate it — the policy registry, the spec's pins, its conditioning
filter, its shape), and the canonical sentence each class instantiates (one template per policy value or structural
variant, with a single free slot where the function-specific trigger or tail lives). The docstring sweeps verify the
corpus against this module; the ``expected_bullet`` renderer prints the sentence a conforming docstring must carry,
so authoring a bullet is copying the renderer's output and filling the slot.

The templates are distilled from the corpus's own majority phrasings, not invented: where the same behavior was
worded several ways, the most frequent form won and the variants converge on it.
"""

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import cast

from tests.support.spec import Spec

from pomata._policy import POLICIES, NanPolicy, NullPolicy

__all__ = (
    "BULLET_CANON",
    "PIN_CLASS_MARKERS",
    "ROLE_NOUNS",
    "EdgeClass",
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
    OVERFLOW = "Overflow"
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
# markers is that class's witness. Content detectors (an infinity in the inputs or the expected lanes) back the two
# non-finite classes independently of labels.
PIN_CLASS_MARKERS: Mapping[EdgeClass, tuple[str, ...]] = {
    EdgeClass.DOMAIN: ("wiped_out", "domain", "non_positive", "wipeout"),
    EdgeClass.INSUFFICIENT_SAMPLE: ("single_row", "single_pair", "single_period", "fewer_than", "one_pair"),
    EdgeClass.DEGENERATE_DENOMINATOR: (
        "constant",
        "flat",
        "zero_variance",
        "zero_volatility",
        "zero_volume",
        "zero_range",
        "zero_excess",
        "zero_active",
        "zero_tracking",
        "zero_beta",
        "zero_denominator",
        "zero_dispersion",
        "no_drawdown",
        "no_decline",
        "no_losses",
        "no_downside",
        "monotonic_rise",
        "all_wins",
        "zero_lagged",
    ),
    EdgeClass.WINDOW_ONE: ("window_one", "window_equals_one"),
}


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


def _pin_classes(spec: Spec) -> set[EdgeClass]:
    """The conditional classes a spec's pins activate, by label marker and by content."""
    active: set[EdgeClass] = set()
    for pin in spec.pins:
        for edge_class, markers in PIN_CLASS_MARKERS.items():
            if any(marker in pin.label for marker in markers):
                active.add(edge_class)
        if _has_inf(pin.inputs):
            active.add(EdgeClass.NON_FINITE_INPUT)
        elif _has_inf(pin.expected):
            # An infinity produced from finite inputs: a degenerate denominator when a zero-family marker names the
            # trigger, an overflow otherwise (a magnitude that leaves float64 without any division by zero).
            zero_family = any(
                marker in pin.label for marker in PIN_CLASS_MARKERS[EdgeClass.DEGENERATE_DENOMINATOR]
            ) or any(marker in pin.label for marker in ("negative_zero", "zero_previous"))
            active.add(EdgeClass.DEGENERATE_DENOMINATOR if zero_family else EdgeClass.OVERFLOW)
    return active


def required_classes(spec: Spec) -> tuple[EdgeClass, ...]:
    """
    The exact, ordered bullet classes ``spec``'s declarations demand: Null, NaN, and Partitioning always; each
    conditional class exactly when a pin (or the conditioning filter) witnesses it.
    """
    active = _pin_classes(spec)
    if spec.conditioning is not None:
        active.add(EdgeClass.STABILITY)
    active.update((EdgeClass.NULL, EdgeClass.NAN, EdgeClass.PARTITIONING))
    return tuple(edge_class for edge_class in EdgeClass if edge_class in active)


@dataclass(frozen=True)
class CanonSentence:
    """A canonical sentence template; ``{slot}`` marks the single free position, ``{noun}``/``{window}`` the slots
    the renderer fills from the spec.
    """

    template: str

    def render(self, noun: str = "value", window: str = "window", slot: str = "<trigger>") -> str:
        """The sentence with every slot filled (``slot`` defaults to the author-facing placeholder)."""
        return self.template.format(slot=slot, noun=noun, window=window)


# The canonical wording, one entry per (class, policy-or-variant) pair. The Null and NaN entries key on the policy
# value the registry declares; the structural classes key on a single canonical form. Distilled from the corpus's
# majority phrasings; the docstring sweeps hold every bullet to the matching entry.
BULLET_CANON: Mapping[str, CanonSentence] = {
    # -- Null, by NullPolicy ------------------------------------------------------------------------------------
    "null/skipped": CanonSentence("a ``null`` {noun} is skipped; an all-null (or empty) series yields ``null``."),
    "null/skipped_pairwise": CanonSentence(
        "an observation is used only where both legs are present; a ``null`` in either drops that pair."
    ),
    "null/in_window_is_null": CanonSentence(
        "a window containing a ``null`` yields ``null`` (the window must hold ``{window}`` non-null values)."
    ),
    "null/propagates": CanonSentence(
        "a ``null`` {noun} makes that row ``null`` (``null`` takes precedence over ``NaN``)."
    ),
    "null/propagates_endpoint": CanonSentence(
        "a ``null`` at either window endpoint yields ``null``; being an endpoint quantity, an interior ``null`` "
        "does not affect the result."
    ),
    "null/propagates_peak": CanonSentence(
        "a ``null`` {noun} yields ``null`` at that row while the running peak carries across it unchanged."
    ),
    "null/bridged": CanonSentence(
        "a leading ``null`` run stays ``null`` until the first non-null seed; an interior ``null`` yields ``null`` "
        "at that position while the recursion continues across the gap."
    ),
    "null/latches": CanonSentence("a ``null`` {noun} latches ``null`` for every row from there."),
    # -- NaN, by NanPolicy --------------------------------------------------------------------------------------
    "nan/poisons": CanonSentence("a ``NaN`` {noun} propagates, yielding ``NaN``."),
    "nan/poisons_pairwise": CanonSentence("a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``."),
    "nan/propagates_window": CanonSentence("a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
    "nan/propagates_row": CanonSentence("a ``NaN`` {noun} yields ``NaN`` for that row."),
    "nan/latches": CanonSentence(
        "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null position."
    ),
    "nan/latches_null": CanonSentence(
        "a ``NaN`` {noun} latches ``null`` for every row from there, as any non-finite value does."
    ),
    # -- The structural classes (the free slot carries the function-specific trigger) ----------------------------
    "domain": CanonSentence("{slot}, so the result is a loud ``NaN`` — never a plausible wrong number."),
    "insufficient_sample": CanonSentence("{slot}, so the result is ``null``."),
    "degenerate_denominator_nan": CanonSentence("{slot}, so the result is a ``0 / 0``, i.e. ``NaN``."),
    "degenerate_denominator_inf": CanonSentence("{slot}, so the result is ``+/-inf`` — reported, not clipped."),
    "overflow": CanonSentence("{slot} — reported, not clipped."),
    "non_finite_input": CanonSentence(
        "an ``inf`` {noun} follows IEEE-754 through the arithmetic{slot} (the sign, and any ``inf - inf = NaN``, "
        "included)."
    ),
    "stability": CanonSentence("{slot}"),
    "window_one": CanonSentence("{slot}"),
    "partitioning/standard": CanonSentence(
        "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history{slot}."
    ),
    "partitioning/window": CanonSentence(
        "wrap the call in ``.over(...)`` so the window never spans series boundaries{slot}."
    ),
    "partitioning/elementwise": CanonSentence(
        "already correct on a multi-series panel: ``.over(...)`` partitions identically and is therefore optional "
        "here{slot}."
    ),
}


def expected_bullet(key: str, noun: str = "value", window: str = "window", slot: str = "<trigger>") -> str:
    """
    The canonical bullet body under canon ``key``, slots filled — what a conforming docstring carries after
    ``- **<label>** — `` (the free slot rendered as ``<trigger>`` for the author to replace).
    """
    return BULLET_CANON[key].render(noun=noun, window=window, slot=slot)


def null_canon_key(spec: Spec) -> str:
    """The Null-bullet canon key ``spec``'s declarations select."""
    policy = POLICIES[spec.name][0]
    if policy is NullPolicy.SKIPPED:
        return "null/skipped_pairwise" if len(spec.inputs) == 2 else "null/skipped"
    return f"null/{policy.name.lower()}"


def nan_canon_key(spec: Spec) -> str:
    """The NaN-bullet canon key ``spec``'s declarations select."""
    policy = POLICIES[spec.name][1]
    if policy is NanPolicy.POISONS:
        return "nan/poisons_pairwise" if len(spec.inputs) == 2 else "nan/poisons"
    if policy is NanPolicy.LATCHES:
        return "nan/latches_null" if POLICIES[spec.name][0] is NullPolicy.LATCHES else "nan/latches"
    return "nan/propagates_window" if spec.warmup is not None else "nan/propagates_row"
