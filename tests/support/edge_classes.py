"""
The edge-case taxonomy the docstring truth-couplers read: which Note bullets a declaration demands and which
crafted-input cases witness each.

Every public docstring closes its Note with an ``**Edge-case behavior:**`` list. The list's *membership* is a
contract the declaration owns: the always-present classes (``Null`` / ``NaN`` / ``Partitioning``), and each
conditional class exactly when the declaration's own data activates it. That data is read straight off the
:class:`~tests.support.declaration.Declaration` — the ``conditioning`` filter activates ``Stability``, a pin whose
inputs carry an infinity activates ``Non-finite input``, a pin whose primary window resolves to ``1`` activates
``window == 1`` — never sniffed from prose. The three genuinely semantic regimes a number alone cannot tell apart
(a ``Domain`` violation, an ``Insufficient sample``, and a ``Degenerate denominator``, which all read as the same
``NaN`` / ``inf`` / ``0``) are keyed off the pin's snake_case label, the one declared vocabulary that names the
regime — the same labels the suite freezes to snake_case so a category can never silently vanish.

The wording of each bullet is NOT enforced here; a bullet's *phrasing* is the author's, checked by the docstring
generator's round-trip, not by a frozen phrase list. This module answers three questions only: which classes
a declaration demands (:func:`required_classes`), which cases the Examples block must demonstrate
(:func:`scenario_witnesses`), and which degenerate outcomes a Degenerate-denominator bullet's pins witness
(:func:`degenerate_witness_kinds` vs :func:`asserted_outcome_kinds`, the claim-pin link).
"""

import math
import re
from collections.abc import Iterator, Mapping
from enum import Enum
from typing import cast

from tests.support.declaration import Declaration, Pin

__all__ = (
    "EdgeClass",
    "asserted_outcome_kinds",
    "degenerate_witness_kinds",
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


# The snake_case pin-label markers that name each of the three semantic regimes a number alone cannot distinguish.
# A pin whose label carries one of a class's markers is that class's candidate witness; the pin's expected lanes must
# then also show a degenerate outcome (null / NaN / an exact 0 / an infinity) for the class to activate — an ordinary
# finite outcome under a matching label is a definedness witness, not an edge case. The ``window == 1`` and
# ``Non-finite input`` classes are NOT here: both are read from the pin's declared data (its resolved window and its
# input lanes), never from its label.
_PIN_MARKERS: Mapping[EdgeClass, tuple[str, ...]] = {
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
        "zero_close",
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
}

# Classes a docstring carries although no pin outcome demands them — a deliberate, noteworthy defined behavior: the
# parametric VaR collapses onto the mean on a zero dispersion, so its ``z * sigma`` term vanishes without any division
# and no pin can witness a degenerate denominator. Keyed by function name; shrink-only.
_ALSO_REQUIRED: frozenset[tuple[str, str]] = frozenset(
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


def _degenerate_outcome(expected: object) -> bool:
    """Whether a pin's expected lanes show a degenerate outcome: a null, a ``NaN``, an exact ``0``, or an infinity."""
    return any(
        scalar is None or math.isnan(scalar) or math.isinf(scalar) or scalar == 0.0
        for scalar in _iter_scalars(expected)
    )


def _matches(label: str, edge_class: EdgeClass) -> bool:
    return any(marker in label for marker in _PIN_MARKERS[edge_class])


def _primary_window(declaration: Declaration, pin: Pin) -> int | None:
    """The pin's effective primary lookback: the named ``window`` parameter (or the ``window`` key), under the pin's
    overrides — the declared datum the ``window == 1`` class reads instead of the pin's label. ``None`` when the
    function has no single primary window (a multi-window indicator, an unwindowed transform).
    """
    params = {**declaration.params, **pin.params_override}
    if declaration.window is not None and declaration.window in params:
        value = params[declaration.window]
    elif "window" in params:
        value = params["window"]
    else:
        return None
    return int(value) if isinstance(value, int) else None


def _pin_classes(declaration: Declaration) -> set[EdgeClass]:
    """The conditional classes a declaration's pins activate: an infinite input, a window-one lookback, or a
    degenerate outcome under a semantic-regime marker; a domain-marked pin is a domain witness only.
    """
    active: set[EdgeClass] = set()
    for pin in declaration.pins:
        if _has_inf(pin.inputs):
            active.add(EdgeClass.NON_FINITE_INPUT)
            continue
        if _primary_window(declaration, pin) == 1:
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


def required_classes(declaration: Declaration) -> tuple[EdgeClass, ...]:
    """
    The exact, ordered bullet classes ``declaration`` demands: Null, NaN, and Partitioning always; each conditional
    class exactly when the declaration's data activates it (a pin, or the conditioning filter); plus the sanctioned
    ``_ALSO_REQUIRED`` entries.
    """
    active = _pin_classes(declaration)
    if declaration.conditioning is not None:
        active.add(EdgeClass.STABILITY)
    active.update(EdgeClass(value) for name, value in _ALSO_REQUIRED if name == declaration.name)
    active.update((EdgeClass.NULL, EdgeClass.NAN, EdgeClass.PARTITIONING))
    return tuple(edge_class for edge_class in EdgeClass if edge_class in active)


# The degenerate outcome tokens a Degenerate-denominator bullet may assert, and a token's kind. A token a clause
# negates with "cannot" / "spurious" (a residue explicitly ruled OUT, e.g. a guarded ``+/-inf``) is not an assertion
# and needs no pin; every asserted ``NaN`` or infinity does (the claim-pin link).
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


def degenerate_witness_kinds(declaration: Declaration) -> set[str]:
    """The outcome kinds the declaration's Degenerate-denominator pins witness: every pin whose label matches a
    degenerate-denominator marker contributes its expected kinds (a pin with an infinity in its INPUTS is a
    non-finite-input witness instead, so it is skipped).
    """
    kinds: set[str] = set()
    for pin in declaration.pins:
        if _has_inf(pin.inputs):
            continue
        if _matches(pin.label, EdgeClass.DEGENERATE_DENOMINATOR):
            kinds |= _pin_kinds(pin.expected)
    return kinds


# The classes whose activation demands a demonstration scenario in the Examples block: Null, NaN, and Partitioning
# ride the three canonical usage scenarios every function opens with, and Stability states a precision regime with no
# byte-exact outcome a doctest could print.
_SCENARIO_CLASSES: tuple[EdgeClass, ...] = (
    EdgeClass.DOMAIN,
    EdgeClass.INSUFFICIENT_SAMPLE,
    EdgeClass.DEGENERATE_DENOMINATOR,
    EdgeClass.NON_FINITE_INPUT,
    EdgeClass.WINDOW_ONE,
)

# The fixed order the Degenerate-denominator kind collapse walks, so the demanded scenario sequence is deterministic.
_DEGENERATE_KIND_ORDER: tuple[str, ...] = ("null", "nan", "inf", "zero")


def _class_witnesses(declaration: Declaration, edge_class: EdgeClass) -> list[Pin]:
    """The pins that witness ``edge_class`` for an Examples demonstration, in declaration order — the ``_pin_classes``
    triage with the domain-marked pins reserved to the Domain class.
    """
    witnesses: list[Pin] = []
    for pin in declaration.pins:
        if _has_inf(pin.inputs):
            if edge_class is EdgeClass.NON_FINITE_INPUT:
                witnesses.append(pin)
            continue
        if edge_class is EdgeClass.WINDOW_ONE and _primary_window(declaration, pin) == 1:
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


def scenario_witnesses(declaration: Declaration) -> tuple[tuple[EdgeClass, str, frozenset[str]], ...]:
    """
    The ordered edge scenarios ``declaration`` demands of its Examples block, as ``(class, pin label, outcome kinds)``:
    for each conditional class the declaration's pins activate, one demonstration per witnessing pin — collapsed, for
    the Degenerate-denominator class, to one scenario per distinct witnessed outcome kind (``null`` / ``nan`` / ``inf``
    / ``zero``), each carried by the first pin in declaration order that shows it. A required class none of whose
    witnesses shows a degenerate outcome (the parametric VaR's mean collapse) demands one scenario from its first
    witness, with no asserted kind.
    """
    active = set(required_classes(declaration))
    demanded: list[tuple[EdgeClass, str, frozenset[str]]] = []
    for edge_class in _SCENARIO_CLASSES:
        if edge_class not in active:
            continue
        pins = _class_witnesses(declaration, edge_class)
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
