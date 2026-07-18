"""
The single failure formatter every derived check raises through.

A derived failure is only useful if it says, in one glance, three things: what the function was fed, how it disagreed,
and whether to suspect the declaration or the code. This module builds that message — the declaration that generated the
check, the exact probe frame printed whole, the expected (oracle) and observed (factory) lanes with the first
divergence pointed at, a triage line that names the declared axis and suggests the value that would have matched, and a
copy-pasteable reproduction. Pure string building: no comparison logic, no Polars evaluation.
"""

import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass

from tests.support.declaration import Declaration
from tests.support.synthesis import Probe


@dataclass(frozen=True)
class Disagreement:
    """A located lane disagreement: the lane name, the two lanes, and the first index they differ at."""

    lane: str
    expected: Sequence[float | None]
    observed: Sequence[float | None]
    index: int


def _render_one(value: float | None | str) -> str:
    """Render one lane value, spelling out ``None`` / ``nan`` / ``±inf`` so the message is unambiguous."""
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0.0 else "-inf"
    return repr(value)


def _render_lane(values: Sequence[float | None]) -> str:
    """Render a whole lane as a bracketed, comma-separated list of :func:`_render_one` values."""
    return "[" + ", ".join(_render_one(value) for value in values) + "]"


def triage_for_enum(function_name: str, declared: enum.Enum) -> str:
    """
    The triage line for a declared enum axis: name the declaration, suggest the value that would have matched, and pose
    the two suspects.

    Args:
        function_name: The function under test, named as the second suspect.
        declared: The enum member the failing check was gating on (a behavior, a non-finite flow, ...).

    Returns:
        A one-line triage: ``you declared X. Either the declaration is wrong (did you mean Y?) or f has a bug.``
    """
    others = [member for member in type(declared) if member is not declared]
    label = f"{type(declared).__name__}.{declared.name}"
    if not others:
        hint = "this axis allows no other value"
    elif len(others) == 1:
        other = others[0]
        hint = f"did you mean {type(other).__name__}.{other.name}?"
    else:
        names = ", ".join(f"{type(member).__name__}.{member.name}" for member in others)
        hint = f"did you mean one of {names}?"
    return f"you declared {label}. Either the declaration is wrong ({hint}) or {function_name} has a bug."


def triage_generic(function_name: str, subject: str) -> str:
    """The triage line for a check with no single enum axis (a golden, a pin, an oracle band)."""
    return f"either the {subject} is wrong or {function_name} has a bug."


def describe_failure(
    *,
    declaration: Declaration,
    check: str,
    probe: Probe,
    disagreement: Disagreement,
    triage: str,
    expected_label: str = "oracle",
) -> str:
    """
    Assemble the full failure message: header, the printed probe, the two lanes with the first divergence, the triage,
    and the reproduction snippet.

    Args:
        declaration: The declaration that generated the failing check (named in the header).
        check: The check's name (e.g. ``check_behavior_nan``).
        probe: The synthesized probe (its frame, description, and reproduction snippet).
        disagreement: The located lane disagreement.
        triage: The triage line (from :func:`triage_for_enum` or :func:`triage_generic`).
        expected_label: What the expected lane IS (``oracle`` by default; a declared value names itself).

    Returns:
        The multi-line message a rung raises as its ``AssertionError``.
    """
    expected_value = (
        disagreement.expected[disagreement.index] if disagreement.index < len(disagreement.expected) else "<missing>"
    )
    observed_value = (
        disagreement.observed[disagreement.index] if disagreement.index < len(disagreement.observed) else "<missing>"
    )
    lines = [
        f"{declaration.name} ({declaration.family}): {check} disagreed.",
        "",
        f"Probe — {probe.description}:",
        str(probe.frame),
        "",
        f"lane {disagreement.lane!r}:",
        f"  expected ({expected_label}): {_render_lane(disagreement.expected)}",
        f"  observed (factory): {_render_lane(disagreement.observed)}",
        f"  first divergence at index {disagreement.index}: "
        f"expected {_render_one(expected_value)} vs observed {_render_one(observed_value)}",
        "",
        f"Triage: {triage}",
        "",
        "Reproduce:",
        *(f"  {line}" for line in probe.snippet),
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class FlowEvidence:
    """The structural-flow evidence: the violating lane, its clean and poisoned runs, and the injected row."""

    lane: str
    lane_clean: list[float | None]
    lane_poisoned: list[float | None]
    row: int


def describe_flow_violation(
    *,
    declaration: Declaration,
    check: str,
    probe: Probe,
    declared: enum.Enum,
    violation: str,
    evidence: FlowEvidence,
) -> str:
    """
    Assemble the structural-flow failure message: the declared behavior's shape was not observed, even though the
    factory and the oracle agree by value — the declaration itself is under test here.

    Args:
        declaration: The declaration that generated the failing check.
        check: The check's name (e.g. ``check_behavior_nan``).
        probe: The synthesized poisoned probe.
        declared: The declared behavior enum member the shape was derived from.
        violation: The one-line description of the shape violation.
        evidence: The violating lane with its clean and poisoned runs and the injected row.

    Returns:
        The multi-line message the rung raises as its ``AssertionError``.
    """
    lines = [
        f"{declaration.name} ({declaration.family}): {check} — the declared behavior's shape was not observed.",
        "",
        f"Probe — {probe.description}:",
        str(probe.frame),
        "",
        f"lane {evidence.lane!r} (factory output, clean vs poisoned; injection at row {evidence.row}):",
        f"  clean:    {_render_lane(evidence.lane_clean)}",
        f"  poisoned: {_render_lane(evidence.lane_poisoned)}",
        "",
        f"Shape violation: {violation}",
        "",
        triage_for_enum(declaration.name, declared),
        "",
        "Reproduce:",
        *(f"  {line}" for line in probe.snippet),
    ]
    return "\n".join(lines)
