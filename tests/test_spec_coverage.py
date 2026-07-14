"""
Registry-derived sweeps of the declaration surface itself: the coverage facts a single spec cannot see.

Two guards, both swept over ``ALL_SPECS`` so a new function is held to them the moment its spec lands. First, every
scalar parameter that carries a signature default must be exercised at a non-default value somewhere in the
declarations — ``params``, ``golden_params``, or a pin's ``params_override`` — because every probe, golden, oracle
comparison, and fuzz draw builds its call from those three sources: a defaulted knob no declaration varies is a live
code path no tier ever observes (the gap that once left ``variance_ewma``'s ``adjust`` / ``bias`` branches untested).
Second, a validation chain over ordered windows must declare one ``raises`` counterexample per boundary: a chained
``a <= b <= c`` check short-circuits, so a single counterexample can leave the later clause with no born-red witness
(the gap that once left ``ichimoku``'s ``kijun <= senkou`` clause unwitnessed).

The ratchet below lists the knobs whose non-default branch predates these guards and is not yet pinned. It may only
shrink: an entry whose knob gains coverage turns stale and fails loudly (the same fail-closed shape as
``test_docstrings``' pinned deviants), and a new function cannot add itself to it without editing this file.
"""

import inspect

import pytest
from tests.all_specs import ALL_SPECS
from tests.support.spec import Spec, spec_id

# Knobs with a signature default that no declaration varies yet, frozen at this guard's introduction: the scale /
# smoothing constants of the sequential kernels (their golden masters run the canonical defaults), the risk-free-rate
# legs whose zero default is itself the canonical convention, the tail-confidence levels (validated by ``raises``,
# exercised only at 0.95), and sterling's excess cushion (whose pins exist precisely to show the DEFAULT cushion's
# behavior). Shrink this set by pinning a non-default case; never grow it.
_UNVARIED: frozenset[tuple[str, str]] = frozenset(
    {
        ("adjusted_sharpe_ratio", "risk_free_rate"),
        ("alpha_rolling", "risk_free_rate"),
        ("burke_ratio", "risk_free_rate"),
        ("keltner_channels", "multiplier"),
        ("mama", "limit_fast"),
        ("mama", "limit_slow"),
        ("pain_ratio", "risk_free_rate"),
        ("parabolic_sar", "acceleration"),
        ("parabolic_sar", "maximum"),
        ("probabilistic_sharpe_ratio", "risk_free_rate"),
        ("sharpe_ratio", "risk_free_rate"),
        ("sharpe_ratio_rolling", "risk_free_rate"),
        ("sterling_ratio", "excess"),
        ("sterling_ratio", "risk_free_rate"),
        ("t3", "volume_factor"),
        ("ulcer_performance_ratio", "risk_free_rate"),
        ("value_at_risk", "confidence"),
        ("value_at_risk_modified", "confidence"),
        ("value_at_risk_parametric", "confidence"),
        ("value_at_risk_rolling", "confidence"),
    }
)


def _defaulted_scalars(spec: Spec) -> list[tuple[str, object]]:
    """Every non-``Expr`` parameter of the factory that carries a signature default, with that default."""
    return [
        (parameter.name, parameter.default)
        for parameter in inspect.signature(spec.factory).parameters.values()
        if parameter.default is not inspect.Parameter.empty and "Expr" not in str(parameter.annotation)
    ]


def _varies(spec: Spec, name: str, default: object) -> bool:
    """Whether any declaration source sets ``name`` to a non-default value."""
    if spec.params.get(name, default) != default or spec.golden_params.get(name, default) != default:
        return True
    return any(pin.params_override.get(name, default) != default for pin in spec.pins)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_every_defaulted_scalar_is_varied_somewhere(spec: Spec) -> None:
    """Verifies each defaulted scalar knob is exercised at a non-default value, or sits in the shrinking ratchet."""
    for name, default in _defaulted_scalars(spec):
        varied = _varies(spec, name, default)
        if (spec.name, name) in _UNVARIED:
            assert not varied, f"{spec.name}.{name}: now varied — remove its stale entry from _UNVARIED"
        else:
            assert varied, (
                f"{spec.name}.{name}: a defaulted scalar no declaration varies — no tier ever leaves the default "
                f"branch; add a golden_params or pin params_override case (or, if truly unreachable, extend "
                f"_UNVARIED with a written reason)"
            )


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_every_ordering_boundary_has_a_counterexample(spec: Spec) -> None:
    """Verifies an ordered-window chain declares one raises counterexample per ``<=`` boundary it validates."""
    ordering = [match for _, match in spec.raises if "ordered" in match]
    if not ordering:
        return
    boundaries = max(match.count("<=") for match in ordering)
    assert len(ordering) >= boundaries, (
        f"{spec.name}: the ordering chain has {boundaries} boundaries but only {len(ordering)} counterexample(s) — "
        f"a chained check short-circuits, so each boundary needs its own born-red witness"
    )
