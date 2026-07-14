"""
Registry-derived sweeps of the declaration surface itself: the coverage facts a single spec cannot see.

Four guards, all swept over ``ALL_SPECS`` so a new function is held to them the moment its spec lands. First, every
scalar parameter that carries a signature default must be exercised at a non-default value somewhere in the
declarations — ``params``, ``golden_params``, or a pin's ``params_override`` — because every probe, golden, oracle
comparison, and fuzz draw builds its call from those three sources: a defaulted knob no declaration varies is a live
code path no tier ever observes (the gap that once left ``variance_ewma``'s ``adjust`` / ``bias`` branches untested).
Second, a validation chain over ordered windows must declare one ``raises`` counterexample per boundary: a chained
``a <= b <= c`` check short-circuits, so a single counterexample can leave the later clause with no born-red witness
(the gap that once left ``ichimoku``'s ``kijun <= senkou`` clause unwitnessed). Third, a ``ScaleExempt`` declaration
must be genuinely exempt: rescaling the probe by 4 must NOT reproduce an integer-degree homogeneity on informative
values, or the exemption is hiding a derivable axis and silently skipping the scale rung (the gap that once hid the
``treynor`` pair's and ``modigliani``'s degree-1 axes at their zero-rate defaults). Fourth, a declared ``ScaleAxis``
must be exercised on a non-trivial output: a probe whose every lane is zero or undefined turns the scale rung into
``0 == 0`` (the gap that once left ``cost_borrow``'s short-only notional unverified under rescaling).

Each ratchet below is frozen at its guard's introduction and may only shrink: an entry whose site gains coverage
turns stale and fails loudly (the same fail-closed shape as ``test_docstrings``' pinned deviants), and a new
function cannot add itself to one without editing this file.
"""

import inspect
import math

import polars as pl
import pytest
from tests.all_specs import ALL_SPECS
from tests.support import RELATIVE_TOLERANCE_SCALE
from tests.support.spec import ScaleExempt, Spec, actual_lanes, probe_frame, spec_id, widest_warmup

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


# Values whose magnitude clears this floor carry real scale information: a smaller base is indistinguishable from an
# exact zero at the scale band, and a zero satisfies every degree at once.
_INFORMATIVE_FLOOR = 1e-6

# Specs whose every probe lane sits at or under the informative floor, frozen at this guard's introduction with the
# reason the (near-)zero is CORRECT there: the strictly-rising equity probe has zero drawdown by definition (the
# whole drawdown family reduces to exactly 0.0 / a zero-length tail), the monotone-up bars have no down-moves (the
# minus legs of the directional-movement pair smooth an all-zero series), and the alternating two-value returns
# probe is symmetric about its mean, so the skewness pair's true answer is 0 (computed as an exact 0.0 or a
# platform-dependent ~1e-16 residual — which is why this guard compares against the floor, never against the bit).
# Their scale rungs are vacuous on the probe; their value correctness lives in the goldens, oracles, and pins, which
# all run varied inputs. De-vacuating them needs a drawdown-bearing / down-moving / asymmetric probe path — a
# deliberate engine change, not an exemption edit. Shrink only.
_ZERO_ON_THE_PROBE: frozenset[str] = frozenset(
    {
        "conditional_drawdown_at_risk",
        "di_minus",
        "dm_minus",
        "drawdown",
        "drawdown_rolling",
        "max_drawdown",
        "max_drawdown_duration",
        "pain_index",
        "skewness",
        "skewness_rolling",
        "ulcer_index",
    }
)


def _probe_lanes(spec: Spec, factor: float) -> dict[str, list[float | None]]:
    """The spec's output lanes on the deterministic probe, with every input column scaled by ``factor``."""
    frame = probe_frame(spec.inputs, widest_warmup(spec) + 12)
    if factor != 1.0:
        frame = frame.select(pl.col(column) * factor for column in frame.columns)
    return actual_lanes(spec, frame)


@pytest.mark.parametrize("spec", [s for s in ALL_SPECS if isinstance(s.scale, ScaleExempt)], ids=spec_id)
def test_scale_exemption_is_not_secretly_homogeneous(spec: Spec) -> None:
    """Verifies a ScaleExempt spec really breaks every integer-degree homogeneity at its default params."""
    base = _probe_lanes(spec, 1.0)
    scaled = _probe_lanes(spec, 4.0)
    for degree in (0, 1, 2):
        informative = 0
        holds = True
        for name, base_values in base.items():
            for value_base, value_scaled in zip(base_values, scaled[name], strict=True):
                if value_base is None or value_scaled is None:
                    continue
                if not (math.isfinite(value_base) and math.isfinite(value_scaled)):
                    continue
                if abs(value_base) <= _INFORMATIVE_FLOOR:
                    continue
                informative += 1
                if not math.isclose(value_scaled, value_base * 4.0**degree, rel_tol=RELATIVE_TOLERANCE_SCALE):
                    holds = False
                    break
            if not holds:
                break
        assert not (holds and informative >= 1), (
            f"{spec.name}: declared ScaleExempt but the probe rescale reproduces degree-{degree} homogeneity on "
            f"{informative} informative value(s) — the exemption hides a derivable axis; declare the ScaleAxis "
            f"(scoped to the default params if need be) instead of skipping the scale rung"
        )


@pytest.mark.parametrize("spec", [s for s in ALL_SPECS if not isinstance(s.scale, ScaleExempt)], ids=spec_id)
def test_scale_probe_output_is_not_all_zero(spec: Spec) -> None:
    """Verifies a declared ScaleAxis meets at least one informative probe value, or sits in the shrinking ratchet."""
    lanes = _probe_lanes(spec, 1.0)
    # Compare against the informative floor, never the bit: a lane whose true answer is zero comes back as an exact
    # 0.0 on one platform and a ~1e-16 accumulation residual on another (the skewness pair), and both mean "zero".
    informative = any(
        value is not None and math.isfinite(value) and abs(value) > _INFORMATIVE_FLOOR
        for values in lanes.values()
        for value in values
    )
    if spec.name in _ZERO_ON_THE_PROBE:
        assert not informative, (
            f"{spec.name}: now informative on the probe — remove its stale entry from _ZERO_ON_THE_PROBE"
        )
    else:
        assert informative, (
            f"{spec.name}: every probe lane sits at zero (or under the informative floor), so the scale rung "
            f"compares 0 with 0 and verifies nothing — give the probe a branch-exercising path or extend "
            f"_ZERO_ON_THE_PROBE with a written reason"
        )
