"""
The generic checks, each a plain function taking a :class:`Declaration`.

There is no inheritance and no capability mixin: a check reads the declared fields it needs and, where the declaration
does not activate it (no golden, no pins, a scale-exempt function), skips cleanly with a reason. The correctness core
compares the factory against the naive oracle — on the deterministic probe, on fuzzed frames, and (the severity upgrade
over hand-pinned cases) on the synthesized degenerate regimes too, comparing VALUES against the oracle rather than only
the kind of outcome. Every value disagreement raises through :mod:`tests_new.support.messages`, so a failure names the
declaration that generated it, prints the tiny probe whole, shows expected vs observed, triages the two suspects, and
carries a copy-pasteable reproduction.
"""

import enum
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import cast

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from tests_new.support import messages, synthesis
from tests_new.support.compare import first_mismatch
from tests_new.support.declaration import (
    Declaration,
    Pin,
    ScaleExempt,
    Shape,
    actual_lanes,
    build_expr,
    horizon,
    lane_series,
    reference_lanes,
    widest_warmup,
)
from tests_new.support.frames import count_leading_nulls, probe_frame
from tests_new.support.synthesis import Probe, fuzz_frames
from tests_new.support.tolerances import (
    TOLERANCE_ABSOLUTE_EXACT,
    TOLERANCE_ABSOLUTE_PROPERTY,
    TOLERANCE_ABSOLUTE_REFERENCE,
    TOLERANCE_FACTOR_EXACT,
    TOLERANCE_RELATIVE_EXACT,
    TOLERANCE_RELATIVE_PROPERTY,
    TOLERANCE_RELATIVE_REFERENCE,
    TOLERANCE_RELATIVE_SCALE,
    input_scale,
)

_SCALE_FACTOR = 4.0
_Lanes = dict[str, list[float | None]]


def _oracle_bands(declaration: Declaration, rel_default: float, abs_default: float) -> tuple[float, float]:
    """The oracle-agreement band: the declaration's override where set, else the tier default."""
    rel = declaration.oracle_rel_tol if declaration.oracle_rel_tol is not None else rel_default
    abs_ = declaration.oracle_abs_tol if declaration.oracle_abs_tol is not None else abs_default
    return rel, abs_


def _assert_lanes(
    declaration: Declaration,
    check: str,
    probe: Probe,
    *,
    expected: _Lanes,
    actual: _Lanes,
    rel_tol: float,
    abs_tol: float,
    triage: str,
) -> None:
    """Assert the named lanes agree; on the first disagreement raise the rich, triaged failure message."""
    if sorted(actual) != sorted(expected):
        msg = f"{declaration.name}: {check} produced lanes {sorted(actual)}, the oracle has {sorted(expected)}"
        raise AssertionError(msg)
    for name in sorted(expected):
        index = first_mismatch(actual[name], expected[name], rel_tol=rel_tol, abs_tol=abs_tol)
        if index is not None:
            disagreement = messages.Disagreement(lane=name, expected=expected[name], observed=actual[name], index=index)
            raise AssertionError(
                messages.describe_failure(
                    declaration=declaration, check=check, probe=probe, disagreement=disagreement, triage=triage
                )
            )


# ======================================================================================================================
# Correctness against the oracle
# ======================================================================================================================


def _assert_reference(declaration: Declaration, probe: Probe, rel_tol: float, abs_tol: float) -> None:
    """The shared oracle-agreement body: the factory's lanes match the oracle's on ``probe.frame`` within tolerance."""
    expected = reference_lanes(declaration, probe.frame)
    actual = actual_lanes(declaration, probe.frame)
    triage = messages.triage_generic(declaration.name, "oracle or the declaration")
    _assert_lanes(
        declaration,
        "check_oracle_agreement",
        probe,
        expected=expected,
        actual=actual,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        triage=triage,
    )


def _check_property(declaration: Declaration, *, missing: bool) -> None:
    """Fuzz the input frame and hold the factory to the oracle at the property tier (``@given`` inside the check)."""
    rel, abs_ = _oracle_bands(declaration, TOLERANCE_RELATIVE_PROPERTY, TOLERANCE_ABSOLUTE_PROPERTY)
    label = "a fuzzed frame with interior null / NaN" if missing else "a fuzzed frame"

    @given(data=st.data())
    def run(data: st.DataObject) -> None:
        frame = data.draw(fuzz_frames(declaration, missing=missing))
        if declaration.conditioning is not None:
            assume(declaration.conditioning(frame))
        _assert_reference(declaration, synthesis.describe(declaration, frame, label), rel, abs_)

    run()


def check_oracle_agreement(declaration: Declaration) -> None:
    """The factory reproduces its oracle — on the deterministic probe and over the fuzz domain (both missing modes)."""
    rel, abs_ = _oracle_bands(declaration, TOLERANCE_RELATIVE_REFERENCE, TOLERANCE_ABSOLUTE_REFERENCE)
    frame = probe_frame(declaration.inputs, widest_warmup(declaration) + 12)
    _assert_reference(declaration, synthesis.describe(declaration, frame, "the deterministic probe frame"), rel, abs_)
    _check_property(declaration, missing=False)
    _check_property(declaration, missing=True)


def check_golden(declaration: Declaration) -> None:
    """The frozen golden master, rounded expression-side so it can never flake cross-platform."""
    golden = declaration.golden
    if golden is None:
        pytest.skip(f"{declaration.name}: no golden master declared")
    frame = pl.DataFrame({role: pl.Series(list(values), dtype=pl.Float64) for role, values in golden.inputs.items()})
    out = frame.select(build_expr(declaration, **golden.params).alias("out"))
    probe = synthesis.describe(declaration, frame, "the golden master input")
    triage = messages.triage_generic(declaration.name, "golden master")
    schema = out.schema["out"]
    expected_output = golden.output
    if isinstance(schema, pl.Struct):
        assert isinstance(expected_output, Mapping)  # guaranteed by the golden-shape constructor check
        actual = {
            field.name: out["out"].struct.field(field.name).round(golden.round_to).to_list() for field in schema.fields
        }
        expected = {str(name): list(values) for name, values in expected_output.items()}
    else:
        assert not isinstance(expected_output, Mapping)
        actual = {"out": out["out"].round(golden.round_to).to_list()}
        expected = {"out": list(expected_output)}
    _assert_lanes(
        declaration,
        "check_golden",
        probe,
        expected=expected,
        actual=actual,
        rel_tol=TOLERANCE_RELATIVE_EXACT,
        abs_tol=TOLERANCE_ABSOLUTE_EXACT,
        triage=triage,
    )


def _assert_signs(actual: _Lanes, expected: _Lanes) -> None:
    """When a pin marks the sign, compare ``copysign`` so ``-0.0`` and ``0.0`` are distinguished."""
    for name, expected_lane in expected.items():
        for got, want in zip(actual[name], expected_lane, strict=True):
            if got is not None and want is not None:
                assert math.copysign(1.0, got) == math.copysign(1.0, want), f"{name}: sign mismatch {got} vs {want}"


def _check_one_pin(declaration: Declaration, pin: Pin) -> None:
    """One crafted-input case: the pinned frame maps to the pinned lanes (rounded and sign-checked where declared)."""
    frame = pl.DataFrame({role: pl.Series(list(values), dtype=pl.Float64) for role, values in pin.inputs.items()})
    lanes = lane_series(frame.select(build_expr(declaration, **pin.params_override).alias("out")))
    if pin.round_to is not None:  # platform-dependent transcendental lanes compare at the declared rounding
        lanes = [lane.round(pin.round_to) for lane in lanes]
    probe = synthesis.describe(declaration, frame, f"the pinned case {pin.label!r}")
    triage = messages.triage_generic(declaration.name, f"pinned case {pin.label!r}")
    if isinstance(pin.expected, Mapping):
        actual = {lane.name: lane.to_list() for lane in lanes}
        expected = {str(name): list(values) for name, values in pin.expected.items()}
    else:
        (single,) = lanes  # a non-struct output is exactly one lane
        actual = {"out": single.to_list()}
        expected = {"out": list(pin.expected)}
    _assert_lanes(
        declaration,
        "check_pins",
        probe,
        expected=expected,
        actual=actual,
        rel_tol=TOLERANCE_RELATIVE_EXACT,
        abs_tol=TOLERANCE_ABSOLUTE_EXACT,
        triage=triage,
    )
    if pin.signed:
        _assert_signs(actual, expected)


def check_pins(declaration: Declaration) -> None:
    """Every crafted-input case: the exact hand-computed values the synthesis and the oracle cannot derive."""
    if not declaration.pins:
        pytest.skip(f"{declaration.name}: no pinned cases declared")
    for pin in declaration.pins:
        _check_one_pin(declaration, pin)


# ======================================================================================================================
# Missing-data and non-finite flow
# ======================================================================================================================


def _lane_value_same(left: float | None, right: float | None) -> bool:
    """Exact lane equality, ``None``-aware and ``NaN``-aware (a latch compares equal to a latch)."""
    if left is None or right is None:
        return left is None and right is None
    if math.isnan(left) and math.isnan(right):
        return True
    return left == right


def _lane_is_nan(value: float | None) -> bool:
    return value is not None and math.isnan(value)


def _null_propagates(clean: list[float | None], poisoned: list[float | None], row: int, reach: int) -> str | None:
    if clean[row] is not None and poisoned[row] is not None:
        return f"declared PROPAGATES: the injected row {row} must be null"
    for j in range(row + reach + 1, len(poisoned)):
        if not _lane_value_same(poisoned[j], clean[j]):
            return (
                f"declared PROPAGATES: beyond the horizon (row {row} + {reach}) the lane must equal the clean run; "
                f"row {j} differs"
            )
    return None


def _null_bridged(clean: list[float | None], poisoned: list[float | None], row: int, reach: int) -> str | None:
    del reach
    if clean[row] is not None and poisoned[row] is not None:
        return f"declared BRIDGED: the injected row {row} must be null"
    tail = range(row + 1, len(poisoned))
    if not any(poisoned[j] is not None for j in tail):
        return f"declared BRIDGED: the flow must resume after row {row}, but every later lane is null"
    if all(_lane_value_same(poisoned[j], clean[j]) for j in tail):
        return (
            f"declared BRIDGED: the flow resumes identical to the clean run after row {row} — the missing bar's "
            "contribution was not absorbed; that shape is PROPAGATES"
        )
    return None


def _nan_latches(clean: list[float | None], poisoned: list[float | None], row: int, reach: int) -> str | None:
    del reach
    bad = [j for j in range(row, len(poisoned)) if clean[j] is not None and not _lane_is_nan(poisoned[j])]
    if bad:
        return f"declared LATCHES: every defined lane from row {row} on must be NaN; rows {bad[:4]} are not"
    return None


def _nan_propagates(clean: list[float | None], poisoned: list[float | None], row: int, reach: int) -> str | None:
    if clean[row] is not None and not _lane_is_nan(poisoned[row]):
        return f"declared PROPAGATES: the injected row {row} must be NaN"
    beyond = range(row + reach + 1, len(poisoned))
    bad = [j for j in beyond if _lane_is_nan(poisoned[j]) and not _lane_is_nan(clean[j])]
    if bad:
        return (
            f"declared PROPAGATES: beyond the horizon (row {row} + {reach}) no new NaN may appear; rows {bad[:4]} "
            "are contaminated"
        )
    return None


_FlowHandler = Callable[["list[float | None]", "list[float | None]", int, int], "str | None"]
_SHAPES_NULL: Mapping[str, _FlowHandler] = {"PROPAGATES": _null_propagates, "BRIDGED": _null_bridged}
_SHAPES_NAN: Mapping[str, _FlowHandler] = {"PROPAGATES": _nan_propagates, "LATCHES": _nan_latches}


def _flow_violation(
    shapes: Mapping[str, _FlowHandler],
    member: str,
    clean: list[float | None],
    poisoned: list[float | None],
    row: int,
    reach: int,
) -> str | None:
    """The declared behavior's structural shape, checked on the factory's clean-vs-poisoned lanes — fail-closed:
    a behavior member with no registered shape is a hole in the contract, never a silent pass.
    """
    handler = shapes.get(member)
    if handler is None:
        return f"no structural contract for behavior {member!r} — extend the flow shapes in rungs.py"
    return handler(clean, poisoned, row, reach)


def _assert_flow(
    declaration: Declaration,
    check: str,
    pair: synthesis.FlowProbe,
    *,
    declared: enum.Enum,
    shapes: Mapping[str, _FlowHandler],
) -> None:
    """A flow probe checked on BOTH layers: factory-vs-oracle by value, and the declared behavior's shape.

    The value layer proves the code and the naive reimplementation agree on the regime; the shape layer proves the
    DECLARATION tells the truth about it — a wrong behavior enum goes red here even when factory and oracle agree.
    """
    rel, abs_ = _oracle_bands(declaration, TOLERANCE_RELATIVE_REFERENCE, TOLERANCE_ABSOLUTE_REFERENCE)
    expected = reference_lanes(declaration, pair.probe.frame)
    actual = actual_lanes(declaration, pair.probe.frame)
    triage = messages.triage_for_enum(declaration.name, declared)
    _assert_lanes(
        declaration, check, pair.probe, expected=expected, actual=actual, rel_tol=rel, abs_tol=abs_, triage=triage
    )
    if declaration.shape is not Shape.SERIES:
        message = f"{declaration.name}: structural flow contracts for {declaration.shape} land with their family"
        raise NotImplementedError(message)
    clean = actual_lanes(declaration, pair.frame_clean)
    reach = horizon(declaration)
    for name, lane_poisoned in actual.items():
        violation = _flow_violation(shapes, declared.name, clean[name], lane_poisoned, pair.row, reach)
        if violation is not None:
            raise AssertionError(
                messages.describe_flow_violation(
                    declaration=declaration,
                    check=check,
                    probe=pair.probe,
                    declared=declared,
                    violation=violation,
                    evidence=messages.FlowEvidence(name, clean[name], lane_poisoned, pair.row),
                )
            )


def check_behavior_null(declaration: Declaration) -> None:
    """An interior ``null`` plays out as the oracle plays it out AND as the declared null behavior promises."""
    _assert_flow(
        declaration,
        "check_behavior_null",
        synthesis.frame_flow_null(declaration),
        declared=declaration.behavior_null,
        shapes=_SHAPES_NULL,
    )


def check_behavior_nan(declaration: Declaration) -> None:
    """An interior ``NaN`` plays out as the oracle plays it out AND as the declared NaN behavior promises."""
    _assert_flow(
        declaration,
        "check_behavior_nan",
        synthesis.frame_flow_nan(declaration),
        declared=declaration.behavior_nan,
        shapes=_SHAPES_NAN,
    )


def check_nonfinite(declaration: Declaration) -> None:
    """Each input carrying ``±inf`` flows through exactly as the oracle carries it — the declared IEEE behavior."""
    triage = messages.triage_for_enum(declaration.name, declaration.nonfinite)
    for probe in synthesis.frame_infinite_input(declaration):
        expected = reference_lanes(declaration, probe.frame)
        actual = actual_lanes(declaration, probe.frame)
        _assert_lanes(
            declaration,
            "check_nonfinite",
            probe,
            expected=expected,
            actual=actual,
            rel_tol=TOLERANCE_RELATIVE_REFERENCE,
            abs_tol=TOLERANCE_ABSOLUTE_REFERENCE,
            triage=triage,
        )


# ======================================================================================================================
# Properties, edges, and validation
# ======================================================================================================================


def _homogeneous(base: Sequence[float | None], *, k: float, degree: int) -> list[float | None]:
    """The base lane rescaled by ``k ** degree`` element-wise, preserving ``None`` and ``NaN``."""
    factor = k**degree
    result: list[float | None] = []
    for value in base:
        if value is None:
            result.append(None)
        elif math.isnan(value):
            result.append(math.nan)
        else:
            result.append(value * factor)
    return result


def check_scaling(declaration: Declaration) -> None:
    """Each homogeneity axis: scaling only its roles by a power of two scales each lane by ``k ** degree``."""
    scaling = declaration.scaling
    if isinstance(scaling, ScaleExempt):
        pytest.skip(f"{declaration.name}: scale-exempt — {scaling.reason}")
    length = widest_warmup(declaration) + 12
    base_frame = probe_frame(declaration.inputs, length)
    base = actual_lanes(declaration, base_frame)
    for axis in scaling:
        scaled_frame = base_frame.with_columns((pl.col(role) * _SCALE_FACTOR).alias(role) for role in axis.roles)
        scaled = actual_lanes(declaration, scaled_frame)
        roles = "+".join(axis.roles)
        probe = synthesis.describe(declaration, scaled_frame, f"the probe with {roles} scaled by {_SCALE_FACTOR}")
        triage = messages.triage_generic(declaration.name, f"scale axis {roles}")
        for name, base_values in base.items():
            degree = axis.degree[name] if isinstance(axis.degree, Mapping) else axis.degree
            expected = _homogeneous(base_values, k=_SCALE_FACTOR, degree=degree)
            abs_tol = input_scale(base_values) * abs(_SCALE_FACTOR) ** degree * TOLERANCE_FACTOR_EXACT
            index = first_mismatch(scaled[name], expected, rel_tol=TOLERANCE_RELATIVE_SCALE, abs_tol=abs_tol)
            if index is not None:
                disagreement = messages.Disagreement(lane=name, expected=expected, observed=scaled[name], index=index)
                raise AssertionError(
                    messages.describe_failure(
                        declaration=declaration,
                        check="check_scaling",
                        probe=probe,
                        disagreement=disagreement,
                        triage=triage,
                    )
                )


def check_raises(declaration: Declaration) -> None:
    """Each declared validation counterexample raises ``ValueError`` with its canonical message."""
    if not declaration.raises:
        pytest.skip(f"{declaration.name}: no validation counterexamples declared")
    for overrides, match in declaration.raises:
        try:
            build_expr(declaration, **overrides)
        except ValueError as exc:
            if re.search(match, str(exc)) is None:
                msg = (
                    f"{declaration.name}: build under {dict(overrides)} raised ValueError {str(exc)!r}, "
                    f"which does not match {match!r}"
                )
                raise AssertionError(msg) from exc
        else:
            msg = f"{declaration.name}: build under {dict(overrides)} did not raise ValueError (expected {match!r})"
            raise AssertionError(msg)


def _assert_window_exceeds_length(declaration: Declaration, field_name: str | None, count: int) -> None:
    """A frame no longer than a lane's warm-up emits nothing on that lane — the window never completes."""
    lanes = actual_lanes(declaration, probe_frame(declaration.inputs, count))
    values = lanes[field_name] if field_name is not None else next(iter(lanes.values()))
    label = field_name or declaration.name
    assert all(value is None for value in values), f"{label}: a frame no longer than its warm-up emitted a value"


def check_warmup(declaration: Declaration) -> None:
    """The output carries exactly the declared leading nulls, and a frame shorter than the warm-up emits nothing."""
    warmup = declaration.warmup
    if warmup is None:
        pytest.skip(f"{declaration.name}: no warm-up declared")
    lanes = actual_lanes(declaration, probe_frame(declaration.inputs, widest_warmup(declaration) + 8))
    if isinstance(warmup, Mapping):
        for field_name, expected in warmup.items():
            observed = count_leading_nulls(lanes[field_name])
            assert observed == expected, (
                f"{declaration.name}.{field_name}: {observed} leading nulls, declared {expected}"
            )
            _assert_window_exceeds_length(declaration, field_name, expected)
    else:
        (values,) = lanes.values()
        observed = count_leading_nulls(values)
        assert observed == warmup, f"{declaration.name}: {observed} leading nulls, declared {warmup}"
        _assert_window_exceeds_length(declaration, None, warmup)


def _deviant_lanes(expected: object) -> _Lanes:
    """Normalize a deviant's declared answer (a per-field mapping for a struct, else a single lane) to named lanes."""
    if isinstance(expected, Mapping):
        mapping = cast("Mapping[str, Sequence[float | None]]", expected)
        return {str(name): list(values) for name, values in mapping.items()}
    return {"out": list(cast("Sequence[float | None]", expected))}


def check_all_null(declaration: Declaration) -> None:
    """An all-null input yields all-null on every lane, or exactly the declared deviant answer."""
    probe = synthesis.frame_all_null(declaration)
    actual = actual_lanes(declaration, probe.frame)
    if declaration.deviant is None:
        for name, values in actual.items():
            assert all(value is None for value in values), (
                f"{declaration.name}.{name}: an all-null input was not all-null"
            )
        return
    triage = messages.triage_generic(declaration.name, "all-null deviant")
    _assert_lanes(
        declaration,
        "check_all_null",
        probe,
        expected=_deviant_lanes(declaration.deviant.expected),
        actual=actual,
        rel_tol=TOLERANCE_RELATIVE_EXACT,
        abs_tol=TOLERANCE_ABSOLUTE_EXACT,
        triage=triage,
    )


def check_empty(declaration: Declaration) -> None:
    """An empty frame gives zero rows for an elementwise output, one null row for a reduction."""
    out = probe_frame(declaration.inputs, 0).select(build_expr(declaration).alias("out"))
    if declaration.shape is Shape.REDUCING:
        assert out.height == 1, f"{declaration.name}: an empty reduction should yield one row, got {out.height}"
        for lane in lane_series(out):
            assert lane.is_null().all(), f"{declaration.name}: an empty reduction should be null"
    else:
        assert out.height == 0, f"{declaration.name}: an empty elementwise output should be empty, got {out.height}"


def check_single_row(declaration: Declaration) -> None:
    """A one-row input does not crash and keeps the declared shape."""
    out = probe_frame(declaration.inputs, 1).select(build_expr(declaration).alias("out"))
    assert out.height == 1, f"{declaration.name}: a one-row input should give one row, got {out.height}"
    schema = out.schema["out"]
    if declaration.shape is Shape.STRUCT:
        assert isinstance(schema, pl.Struct)
        assert tuple(field.name for field in schema.fields) == declaration.fields
    else:
        assert schema == pl.Float64, f"{declaration.name}: a one-row output should be Float64, got {schema}"
