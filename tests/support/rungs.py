"""
The generic checks, each a plain function taking a :class:`Declaration`.

There is no inheritance and no capability mixin: a check reads the declared fields it needs and, where the declaration
does not activate it (no golden, no pins, a scale-exempt function), skips cleanly with a reason. The correctness core
compares the factory against the naive oracle — on the deterministic probe, on fuzzed frames, and (the severity upgrade
over hand-pinned cases) on the synthesized degenerate regimes too, comparing VALUES against the oracle rather than only
the kind of outcome. Every value disagreement raises through :mod:`tests.support.messages`, so a failure names the
declaration that generated it, prints the tiny probe whole, shows expected vs observed, triages the two suspects, and
carries a copy-pasteable reproduction.
"""

import enum
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from tests.support import messages, synthesis
from tests.support.compare import first_mismatch
from tests.support.declaration import (
    Declaration,
    Pin,
    ScalarParam,
    ScaleExempt,
    Shape,
    actual_lanes,
    build_expr,
    horizon,
    lane_series,
    named_lanes,
    probe_length,
    reference_lanes,
    widest_warmup,
    window_length,
)
from tests.support.frames import count_leading_nulls, probe_frame
from tests.support.synthesis import Probe, fuzz_frames
from tests.support.tolerances import (
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
    bands: tuple[float, float],
    triage: str,
    expected_label: str = "oracle",
) -> None:
    """Assert the named lanes agree; on the first disagreement raise the rich, triaged failure message."""
    if sorted(actual) != sorted(expected):
        msg = (
            f"{declaration.name}: {check} produced lanes {sorted(actual)}, the {expected_label} has {sorted(expected)}"
        )
        raise AssertionError(msg)
    for name in sorted(expected):
        index = first_mismatch(actual[name], expected[name], rel_tol=bands[0], abs_tol=bands[1])
        if index is not None:
            disagreement = messages.Disagreement(lane=name, expected=expected[name], observed=actual[name], index=index)
            raise AssertionError(
                messages.describe_failure(
                    declaration=declaration,
                    check=check,
                    probe=probe,
                    disagreement=disagreement,
                    triage=triage,
                    expected_label=expected_label,
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
        bands=(rel_tol, abs_tol),
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
        bands=(TOLERANCE_RELATIVE_EXACT, TOLERANCE_ABSOLUTE_EXACT),
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
        bands=(TOLERANCE_RELATIVE_EXACT, TOLERANCE_ABSOLUTE_EXACT),
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


def check_recomposition(declaration: Declaration) -> None:
    """The factory reproduces its recomposition from other public functions, lane by lane, on the probe frame.

    A ratio metric equals its numerator over its denominator, an oscillator equals a difference of two lines: where the
    declaration states that identity as a zero-argument ``recomposition`` expression, the factory's output must equal
    the recomposition's on the deterministic probe. The comparison is a deliberate tier mix — accumulation-order noise
    between the two evaluations sits inside the property-tier relative band, while a near-zero lane (a struct's
    difference field) needs the reference-tier absolute floor. Skips cleanly when no recomposition is declared.
    """
    recomposition = declaration.recomposition
    if recomposition is None:
        pytest.skip(f"{declaration.name}: no recomposition declared")
    frame = probe_frame(declaration.inputs, probe_length(declaration))
    actual = actual_lanes(declaration, frame)
    expected = named_lanes(lane_series(frame.select(recomposition().alias("out"))))
    probe = synthesis.describe(declaration, frame, "the deterministic probe frame")
    triage = messages.triage_generic(declaration.name, "recomposition")
    _assert_lanes(
        declaration,
        "check_recomposition",
        probe,
        expected=expected,
        actual=actual,
        bands=(TOLERANCE_RELATIVE_PROPERTY, TOLERANCE_ABSOLUTE_REFERENCE),
        triage=triage,
    )


# ======================================================================================================================
# Missing-data and non-finite flow
# ======================================================================================================================


def _lane_is_nan(value: float | None) -> bool:
    return value is not None and math.isnan(value)


@dataclass(frozen=True)
class _FlowContext:
    """One SERIES lane's clean-vs-poisoned evidence plus the sizing a structural-flow shape reads."""

    clean: list[float | None]
    poisoned: list[float | None]
    row: int  # the injected interior bar
    reach: int  # the horizon past which a bounded effect must have played out
    # The tail-recovery band: a one-pass rolling lane recovers to the clean run within a sub-ULP once the missing bar
    # slides out of every window, not bit-for-bit, so the post-horizon comparison is kind-aware within tolerance.
    rel_tol: float
    abs_tol: float


def _lane_still_missing(value: float | None) -> bool:
    """A missing lane at a bar: a Polars null or a NaN, never a finite value."""
    return value is None or _lane_is_nan(value)


# The shape layer asserts the flow of a missing bar: a missing bar leaves a trace on the lane within the
# horizon window, then the lane recovers past it. Two recoveries — back to the clean run within tolerance (a bounded
# effect: PROPAGATES / IN_WINDOW / ABSORBED, and a NaN's clearing), or merely back to defined values (a recursion that
# carried a different state across the gap: BRIDGED). The value layer has already proved factory and oracle agree on
# the exact lanes, so the shape only classifies the recovery, never the interior null pattern — which differs per
# struct field (a lagged signal, a one-bar direction flag, a per-field window).


def _null_trace(ctx: _FlowContext) -> bool:
    """Whether the missing bar left a ``null`` on some clean-defined lane inside the horizon window."""
    upper = min(ctx.row + ctx.reach, len(ctx.poisoned))
    return any(ctx.poisoned[j] is None for j in range(ctx.row, upper) if ctx.clean[j] is not None)


def _nan_trace(ctx: _FlowContext) -> bool:
    """Whether the missing bar left a ``NaN`` on the lane inside the horizon window."""
    upper = min(ctx.row + ctx.reach, len(ctx.poisoned))
    return any(_lane_is_nan(ctx.poisoned[j]) for j in range(ctx.row, upper))


def _tail_diverges_from_clean(ctx: _FlowContext) -> int | None:
    """The first post-horizon row where the poisoned lane leaves the clean run (within tolerance), or ``None``."""
    start = ctx.row + ctx.reach + 1
    index = first_mismatch(ctx.poisoned[start:], ctx.clean[start:], rel_tol=ctx.rel_tol, abs_tol=ctx.abs_tol)
    return None if index is None else start + index


def _tail_still_missing(ctx: _FlowContext) -> int | None:
    """The first post-horizon row still missing on a clean-defined lane (did not recover), or ``None``."""
    for j in range(ctx.row + ctx.reach + 1, len(ctx.poisoned)):
        if ctx.clean[j] is not None and _lane_still_missing(ctx.poisoned[j]):
            return j
    return None


def _recovers_to_clean(ctx: _FlowContext, member: str) -> str | None:
    """A bounded missing bar leaves a null trace, then the lane returns to the clean run past the horizon."""
    if not _null_trace(ctx):
        return f"declared {member}: the missing bar left no null trace on the lane within the horizon"
    row = _tail_diverges_from_clean(ctx)
    if row is not None:
        return (
            f"declared {member}: beyond the horizon (row {ctx.row} + {ctx.reach}) the lane must equal the clean run; "
            f"row {row} differs"
        )
    return None


def _null_propagates(ctx: _FlowContext) -> str | None:
    return _recovers_to_clean(ctx, "PROPAGATES")


def _null_in_window(ctx: _FlowContext) -> str | None:
    return _recovers_to_clean(ctx, "IN_WINDOW_IS_NULL")


def _null_bridged(ctx: _FlowContext) -> str | None:
    """BRIDGED: a recursion steps over the gap (a null trace), then resumes at defined values — but its carried state
    need not match the clean run (a cumulation stays permanently offset; a contracting recursion reconverges).
    """
    if not _null_trace(ctx):
        return "declared BRIDGED: the missing bar left no null trace on the lane within the horizon"
    row = _tail_still_missing(ctx)
    if row is not None:
        return (
            f"declared BRIDGED: beyond the horizon (row {ctx.row} + {ctx.reach}) the recursion must carry across the "
            f"gap and resume at defined values; row {row} is still missing"
        )
    return None


def _null_absorbed(ctx: _FlowContext) -> str | None:
    """ABSORBED: a null candidate is dropped from the pointwise computation rather than nulling the whole row, so the
    injected row carries no null-trace claim; past the horizon the lane returns to the clean run.
    """
    row = _tail_diverges_from_clean(ctx)
    if row is not None:
        return (
            f"declared ABSORBED: beyond the horizon (row {ctx.row} + {ctx.reach}) the lane must equal the clean run; "
            f"row {row} differs"
        )
    return None


def _latch_violation(ctx: _FlowContext) -> str | None:
    """The shared latch shape (null or NaN): from the injection on, every clean-defined lane stays missing.

    A recursion that carries the contamination forward never recovers to a finite value. A NaN kernel latches a NaN
    (an EWM mean, a MACD line) while a finite-guarded pipeline latches a null (the Ehlers cycle cluster drops the bad
    prefix), so the shape accepts EITHER missing kind — the value layer has already proved factory and oracle agree on
    which one it is.
    """
    bad = [
        j
        for j in range(ctx.row, len(ctx.poisoned))
        if ctx.clean[j] is not None and not _lane_still_missing(ctx.poisoned[j])
    ]
    if bad:
        return (
            f"declared LATCHES: every defined lane from row {ctx.row} on must stay missing (null or NaN); "
            f"rows {bad[:4]} are not"
        )
    return None


def _nan_propagates(ctx: _FlowContext) -> str | None:
    """PROPAGATES: a NaN leaves a NaN trace within the horizon, then the lane clears back to the clean run."""
    if not _nan_trace(ctx):
        return "declared PROPAGATES: the missing bar left no NaN trace on the lane within the horizon"
    row = _tail_diverges_from_clean(ctx)
    if row is not None:
        return (
            f"declared PROPAGATES: beyond the horizon (row {ctx.row} + {ctx.reach}) the lane must clear back to the "
            f"clean run; row {row} differs"
        )
    return None


_SeriesHandler = Callable[["_FlowContext"], "str | None"]
_SHAPES_NULL: Mapping[str, _SeriesHandler] = {
    "PROPAGATES": _null_propagates,
    "BRIDGED": _null_bridged,
    "IN_WINDOW_IS_NULL": _null_in_window,
    "LATCHES": _latch_violation,
    "ABSORBED": _null_absorbed,
}
_SHAPES_NAN: Mapping[str, _SeriesHandler] = {"PROPAGATES": _nan_propagates, "LATCHES": _latch_violation}


@dataclass(frozen=True)
class _ReduceContext:
    """A reduction's poisoned scalar plus the clean frame and injected row a reducing-flow shape reads."""

    declaration: Declaration
    value: float | None  # the poisoned frame's reduction (one row, one lane)
    frame_clean: pl.DataFrame
    row: int  # the injected interior bar
    rel_tol: float
    abs_tol: float


def _reduce_skipped(ctx: _ReduceContext) -> str | None:
    without_row = ctx.frame_clean.with_row_index().filter(pl.col("index") != ctx.row).drop("index")
    (expected_lane,) = actual_lanes(ctx.declaration, without_row).values()
    (expected,) = expected_lane
    if first_mismatch([ctx.value], [expected], rel_tol=ctx.rel_tol, abs_tol=ctx.abs_tol) is not None:
        return (
            f"declared SKIPPED: the reduction must equal the factory recomputed with row {ctx.row} removed "
            f"({expected!r}), got {ctx.value!r}"
        )
    return None


def _reduce_poisons(ctx: _ReduceContext) -> str | None:
    if not _lane_is_nan(ctx.value):
        return f"declared POISONS: a NaN in the input must poison the reduction to NaN, got {ctx.value!r}"
    return None


def _reduce_propagates(ctx: _ReduceContext) -> str | None:
    if ctx.value is not None:
        return f"declared PROPAGATES: a null in the input must null the reduction, got {ctx.value!r}"
    return None


_ReduceHandler = Callable[["_ReduceContext"], "str | None"]
_REDUCE_NULL: Mapping[str, _ReduceHandler] = {"SKIPPED": _reduce_skipped, "PROPAGATES": _reduce_propagates}
_REDUCE_NAN: Mapping[str, _ReduceHandler] = {"POISONS": _reduce_poisons}


def _series_violation(shapes: Mapping[str, _SeriesHandler], member: str, ctx: _FlowContext) -> str | None:
    """The declared behavior's SERIES shape, checked on the factory's clean-vs-poisoned lane — fail-closed:
    a behavior member with no registered shape is a hole in the contract, never a silent pass.
    """
    handler = shapes.get(member)
    if handler is None:
        return f"no structural contract for behavior {member!r} — extend the flow shapes in rungs.py"
    return handler(ctx)


def _assert_flow(
    declaration: Declaration,
    check: str,
    pair: synthesis.FlowProbe,
    *,
    declared: enum.Enum,
    series_shapes: Mapping[str, _SeriesHandler],
    reduce_shapes: Mapping[str, _ReduceHandler],
) -> None:
    """A flow probe checked on BOTH layers: factory-vs-oracle by value, and the declared behavior's shape.

    The value layer proves the code and the naive reimplementation agree on the regime; the shape layer proves the
    DECLARATION tells the truth about it — a wrong behavior enum goes red here even when factory and oracle agree. The
    shape dialect is the declaration's: a SERIES lane recovers around the missing bar, a REDUCING scalar skips it,
    poisons, or nulls; a STRUCT runs the SERIES shape once per declared field, so every field is held to the contract.
    """
    rel, abs_ = _oracle_bands(declaration, TOLERANCE_RELATIVE_REFERENCE, TOLERANCE_ABSOLUTE_REFERENCE)
    expected = reference_lanes(declaration, pair.probe.frame)
    actual = actual_lanes(declaration, pair.probe.frame)
    triage = messages.triage_for_enum(declaration.name, declared)
    _assert_lanes(declaration, check, pair.probe, expected=expected, actual=actual, bands=(rel, abs_), triage=triage)
    if declaration.shape is Shape.REDUCING:
        _assert_reducing_flow(
            declaration, check, pair, declared=declared, actual=actual, shapes=reduce_shapes, rel_tol=rel, abs_tol=abs_
        )
    else:  # SERIES and STRUCT: one per-lane series-flow contract, run over every declared field of a struct
        _assert_series_flow(
            declaration, check, pair, declared=declared, actual=actual, shapes=series_shapes, rel_tol=rel, abs_tol=abs_
        )


def _assert_series_flow(
    declaration: Declaration,
    check: str,
    pair: synthesis.FlowProbe,
    *,
    declared: enum.Enum,
    actual: _Lanes,
    shapes: Mapping[str, _SeriesHandler],
    rel_tol: float,
    abs_tol: float,
) -> None:
    """The declared behavior's SERIES shape, per lane — one lane for a SERIES, one per declared field for a STRUCT: the
    missing bar's footprint on each lane, then its recovery to the clean run.
    """
    clean = actual_lanes(declaration, pair.frame_clean)
    reach = horizon(declaration)
    for name, lane_poisoned in actual.items():
        ctx = _FlowContext(
            clean=clean[name],
            poisoned=lane_poisoned,
            row=pair.row,
            reach=reach,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        violation = _series_violation(shapes, declared.name, ctx)
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


def _assert_reducing_flow(
    declaration: Declaration,
    check: str,
    pair: synthesis.FlowProbe,
    *,
    declared: enum.Enum,
    actual: _Lanes,
    shapes: Mapping[str, _ReduceHandler],
    rel_tol: float,
    abs_tol: float,
) -> None:
    """The declared REDUCING behavior's shape on the poisoned scalar — SKIPPED equals the row-removed recompute,
    POISONS goes NaN, PROPAGATES goes null; an unregistered member is a hole in the contract, never a silent pass.
    """
    (poisoned_lane,) = actual.values()
    (value,) = poisoned_lane
    (clean_lane,) = actual_lanes(declaration, pair.frame_clean).values()
    handler = shapes.get(declared.name)
    if handler is None:
        violation: str | None = (
            f"no structural contract for behavior {declared.name!r} — extend the reducing flow shapes in rungs.py"
        )
    else:
        ctx = _ReduceContext(
            declaration=declaration,
            value=value,
            frame_clean=pair.frame_clean,
            row=pair.row,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        violation = handler(ctx)
    if violation is not None:
        raise AssertionError(
            messages.describe_flow_violation(
                declaration=declaration,
                check=check,
                probe=pair.probe,
                declared=declared,
                violation=violation,
                evidence=messages.FlowEvidence("out", clean_lane, poisoned_lane, pair.row),
            )
        )


def check_behavior_null(declaration: Declaration) -> None:
    """An interior ``null`` plays out as the oracle plays it out AND as the declared null behavior promises."""
    if declaration.flow_deviation:
        pytest.skip(f"{declaration.name}: flow deviation — {declaration.flow_deviation}")
    _assert_flow(
        declaration,
        "check_behavior_null",
        synthesis.frame_flow_null(declaration),
        declared=declaration.behavior_null,
        series_shapes=_SHAPES_NULL,
        reduce_shapes=_REDUCE_NULL,
    )


def check_behavior_nan(declaration: Declaration) -> None:
    """An interior ``NaN`` plays out as the oracle plays it out AND as the declared NaN behavior promises."""
    if declaration.flow_deviation:
        pytest.skip(f"{declaration.name}: flow deviation — {declaration.flow_deviation}")
    _assert_flow(
        declaration,
        "check_behavior_nan",
        synthesis.frame_flow_nan(declaration),
        declared=declaration.behavior_nan,
        series_shapes=_SHAPES_NAN,
        reduce_shapes=_REDUCE_NAN,
    )


def check_nonfinite(declaration: Declaration) -> None:
    """Each input carrying ``±inf`` flows through exactly as the oracle carries it — the declared IEEE behavior."""
    if declaration.nonfinite is None:
        pytest.skip(f"{declaration.name}: no non-finite flow contract declared")
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
            bands=(TOLERANCE_RELATIVE_REFERENCE, TOLERANCE_ABSOLUTE_REFERENCE),
            triage=triage,
        )


# ======================================================================================================================
# Twin coherence and annualization (the metrics dialect)
# ======================================================================================================================


def _twin_reduce(twin: Declaration, window_slice: pl.DataFrame, params: Mapping[str, ScalarParam]) -> float | None:
    """The twin factory over one trailing window: a reducing twin's single value, a series twin's last value."""
    (lane,) = lane_series(window_slice.select(build_expr(twin, **params).alias("out")))
    values = lane.to_list()
    return cast("float | None", values[-1] if twin.shape is Shape.SERIES else values[0])


def check_twin_coherence(declaration: Declaration) -> None:
    """A rolling function's row ``i`` equals its twin reduced over the trailing window ending at ``i`` (factory vs
    factory, no oracle).

    On the deterministic probe each defined row must equal the twin evaluated on the slice ``[i - window + 1, i]`` — the
    reducing twin's single value, or the series twin's last value — within the declaration's oracle band (a one-pass
    rolling kernel and a two-pass reducing one round the same window differently). Skips cleanly when the declaration is
    not a rolling twin.
    """
    twin = declaration.rolling_of
    if twin is None:
        pytest.skip(f"{declaration.name}: no rolling twin declared")
    window = window_length(declaration)
    frame = probe_frame(declaration.inputs, widest_warmup(declaration) + 12)
    (rolling,) = actual_lanes(declaration, frame).values()
    rel, abs_ = _oracle_bands(declaration, TOLERANCE_RELATIVE_REFERENCE, TOLERANCE_ABSOLUTE_REFERENCE)
    twin_params: dict[str, ScalarParam] = {
        name: value for name, value in declaration.params.items() if name != declaration.window
    }
    triage = messages.triage_generic(declaration.name, f"rolling_of={twin.name} coherence")
    for i, value in enumerate(rolling):
        if value is None:
            continue  # warm-up or an incomplete window carries no coherence claim
        window_slice = frame.slice(i - window + 1, window)
        reduced = _twin_reduce(twin, window_slice, twin_params)
        if first_mismatch([value], [reduced], rel_tol=rel, abs_tol=abs_) is not None:
            probe = synthesis.describe(declaration, window_slice, f"the trailing window ending at row {i}")
            disagreement = messages.Disagreement(lane="out", expected=[reduced], observed=[value], index=0)
            raise AssertionError(
                messages.describe_failure(
                    declaration=declaration,
                    check="check_twin_coherence",
                    probe=probe,
                    disagreement=disagreement,
                    triage=triage,
                )
            )


# The two period counts the annualization ratio is read at — chosen so both ratios are exact: sqrt(252/63) == 2 and
# 252/63 == 4, and both are valid ``periods_per_year`` (>= 1).
_ANNUAL_PARAM = "periods_per_year"
_ANNUAL_P_HIGH = 252
_ANNUAL_P_LOW = 63
_ANNUAL_RATIO: Mapping[str, float] = {
    "SQRT_TIME": math.sqrt(_ANNUAL_P_HIGH / _ANNUAL_P_LOW),
    "LINEAR": _ANNUAL_P_HIGH / _ANNUAL_P_LOW,
}
_ANNUAL_NO_CLOSED_FORM = frozenset({"GEOMETRIC", "NONE"})
_ANNUAL_BASE_FLOOR = 1e-9  # skip a probe row whose base value is ~0: the annualization ratio is undefined there


def check_annualization(declaration: Declaration) -> None:
    """A closed-form annualization scales the output by a known power of the period count.

    Where the convention is closed-form, ``factory(P=252) / factory(P=63)`` on one probe frame must equal the declared
    ratio (``sqrt(252/63)`` for SQRT_TIME, ``252/63`` for LINEAR), with the ``risk_free_rate``-like knobs held at their
    neutral default (the declaration's own params); a near-zero base row is skipped, the ratio being undefined there.
    GEOMETRIC and NONE carry no closed-form ratio — the oracle already encodes the convention — so the rung skips with
    that reason; an unknown member is fail-closed.
    """
    annualization = declaration.annualization
    if annualization is None:
        pytest.skip(f"{declaration.name}: no annualization declared")
    member = annualization.name
    if member in _ANNUAL_NO_CLOSED_FORM:
        pytest.skip(f"{declaration.name}: {member} has no closed-form annualization ratio — the oracle carries it")
    if member not in _ANNUAL_RATIO:
        msg = f"{declaration.name}: no closed-form annualization contract for {member!r} — extend check_annualization"
        raise NotImplementedError(msg)
    assert _ANNUAL_PARAM in declaration.params, (
        f"{declaration.name}: declared {member} annualization but has no {_ANNUAL_PARAM!r} parameter to vary"
    )
    frame = probe_frame(declaration.inputs, widest_warmup(declaration) + 12)
    (high,) = lane_series(frame.select(build_expr(declaration, **{_ANNUAL_PARAM: _ANNUAL_P_HIGH}).alias("out")))
    (low,) = lane_series(frame.select(build_expr(declaration, **{_ANNUAL_PARAM: _ANNUAL_P_LOW}).alias("out")))
    expected = _ANNUAL_RATIO[member]
    rel, abs_ = _oracle_bands(declaration, TOLERANCE_RELATIVE_REFERENCE, TOLERANCE_ABSOLUTE_REFERENCE)
    checked = 0
    for i, (hi, lo) in enumerate(zip(high.to_list(), low.to_list(), strict=True)):
        if lo is None or hi is None or math.isnan(lo) or math.isnan(hi) or math.isinf(lo) or math.isinf(hi):
            continue
        if abs(lo) < _ANNUAL_BASE_FLOOR:  # denominator guard: the ratio is undefined at a ~0 base
            continue
        checked += 1
        ratio = hi / lo
        assert math.isclose(ratio, expected, rel_tol=rel, abs_tol=abs_), (
            f"{declaration.name}: {member} annualization at row {i} — factory(P={_ANNUAL_P_HIGH}) / "
            f"factory(P={_ANNUAL_P_LOW}) = {ratio} != {expected}. "
            f"Either the declaration is wrong (did you mean another Annualization?) or {declaration.name} has a bug."
        )
    if checked == 0:
        pytest.skip(f"{declaration.name}: every probe row had a ~0 base value — the annualization ratio is undefined")


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


def _refute_hidden_homogeneity(declaration: Declaration, exemption: ScaleExempt) -> None:
    """A ``ScaleExempt`` must not be secretly homogeneous: scaling every input must fit no clean integer degree.

    The probe scales all inputs at once and tries each degree in ``[-3, 3]`` against every lane; a fit means the
    exemption hides a declarable ``ScaleAxis``. A vacuous fit does not count — the comparison must be grounded in at
    least a handful of finite values.
    """
    length = widest_warmup(declaration) + 12
    base_frame = probe_frame(declaration.inputs, length)
    if not {"high", "low"} <= set(declaration.inputs):
        # The deterministic builders can starve the probe of evidence: an exact cross-role proportionality makes alpha
        # identically zero, and a monotone equity curve has zero drawdown, so the drawdown ratios are all infinite —
        # neither grounds any degree. A small multiplicative row-alternating wiggle on the last role restores nonzero
        # finite values while staying in-domain. Applied before scaling, so the homogeneity question — f(k * x) versus
        # f(x) — is unchanged. OHLC frames are left alone (a wiggle could break bar coherence, which is out-of-domain
        # by contract).
        last = declaration.inputs[-1]
        wiggle = 1.0 + 0.05 * ((pl.int_range(0, pl.len()) % 2) * 2 - 1).cast(pl.Float64)
        base_frame = base_frame.with_columns((pl.col(last) * wiggle).alias(last))
    base = actual_lanes(declaration, base_frame)
    scaled_frame = base_frame.with_columns((pl.col(role) * _SCALE_FACTOR).alias(role) for role in declaration.inputs)
    scaled = actual_lanes(declaration, scaled_frame)
    pairs = [
        (base_value, scaled_value)
        for name, values in base.items()
        for base_value, scaled_value in zip(values, scaled[name], strict=True)
        if base_value is not None
        and scaled_value is not None
        and math.isfinite(base_value)
        and math.isfinite(scaled_value)
    ]
    if not any(abs(base_value) > TOLERANCE_ABSOLUTE_PROPERTY for base_value, _ in pairs):
        pytest.skip(f"{declaration.name}: scale-exempt — the probe grounds no nonzero value to refute homogeneity on")
    for degree in range(-3, 4):
        factor = _SCALE_FACTOR**degree
        fits = all(
            math.isclose(
                scaled_value, base_value * factor, rel_tol=TOLERANCE_RELATIVE_SCALE, abs_tol=TOLERANCE_ABSOLUTE_PROPERTY
            )
            for base_value, scaled_value in pairs
        )
        if fits:
            pytest.fail(
                f"{declaration.name}: declared scale-exempt ({exemption.reason}) but scaling every input by "
                f"{_SCALE_FACTOR} scales every lane by {_SCALE_FACTOR} ** {degree} — declare a ScaleAxis of degree "
                f"{degree} instead"
            )


def check_scaling(declaration: Declaration) -> None:
    """Each homogeneity axis: scaling only its roles by a power of two scales each lane by ``k ** degree``; a
    ``ScaleExempt`` is counter-probed instead, so an exemption cannot hide a declarable axis.
    """
    scaling = declaration.scaling
    if isinstance(scaling, ScaleExempt):
        _refute_hidden_homogeneity(declaration, scaling)
        return
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
                        expected_label="declared scaling",
                    )
                )


def check_type_error(declaration: Declaration) -> None:
    """A bare column-name string in place of a ``pl.Expr`` raises the canonical ``TypeError``.

    The public signatures promise expressions; the shared validation hub narrows a stray string at runtime with
    one canonical message, and every factory must route through it.
    """
    arguments: list[object] = ["close", *(pl.col(role) for role in declaration.inputs[1:])]
    with pytest.raises(TypeError, match="expected a Polars expression"):
        declaration.factory(*arguments, **declaration.params)


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
        bands=(TOLERANCE_RELATIVE_EXACT, TOLERANCE_ABSOLUTE_EXACT),
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
