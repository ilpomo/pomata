"""
The shared rungs of the ladder, each written once as a module-level function and parametrized over the applicable
spec subset.

Applicability is a plain comprehension on spec *fields* — no capability mixins, no ``__init_subclass__``. A subset
cannot silently drop an obligation, because the field it filters on is either required by the language (``shape``) or
made mandatory by :meth:`Spec.__post_init__` (``raises`` whenever ``params`` is non-empty; a non-empty ``scale``
tuple). Sub-parametrized rungs give one case per struct field, per validation counterexample, or per scale axis, with
a readable id (``ichimoku-senkou_b``, ``sharpe_ratio-0``, ``ichimoku-high+low``).

To read a failure: find the rung by the name in the pytest id, read its few lines, then read the spec row the id
names. Two obvious places, no inheritance graph to walk. The rungs run in the canonical order of the method of record
— contract, then edge, then correctness, then properties.
"""

import math
from collections.abc import Mapping
from typing import cast

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from tests.support import (
    ABSOLUTE_TOLERANCE_PROPERTY,
    ABSOLUTE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    assert_matches,
    assert_scale_homogeneous,
    count_leading_nulls,
)
from tests_new.all_specs import ALL_SPECS
from tests_new.support.spec import (
    SPEC_LANE,
    SPEC_SCALAR,
    ScaleAxis,
    Shape,
    Spec,
    SpecPin,
    actual_lanes,
    build_expr,
    flat,
    fuzz_frames,
    horizon,
    lane_series,
    probe_frame,
    probe_length,
    reference_lanes,
    spec_id,
    widest_warmup,
)

from pomata._policy import NanPolicy, NullPolicy

# --- applicability subsets: pure filters on declared fields, with per-case ids ---

NON_REDUCING_SPECS = [spec for spec in ALL_SPECS if spec.shape is not Shape.REDUCING]


def _raises_cases() -> tuple[list[tuple[Spec, Mapping[str, SPEC_SCALAR], str]], list[str]]:
    cases = [(spec, overrides, match) for spec in ALL_SPECS for overrides, match in spec.raises]
    ids = [f"{spec.name}-{index}" for spec in ALL_SPECS for index, _ in enumerate(spec.raises)]
    return cases, ids


def _warmup_cases() -> tuple[list[tuple[Spec, str | None, int]], list[str]]:
    cases: list[tuple[Spec, str | None, int]] = []
    ids: list[str] = []
    for spec in ALL_SPECS:
        warmup = spec.warmup
        if warmup is None:
            continue
        if spec.shape is Shape.STRUCT:
            for field_name in spec.fields:
                expected = warmup[field_name] if isinstance(warmup, Mapping) else warmup
                cases.append((spec, field_name, expected))
                ids.append(f"{spec.name}-{field_name}")
        else:
            assert isinstance(warmup, int)
            cases.append((spec, None, warmup))
            ids.append(spec.name)
    return cases, ids


def _scale_cases() -> tuple[list[tuple[Spec, ScaleAxis]], list[str]]:
    cases: list[tuple[Spec, ScaleAxis]] = []
    ids: list[str] = []
    for spec in ALL_SPECS:
        if isinstance(spec.scale, tuple):
            for axis in spec.scale:
                cases.append((spec, axis))
                ids.append(f"{spec.name}-{'+'.join(axis.roles)}")
    return cases, ids


def _pin_cases() -> tuple[list[tuple[Spec, SpecPin]], list[str]]:
    cases = [(spec, pin) for spec in ALL_SPECS for pin in spec.pins]
    ids = [f"{spec.name}-{pin.label}" for spec in ALL_SPECS for pin in spec.pins]
    return cases, ids


RAISES_CASES, RAISES_IDS = _raises_cases()
WARMUP_CASES, WARMUP_IDS = _warmup_cases()
SCALE_CASES, SCALE_IDS = _scale_cases()
PIN_CASES, PIN_IDS = _pin_cases()
COMPONENT_SPECS = [spec for spec in ALL_SPECS if spec.component_expr is not None]


# --- shared bodies ---


def _assert_reference(spec: Spec, frame: pl.DataFrame, rel_tol: float, abs_tol: float) -> None:
    """The shared oracle-agreement body: same named lanes, each within tolerance — every struct field, not only one."""
    expected = reference_lanes(spec, frame)
    actual = actual_lanes(spec, frame)
    assert sorted(actual) == sorted(expected)
    rel = spec.oracle_rel_tol if spec.oracle_rel_tol is not None else rel_tol
    abs_ = spec.oracle_abs_tol if spec.oracle_abs_tol is not None else abs_tol
    for name, values in expected.items():
        assert_matches(actual[name], values, rel_tol=rel, abs_tol=abs_)


# ======================================================================================================================
# Contract
# ======================================================================================================================


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_returns_expr(spec: Spec) -> None:
    """Verifies the factory returns a free-standing ``pl.Expr`` without touching any data."""
    assert isinstance(build_expr(spec), pl.Expr)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_output_lands_on_declared_column(spec: Spec) -> None:
    """Verifies the output keeps the declared landing column on distinctly-named input columns."""
    frame = probe_frame(spec.inputs, 16)
    out = frame.select(build_expr(spec))
    assert out.columns == [spec.landing]


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_shape_matches_declaration(spec: Spec) -> None:
    """Verifies the observed output shape is exactly the declared one — the sole coverage guard, on shape only."""
    length = max(widest_warmup(spec) + 8, 4)
    frame = probe_frame(spec.inputs, length)
    out = frame.select(build_expr(spec).alias("out"))
    if isinstance(out.schema["out"], pl.Struct):
        observed = Shape.STRUCT
    elif out.height < frame.height:
        observed = Shape.REDUCING
    else:
        observed = Shape.SERIES
    assert observed is spec.shape


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_lazy_eager_parity(spec: Spec) -> None:
    """Verifies the lazy plan collects to exactly the eager result."""
    frame = probe_frame(spec.inputs, probe_length(spec))
    eager = frame.select(build_expr(spec).alias("out"))
    lazy = frame.lazy().select(build_expr(spec).alias("out")).collect()
    assert eager.equals(lazy)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_over_partitions_independently(spec: Spec) -> None:
    """Verifies two stacked series under ``.over`` reproduce each series computed alone (never a bit-equality)."""
    length = probe_length(spec)
    first = probe_frame(spec.inputs, length)
    second = probe_frame(spec.inputs, length).select(pl.all() * 3.0)
    stacked = pl.concat([first, second]).with_columns(pl.Series("group", ["a"] * length + ["b"] * length))
    grouped = stacked.select(build_expr(spec).over("group").alias("out"))
    if spec.shape is Shape.REDUCING:
        # ``.over`` broadcasts a reduction across its group's rows: each row carries its own series' reduction.
        alone_first = first.select(build_expr(spec).alias("out"))["out"].to_list()
        alone_second = second.select(build_expr(spec).alias("out"))["out"].to_list()
        assert_matches(grouped["out"].to_list(), alone_first * length + alone_second * length)
    else:
        alone = pl.concat([first.select(build_expr(spec).alias("out")), second.select(build_expr(spec).alias("out"))])
        for lane_grouped, lane_alone in zip(lane_series(grouped), lane_series(alone), strict=True):
            assert_matches(lane_grouped.to_list(), lane_alone.to_list())


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_bare_string_raises_type_error(spec: Spec) -> None:
    """Verifies the shared input guard rejects a bare column name on the first input."""
    arguments: list[object] = ["close", *[pl.col(role) for role in spec.inputs[1:]]]
    with pytest.raises(TypeError, match=r"expected a Polars expression"):
        spec.factory(*arguments, **spec.params)


# ======================================================================================================================
# Edge
# ======================================================================================================================


@pytest.mark.parametrize(("spec", "overrides", "match"), RAISES_CASES, ids=RAISES_IDS)
def test_invalid_params_raise(spec: Spec, overrides: Mapping[str, SPEC_SCALAR], match: str) -> None:
    """Verifies each declared validation counterexample raises ``ValueError`` with its canonical message."""
    with pytest.raises(ValueError, match=match):
        build_expr(spec, **overrides)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_all_null_input(spec: Spec) -> None:
    """Verifies an all-null input yields all-null on every lane, or exactly the declared :class:`Deviant`."""
    frame = pl.DataFrame({role: pl.Series([None] * 12, dtype=pl.Float64) for role in spec.inputs})
    actual = actual_lanes(spec, frame)
    if spec.all_null is None:
        for values in actual.values():
            assert all(value is None for value in values)
    else:
        expected = spec.all_null.expected
        lanes = (
            {str(name): values for name, values in cast("Mapping[str, SPEC_LANE]", expected).items()}
            if isinstance(expected, Mapping)
            else {"out": cast("SPEC_LANE", expected)}
        )
        assert sorted(actual) == sorted(lanes)
        for lane_name, lane_values in lanes.items():
            assert_matches(actual[lane_name], list(lane_values))


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_single_row(spec: Spec) -> None:
    """Verifies a one-row input does not crash and keeps the declared shape."""
    frame = probe_frame(spec.inputs, 1)
    out = frame.select(build_expr(spec).alias("out"))
    assert out.height == 1
    schema = out.schema["out"]
    if spec.shape is Shape.STRUCT:
        assert isinstance(schema, pl.Struct)
        assert tuple(field.name for field in schema.fields) == spec.fields
    else:
        assert schema == pl.Float64


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_empty(spec: Spec) -> None:
    """Verifies an empty frame gives zero rows for an elementwise output, one null row for a reduction."""
    frame = probe_frame(spec.inputs, 0)
    out = frame.select(build_expr(spec).alias("out"))
    if spec.shape is Shape.REDUCING:
        assert out.height == 1
        for lane in lane_series(out):
            assert lane.is_null().all()
    else:
        assert out.height == 0


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_interior_null_flow(spec: Spec) -> None:
    """Verifies an interior missing bar plays out exactly as the declared null policy states."""
    _assert_flow(spec, nan=False)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_interior_nan_flow(spec: Spec) -> None:
    """Verifies an interior NaN bar plays out exactly as the declared NaN policy states."""
    _assert_flow(spec, nan=True)


def _assert_flow(spec: Spec, *, nan: bool) -> None:
    length = probe_length(spec)
    injection = widest_warmup(spec) + 3
    clean = probe_frame(spec.inputs, length)
    poisoned = clean.with_columns(
        pl.when(pl.int_range(pl.len()) == injection)
        .then(pl.lit(float("nan"), dtype=pl.Float64) if nan else pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col(role))
        .alias(role)
        for role in spec.inputs
    )
    policy: NullPolicy | NanPolicy = spec.nan_policy if nan else spec.null_policy
    baseline_lanes = flat(spec, clean)
    poisoned_lanes = flat(spec, poisoned)
    reach = horizon(spec)
    for baseline, poisoned_lane in zip(baseline_lanes, poisoned_lanes, strict=True):
        if policy in (NullPolicy.SKIPPED, NanPolicy.POISONS):
            _assert_reducing_flow(spec, clean, injection, poisoned_lane, nan=nan)
        elif policy in (NullPolicy.LATCHES, NanPolicy.LATCHES):
            tail = poisoned_lane.slice(injection + reach)
            assert tail.len() > 0, "the probe leaves no tail to observe — lengthen the probe"
            defined = tail.is_not_null() & tail.is_not_nan().fill_null(value=False)
            assert not bool(defined.any()), f"{poisoned_lane.name}: a latched lane recovered"
        else:  # ABSORBED, PROPAGATES, IN_WINDOW_IS_NULL, BRIDGED — the effect is bounded and the lane recovers
            affected = poisoned_lane.slice(injection, reach)
            if policy is not NullPolicy.ABSORBED:
                hit = affected.is_null() if not nan else affected.is_nan().fill_null(value=True)
                assert bool(hit.any()), f"{poisoned_lane.name}: the bar left no trace"
            tail = poisoned_lane.slice(injection + reach)
            assert tail.len() > 0, "the probe leaves no tail to observe — lengthen the probe"
            assert not bool(tail.is_null().any()), f"{poisoned_lane.name}: did not recover to defined values"
            assert not bool(tail.is_nan().any()), f"{poisoned_lane.name}: NaN survived past the horizon"
            if policy in (NullPolicy.PROPAGATES, NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES):
                # Past the window the lane must return to the clean baseline. A one-pass rolling moment (a cumulative
                # -sum kernel, e.g. rolling covariance or a standardized moment) carries a sub-ULP rounding of the
                # missing bar forever, so the recovery is to the baseline values within tolerance, not bit-for-bit.
                expected_tail = baseline.slice(injection + reach)
                assert_matches(
                    tail.to_list(),
                    expected_tail.to_list(),
                    rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                    abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
                )


def _assert_reducing_flow(spec: Spec, clean: pl.DataFrame, injection: int, poisoned: pl.Series, *, nan: bool) -> None:
    if nan:  # POISONS: the scalar goes NaN
        assert poisoned.len() == 1
        value = cast("float | None", poisoned.item())
        assert value is not None, "a NaN input nulled the reduction instead of poisoning it"
        assert math.isnan(value), "a NaN input did not poison the reduction"
        return
    # SKIPPED: the scalar is exactly what it would be if the row were absent.
    without_row = clean.with_row_index().filter(pl.col("index") != injection).drop("index")
    expected = without_row.select(build_expr(spec).alias("out"))["out"]
    assert poisoned.equals(expected), "a skipped null changed the reduction"


@pytest.mark.parametrize(("spec", "field_name", "expected"), WARMUP_CASES, ids=WARMUP_IDS)
def test_warmup_null_count(spec: Spec, field_name: str | None, expected: int) -> None:
    """Verifies the output carries exactly the declared leading nulls — per field for a struct."""
    frame = probe_frame(spec.inputs, widest_warmup(spec) + 8)
    lanes = actual_lanes(spec, frame)
    if field_name is None:
        (values,) = lanes.values()  # a windowed series has exactly one lane
        assert count_leading_nulls(values) == expected
    else:
        observed = count_leading_nulls(lanes[field_name])
        assert observed == expected, f"{field_name}: {observed} != {expected}"


@pytest.mark.parametrize(("spec", "field_name", "warmup"), WARMUP_CASES, ids=WARMUP_IDS)
def test_window_exceeds_length(spec: Spec, field_name: str | None, warmup: int) -> None:
    """Verifies a frame no longer than a lane's warm-up emits nothing on that lane — the window never completes."""
    frame = probe_frame(spec.inputs, warmup)
    lanes = actual_lanes(spec, frame)
    values = lanes[field_name] if field_name is not None else next(iter(lanes.values()))
    assert all(value is None for value in values), f"{field_name or 'out'}: a window shorter than its warm-up emitted"


@pytest.mark.parametrize("spec", NON_REDUCING_SPECS, ids=spec_id)
def test_no_lookahead(spec: Spec) -> None:
    """Verifies a prefix of the frame gives the prefix of the full output — no lane reads a future bar."""
    length = widest_warmup(spec) + 12
    prefix_length = length - 4
    frame = probe_frame(spec.inputs, length)
    full = actual_lanes(spec, frame)
    prefix = actual_lanes(spec, frame.slice(0, prefix_length))
    assert sorted(full) == sorted(prefix)
    for name, values in full.items():
        assert_matches(prefix[name], values[:prefix_length])


# ======================================================================================================================
# Correctness
# ======================================================================================================================


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_matches_reference(spec: Spec) -> None:
    """Verifies agreement with the oracle on the deterministic probe frame, at the reference tier."""
    frame = probe_frame(spec.inputs, widest_warmup(spec) + 12)
    _assert_reference(spec, frame, RELATIVE_TOLERANCE_REFERENCE, ABSOLUTE_TOLERANCE_REFERENCE)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_golden_master(spec: Spec) -> None:
    """Verifies the frozen golden master, rounded expression-side so it can never flake cross-platform."""
    frame = pl.DataFrame(
        {role: pl.Series(list(values), dtype=pl.Float64) for role, values in spec.golden_input.items()}
    )
    out = frame.select(build_expr(spec, **spec.golden_params).alias("out"))
    schema = out.schema["out"]
    expected = spec.golden_output
    if isinstance(schema, pl.Struct):
        assert isinstance(expected, Mapping)
        for field in schema.fields:
            lane = out["out"].struct.field(field.name).round(spec.golden_round).to_list()
            assert_matches(lane, list(expected[field.name]))
    else:
        assert not isinstance(expected, Mapping)
        lane = out["out"].round(spec.golden_round).to_list()
        assert_matches(lane, list(expected))


def _assert_pinned_lane(actual: list[float | None], expected: list[float | None], *, signed: bool) -> None:
    """Match a pinned lane by value; when the pin marks the sign, also compare ``copysign`` so ``-0.0`` != ``0.0``."""
    assert_matches(actual, expected)
    if signed:
        for got, want in zip(actual, expected, strict=True):
            if got is not None and want is not None:
                assert math.copysign(1.0, got) == math.copysign(1.0, want), f"sign mismatch: {got} vs {want}"


@pytest.mark.parametrize(("spec", "pin"), PIN_CASES, ids=PIN_IDS)
def test_pinned_cases(spec: Spec, pin: SpecPin) -> None:
    """Verifies each crafted-input case ported from the old suite: the pinned frame maps to the pinned lanes exactly."""
    frame = pl.DataFrame({role: pl.Series(list(values), dtype=pl.Float64) for role, values in pin.inputs.items()})
    lanes = lane_series(frame.select(build_expr(spec, **pin.params_override).alias("out")))
    if isinstance(pin.expected, Mapping):
        actual = {lane.name: lane.to_list() for lane in lanes}
        expected = {str(name): list(values) for name, values in pin.expected.items()}
    else:
        (single,) = lanes  # a non-struct output is exactly one lane
        actual = {"out": single.to_list()}
        expected = {"out": list(pin.expected)}
    assert sorted(actual) == sorted(expected)
    for name, values in expected.items():
        _assert_pinned_lane(actual[name], values, signed=pin.signed)


# ======================================================================================================================
# Properties
# ======================================================================================================================


@pytest.mark.parametrize(("spec", "axis"), SCALE_CASES, ids=SCALE_IDS)
def test_scale(spec: Spec, axis: ScaleAxis) -> None:
    """Verifies each homogeneity axis: scaling only its roles by a power of two scales the output by ``k**degree``."""
    length = widest_warmup(spec) + 12
    base_frame = probe_frame(spec.inputs, length)
    scaled_frame = base_frame.with_columns((pl.col(role) * 4.0).alias(role) for role in axis.roles)
    base = actual_lanes(spec, base_frame)
    scaled = actual_lanes(spec, scaled_frame)
    for name, base_values in base.items():
        assert_scale_homogeneous(scaled[name], base_values, k=4.0, degree=axis.degree)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
@given(data=st.data())
def test_matches_reference_for_any_input(spec: Spec, data: st.DataObject) -> None:
    """Verifies oracle agreement over the fuzz domain, at the property tier (``@given`` inside ``@parametrize``)."""
    frame = data.draw(fuzz_frames(spec, missing=False))
    if spec.conditioning is not None:
        assume(spec.conditioning(frame))
    _assert_reference(spec, frame, RELATIVE_TOLERANCE_PROPERTY, ABSOLUTE_TOLERANCE_PROPERTY)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
@given(data=st.data())
def test_matches_reference_under_missing_data(spec: Spec, data: st.DataObject) -> None:
    """Verifies oracle agreement when the fuzz mixes interior nulls and NaNs into the input."""
    frame = data.draw(fuzz_frames(spec, missing=True))
    if spec.conditioning is not None:
        assume(spec.conditioning(frame))
    _assert_reference(spec, frame, RELATIVE_TOLERANCE_PROPERTY, ABSOLUTE_TOLERANCE_PROPERTY)


@pytest.mark.parametrize("spec", COMPONENT_SPECS, ids=spec_id)
def test_matches_component_definition(spec: Spec) -> None:
    """Verifies the factory reproduces its recomposition from public functions, lane by lane, on the probe frame."""
    component = spec.component_expr
    assert component is not None  # COMPONENT_SPECS filters on this; the bind narrows it for the type checker
    frame = probe_frame(spec.inputs, probe_length(spec))
    direct = lane_series(frame.select(build_expr(spec).alias("out")))
    composed = lane_series(frame.select(component().alias("out")))
    for lane_direct, lane_composed in zip(direct, composed, strict=True):
        assert_matches(
            lane_direct.to_list(),
            lane_composed.to_list(),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
