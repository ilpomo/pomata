"""
The contract framework: declarative test classes whose obligations are inherited, never restated.

A per-function test class *declares* facts (the factory, its input columns, canonical parameters, warm-up) and
*inherits* every rung from the capability mixins it composes; the machinery derives the rest — the function's name,
family, and ``(null_policy, nan_policy)`` (from :mod:`pomata._policy`) — so a declaration can never disagree with
the package. Uniformity is structural: a rung has one implementation, one name, and one order, because it exists
exactly once, in the mixin. See ``tests_new/DESIGN.md`` for the axes and the migration map.

Three by-construction guarantees replace the old grammar/policy meta-tests:

- **completeness** — a concrete contract missing a required declaration fails at import with the missing names;
- **honesty** — a concrete contract cannot override an inherited rung unless the name is listed in its
  ``override_ok`` (a visible, greppable consent);
- **bijection** — every concrete contract registers itself; ``tests_new/test_surface.py`` holds the registry in
  exact correspondence with the migrated surface (and, at cutover, with the public ``__all__`` tuples).
"""

import inspect
import math
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import ClassVar, cast

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.support import (
    ABSOLUTE_TOLERANCE_PROPERTY,
    ABSOLUTE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    assert_matches,
    coherent_hl,
    coherent_hl_with_missing,
    count_leading_nulls,
    finite_floats,
    missing_data_floats,
    split_pairs,
)

import pomata.indicators
import pomata.metrics
import pomata.pnl
from pomata._policy import POLICIES, NanPolicy, NullPolicy

_FAMILIES: dict[str, ModuleType] = {
    "indicators": pomata.indicators,
    "metrics": pomata.metrics,
    "pnl": pomata.pnl,
}

# Every concrete contract in the suite, keyed by function name; filled by ``Contract.__init_subclass__`` and held
# in exact correspondence with the migrated surface by ``tests_new/test_surface.py``.
REGISTRY: dict[str, type["Contract"]] = {}

# Input-column roles the probe frame knows how to synthesize; a contract's ``inputs`` must draw from these.
_ROLE_BUILDERS: dict[str, Callable[[int], pl.Series]] = {
    "high": lambda n: pl.Series([float(i) + 1.5 for i in range(n)], dtype=pl.Float64),
    "low": lambda n: pl.Series([float(i) + 0.5 for i in range(n)], dtype=pl.Float64),
    "open": lambda n: pl.Series([float(i) + 0.9 for i in range(n)], dtype=pl.Float64),
    "close": lambda n: pl.Series([float(i) + 1.1 for i in range(n)], dtype=pl.Float64),
    "volume": lambda n: pl.Series([100.0 + float(i) for i in range(n)], dtype=pl.Float64),
    "expr": lambda n: pl.Series([float(i) + 1.0 for i in range(n)], dtype=pl.Float64),
    "price": lambda n: pl.Series([float(i) + 10.0 for i in range(n)], dtype=pl.Float64),
    "equity_curve": lambda n: pl.Series([100.0 * (1.02 ** float(i)) for i in range(n)], dtype=pl.Float64),
    "returns": lambda n: pl.Series([0.01 if i % 2 == 0 else -0.005 for i in range(n)], dtype=pl.Float64),
    "benchmark": lambda n: pl.Series([0.008 if i % 2 == 0 else -0.004 for i in range(n)], dtype=pl.Float64),
    "asset_returns": lambda n: pl.Series([0.01 if i % 2 == 0 else -0.005 for i in range(n)], dtype=pl.Float64),
    "weight": lambda n: pl.Series([0.5 + 0.01 * float(i) for i in range(n)], dtype=pl.Float64),
    "quantity": lambda n: pl.Series([10.0 + float(i % 3) for i in range(n)], dtype=pl.Float64),
    "cost": lambda n: pl.Series([0.1 for _ in range(n)], dtype=pl.Float64),
    "dividend_per_share": lambda n: pl.Series([0.05 for _ in range(n)], dtype=pl.Float64),
    "returns_gross": lambda n: pl.Series([0.01 if i % 2 == 0 else -0.005 for i in range(n)], dtype=pl.Float64),
    "funding_rate": lambda n: pl.Series([0.0001 for _ in range(n)], dtype=pl.Float64),
}


def probe_frame(inputs: tuple[str, ...], length: int) -> pl.DataFrame:
    """A well-conditioned frame carrying one distinctly-named column per declared input role."""
    return pl.DataFrame({role: _ROLE_BUILDERS[role](length) for role in inputs})


def _pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def _wrap_rung(impl: "Callable[[Contract, st.DataObject], None]") -> "Callable[[Contract, st.DataObject], None]":
    """A fresh, default-free wrapper around a Hypothesis rung implementation (``@given`` rejects defaults)."""

    def rung(self: "Contract", data: st.DataObject) -> None:
        impl(self, data)

    return rung


def _family_of(name: str) -> str:
    for family, module in _FAMILIES.items():
        if name in module.__all__:
            return family
    msg = f"{name} is in no public __all__"
    raise TypeError(msg)


class Contract:
    """
    The root of every per-function contract: shared declarations, derivations, and the machinery's three locks.

    Concrete subclasses (names starting with ``Test``) declare ``factory``, ``inputs``, and ``params``; the
    machinery derives ``name`` (from the factory), ``family`` (from the public ``__all__`` tuples), and
    ``null_policy`` / ``nan_policy`` (from :mod:`pomata._policy`), and validates the declaration at import time.
    """

    _requires: ClassVar[tuple[str, ...]] = ("factory", "inputs", "params")

    # --- declared by every concrete contract ---
    factory: ClassVar[Callable[..., pl.Expr]]
    inputs: ClassVar[tuple[str, ...]]
    params: ClassVar[Mapping[str, int | float | bool]]
    # The column the output lands on; defaults to the first input's root name.
    lands_on: ClassVar[str] = ""
    # Inherited rungs a subclass consciously redefines; empty by default so an accidental shadow is an error.
    override_ok: ClassVar[frozenset[str]] = frozenset()
    # Test-harness knob for the machinery's own self-tests: validate but do not register. Never set in real contracts.
    register: ClassVar[bool] = True

    # --- derived by the machinery (never declared) ---
    name: ClassVar[str]
    family: ClassVar[str]
    null_policy: ClassVar[NullPolicy]
    nan_policy: ClassVar[NanPolicy]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Validate a concrete contract at import: completeness, naming, derivations, honesty, registration."""
        super().__init_subclass__(**kwargs)
        if not cls.__name__.startswith("Test"):
            return  # a capability mixin, not a concrete contract
        required = {
            requirement
            for klass in cls.__mro__
            for requirement in klass.__dict__.get("_requires", ())
            if isinstance(requirement, str)
        }
        missing = sorted(requirement for requirement in required if not hasattr(cls, requirement))
        if missing:
            msg = f"{cls.__name__} does not declare: {missing}"
            raise TypeError(msg)
        factory = inspect.unwrap(cls.factory)
        cls.name = factory.__name__
        expected = f"Test{_pascal(cls.name)}"
        if cls.__name__ != expected:
            msg = f"{cls.__name__} must be named {expected} (after {cls.name})"
            raise TypeError(msg)
        cls.family = _family_of(cls.name)
        if cls.name not in POLICIES:
            msg = f"{cls.name} has no declared policy in pomata._policy"
            raise TypeError(msg)
        cls.null_policy, cls.nan_policy = POLICIES[cls.name]
        if not cls.lands_on:
            cls.lands_on = cls.inputs[0]
        unknown = sorted(role for role in cls.inputs if role not in _ROLE_BUILDERS)
        if unknown:
            msg = f"{cls.__name__} declares inputs the probe frame cannot build: {unknown}"
            raise TypeError(msg)
        if cls.params and not cls.raises:
            msg = f"{cls.__name__} declares params but no raises counterexamples — the validation rung would be a no-op"
            raise TypeError(msg)
        shadowed = sorted(
            attribute
            for attribute in vars(cls)
            if attribute.startswith("test_")
            and any(attribute in vars(base) for base in cls.__mro__[1:])
            and attribute not in cls.override_ok
        )
        if shadowed:
            msg = f"{cls.__name__} overrides inherited rungs without override_ok consent: {shadowed}"
            raise TypeError(msg)
        if cls.register:
            already = REGISTRY.get(cls.name)
            if already is not None:
                msg = f"{cls.name} is claimed by both {already.__name__} and {cls.__name__}"
                raise TypeError(msg)
            REGISTRY[cls.name] = cls
        cls._stamp_hypothesis_rungs()

    @classmethod
    def _stamp_hypothesis_rungs(cls) -> None:
        """Give this contract its own copy of every Hypothesis rung.

        A ``@given`` method shared through inheritance is one function object executed by every contract class:
        Hypothesis rejects that (``differing_executors``) and its example database would key every contract's
        counterexamples to the same entry. Stamping a per-class wrapper keeps rung names and order structural
        while giving each contract an isolated Hypothesis identity.
        """
        rungs: dict[str, str] = {}
        for klass in reversed(cls.__mro__):
            declared = klass.__dict__.get("_RUNGS_HYPOTHESIS")
            if isinstance(declared, Mapping):
                rungs.update(cast("Mapping[str, str]", declared))
        for rung_name, impl_name in rungs.items():
            if rung_name in vars(cls):
                continue  # a consented override (already vetted by the shadow lock) wins over the stamp
            impl = cast("Callable[[Contract, st.DataObject], None]", getattr(cls, impl_name))
            rung = _wrap_rung(impl)
            rung.__name__ = rung_name
            rung.__qualname__ = f"{cls.__qualname__}.{rung_name}"
            rung.__doc__ = impl.__doc__
            setattr(cls, rung_name, given(data=st.data())(rung))

    # Validation counterexamples: each entry is (kwargs overriding ``params``, the ValueError match regex).
    raises: ClassVar[tuple[tuple[Mapping[str, int | float | bool], str], ...]] = ()
    # Rows after an interior missing bar beyond which the declared flow must have played out (recovered or latched);
    # defaults to warm-up + the widest window; declare it only where an output is displaced (e.g. ichimoku).
    flow_horizon: ClassVar[int] = -1

    # --- shared probes ---

    def _expression(self, **overrides: float | bool) -> pl.Expr:
        columns = (pl.col(role) for role in type(self).inputs)
        return type(self).factory(*columns, **{**type(self).params, **overrides})

    def _probe_length(self) -> int:
        # Long enough that the flow probe's post-horizon tail is never empty: injection + horizon + margin.
        return self._widest_warmup() + 3 + self._horizon() + 8

    def _widest_window(self) -> int:
        windows = [int(v) for k, v in type(self).params.items() if k.startswith("window") and isinstance(v, int)]
        return max(windows) if windows else 1

    def _widest_warmup(self) -> int:
        warmup = cast("int | Mapping[str, int]", getattr(type(self), "warmup", 0))
        if isinstance(warmup, Mapping):
            return max(warmup.values())
        return warmup

    def _horizon(self) -> int:
        declared = type(self).flow_horizon
        return declared if declared >= 0 else self._widest_warmup() + self._widest_window() + 2

    def _lanes(self, out: pl.DataFrame) -> list[pl.Series]:
        """Every scalar lane of a computed output — the struct's fields expanded, so no lane is ever skipped."""
        schema = out.schema["out"]
        if isinstance(schema, pl.Struct):
            return [out["out"].struct.field(field.name) for field in schema.fields]
        return [out["out"]]

    def _flat(self, frame: pl.DataFrame) -> list[pl.Series]:
        """The output lanes of the contract's expression applied to ``frame``."""
        return self._lanes(frame.select(self._expression().alias("out")))

    # --- universal rungs: identical for every public function, so they live exactly once ---

    def test_returns_expr(self) -> None:
        """Verifies the factory returns a free-standing ``pl.Expr`` without touching any data."""
        assert isinstance(self._expression(), pl.Expr)

    def test_output_lands_on_declared_column(self) -> None:
        """Verifies the output keeps the declared root name on distinctly-named input columns."""
        frame = probe_frame(type(self).inputs, 16)
        out = frame.select(self._expression())
        assert out.columns == [type(self).lands_on]

    def test_lazy_eager_parity(self) -> None:
        """Verifies the lazy plan collects to exactly the eager result."""
        frame = probe_frame(type(self).inputs, self._probe_length())
        eager = frame.select(self._expression().alias("out"))
        lazy = frame.lazy().select(self._expression().alias("out")).collect()
        assert eager.equals(lazy)

    def test_bare_string_raises_type_error(self) -> None:
        """Verifies the shared input guard rejects a bare column name on the first input."""
        arguments: list[object] = ["close", *[pl.col(role) for role in type(self).inputs[1:]]]
        with pytest.raises(TypeError, match=r"expected a Polars expression"):
            type(self).factory(*arguments, **type(self).params)

    def test_invalid_params_raise(self) -> None:
        """Verifies every declared validation counterexample raises with its canonical message."""
        for overrides, match in type(self).raises:
            with pytest.raises(ValueError, match=match):
                self._expression(**overrides)

    def test_all_null_input(self) -> None:
        """Verifies an all-null input yields an all-null output on every lane."""
        frame = pl.DataFrame({role: pl.Series([None] * 12, dtype=pl.Float64) for role in type(self).inputs})
        for lane in self._flat(frame):
            assert lane.is_null().all(), lane.name

    def test_over_partitions_independently(self) -> None:
        """Verifies two stacked series under ``.over`` reproduce each series computed alone."""
        length = self._probe_length()
        first = probe_frame(type(self).inputs, length)
        second = probe_frame(type(self).inputs, length).select(pl.all() * 3.0)
        stacked = pl.concat([first, second]).with_columns(
            pl.Series("group", ["a"] * length + ["b"] * length),
        )
        self._assert_over_partitions(stacked, first, second)

    def _assert_over_partitions(self, stacked: pl.DataFrame, first: pl.DataFrame, second: pl.DataFrame) -> None:
        grouped = stacked.select(self._expression().over("group").alias("out"))
        alone = pl.concat(
            [
                first.select(self._expression().alias("out")),
                second.select(self._expression().alias("out")),
            ]
        )
        for lane_grouped, lane_alone in zip(self._lanes(grouped), self._lanes(alone), strict=True):
            # assert_matches, not bit-equality: the grouped and the standalone paths may round differently.
            assert_matches(lane_grouped.to_list(), lane_alone.to_list())

    def test_interior_null_flow(self) -> None:
        """Verifies an interior missing bar plays out exactly as the declared null policy states."""
        self._assert_flow(float("nan"), nan=False)

    def test_interior_nan_flow(self) -> None:
        """Verifies an interior NaN bar plays out exactly as the declared NaN policy states."""
        self._assert_flow(float("nan"), nan=True)

    def _assert_flow(self, _sentinel: float, *, nan: bool) -> None:
        length = self._probe_length()
        injection = self._widest_warmup() + 3
        clean = probe_frame(type(self).inputs, length)
        poisoned = clean.with_columns(
            pl.when(pl.int_range(pl.len()) == injection)
            .then(pl.lit(float("nan"), dtype=pl.Float64) if nan else pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col(role))
            .alias(role)
            for role in type(self).inputs
        )
        policy: NullPolicy | NanPolicy = type(self).nan_policy if nan else type(self).null_policy
        baseline_lanes = self._flat(clean)
        poisoned_lanes = self._flat(poisoned)
        horizon = self._horizon()
        for baseline, poisoned_lane in zip(baseline_lanes, poisoned_lanes, strict=True):
            if policy in (NullPolicy.SKIPPED, NanPolicy.POISONS):
                self._assert_reducing_flow(clean, injection, poisoned_lane, nan=nan)
            elif policy in (NullPolicy.LATCHES, NanPolicy.LATCHES):
                tail = poisoned_lane.slice(injection + horizon)
                assert tail.len() > 0, "the probe leaves no tail to observe — lengthen the probe"
                defined = tail.is_not_null() & tail.is_not_nan().fill_null(value=False)
                assert not bool(defined.any()), f"{poisoned_lane.name}: a latched lane recovered"
            else:  # ABSORBED, PROPAGATES, IN_WINDOW_IS_NULL, BRIDGED — the effect is bounded and the lane recovers
                affected = poisoned_lane.slice(injection, horizon)
                if policy is not NullPolicy.ABSORBED:
                    hit = affected.is_null() if not nan else (affected.is_nan().fill_null(value=True))
                    assert bool(hit.any()), f"{poisoned_lane.name}: the bar left no trace"
                tail = poisoned_lane.slice(injection + horizon)
                assert tail.len() > 0, "the probe leaves no tail to observe — lengthen the probe"
                assert not bool(tail.is_null().any()), f"{poisoned_lane.name}: did not recover to defined values"
                assert not bool(tail.is_nan().any()), f"{poisoned_lane.name}: NaN survived past the horizon"
                if policy in (NullPolicy.PROPAGATES, NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES):
                    expected_tail = baseline.slice(injection + horizon)
                    assert tail.equals(expected_tail), f"{poisoned_lane.name}: tail drifted from the clean baseline"

    def _assert_reducing_flow(self, clean: pl.DataFrame, injection: int, poisoned: pl.Series, *, nan: bool) -> None:
        if nan:  # POISONS: the scalar goes NaN
            assert poisoned.len() == 1
            value = cast("float | None", poisoned.item())
            assert value is not None, "a NaN input nulled the reduction instead of poisoning it"
            assert math.isnan(value), "a NaN input did not poison the reduction"
            return
        # SKIPPED: the scalar is exactly what it would be if the row were absent.
        without_row = clean.with_row_index().filter(pl.col("index") != injection).drop("index")
        expected = without_row.select(self._expression().alias("out"))["out"]
        assert poisoned.equals(expected), "a skipped null changed the reduction"


class ContractReducing(Contract):
    """A whole-series reduction: one output row, whatever the input length."""

    def test_reduces_to_one_row(self) -> None:
        """Verifies the expression reduces the probe frame to a single row."""
        frame = probe_frame(type(self).inputs, 16)
        assert frame.select(self._expression().alias("out")).height == 1

    def _assert_over_partitions(self, stacked: pl.DataFrame, first: pl.DataFrame, second: pl.DataFrame) -> None:
        # ``.over`` broadcasts a reduction across its group's rows: each row carries the reduction of its own
        # series computed alone (assert_matches, not bit-equality — the two paths may round differently).
        grouped = stacked.select(self._expression().over("group").alias("out"))
        length = first.height
        alone_first = first.select(self._expression().alias("out"))["out"].to_list()
        alone_second = second.select(self._expression().alias("out"))["out"].to_list()
        expected = alone_first * length + alone_second * length
        assert_matches(grouped["out"].to_list(), expected)


class ContractSeries(Contract):
    """An elementwise transform: a same-length ``Float64`` series."""

    def test_preserves_length_and_dtype(self) -> None:
        """Verifies the output is one same-length ``Float64`` column."""
        frame = probe_frame(type(self).inputs, 16)
        out = frame.select(self._expression().alias("out"))
        assert out.height == frame.height
        assert out.schema["out"] == pl.Float64


class ContractStruct(Contract):
    """A multi-line output: one ``Struct`` column whose fields are declared, ordered, and all ``Float64``."""

    _requires: ClassVar[tuple[str, ...]] = ("fields",)

    fields: ClassVar[tuple[str, ...]]

    def test_struct_fields_names_order_dtypes(self) -> None:
        """Verifies every declared field, in order, and that each one is ``Float64`` — never only the first."""
        frame = probe_frame(type(self).inputs, 16)
        out = frame.select(self._expression().alias("out"))
        schema = out.schema["out"]
        assert isinstance(schema, pl.Struct)
        assert tuple(field.name for field in schema.fields) == type(self).fields
        for field in schema.fields:
            assert field.dtype == pl.Float64, f"{field.name} is {field.dtype}"

    def test_preserves_length(self) -> None:
        """Verifies the struct column is same-length (a struct output is elementwise by construction here)."""
        frame = probe_frame(type(self).inputs, 16)
        assert frame.select(self._expression().alias("out")).height == frame.height


class ContractWindowed(Contract):
    """A lookback function: declares its exact warm-up under the canonical ``params``."""

    _requires: ClassVar[tuple[str, ...]] = ("warmup",)

    # The exact number of leading null rows under ``params``; a mapping gives the per-field counts of a struct.
    warmup: ClassVar[int | Mapping[str, int]]

    def _warmup_probe_length(self) -> int:
        warmup = type(self).warmup
        widest = warmup if isinstance(warmup, int) else max(warmup.values())
        return widest + 8

    def test_warmup_null_count(self) -> None:
        """Verifies the output carries exactly the declared leading nulls — per field, for a struct."""
        frame = probe_frame(type(self).inputs, self._warmup_probe_length())
        out = frame.select(self._expression().alias("out"))
        warmup = type(self).warmup
        schema = out.schema["out"]
        if isinstance(schema, pl.Struct):
            per_field = warmup if isinstance(warmup, Mapping) else {field.name: warmup for field in schema.fields}
            assert sorted(per_field) == sorted(field.name for field in schema.fields)
            for field_name, field_warmup in per_field.items():
                observed = count_leading_nulls(out["out"].struct.field(field_name).to_list())
                assert observed == field_warmup, f"{field_name}: {observed} != {field_warmup}"
        else:
            assert isinstance(warmup, int)
            assert count_leading_nulls(out["out"].to_list()) == warmup


class ContractCorrectness(Contract):
    """Agreement with the independent oracle and a frozen golden master — the two Correctness rungs."""

    _requires: ClassVar[tuple[str, ...]] = ("oracle", "golden_input", "golden_output")

    # The naive reference; by default it mirrors the factory (positional input lists + ``params`` as kwargs) —
    # a contract whose oracle deviates (different kwarg names, shared mirror) overrides ``_reference`` instead.
    oracle: ClassVar[Callable[..., object]]
    golden_input: ClassVar[Mapping[str, tuple[float | None, ...]]]
    # The golden's own parameters, where they differ from the canonical ``params`` (e.g. a smaller window triple).
    golden_params: ClassVar[Mapping[str, float | bool]] = {}
    # Full expected output, one entry per input row (per field for a struct); rounded via ``golden_round``.
    golden_output: ClassVar[tuple[float | None, ...] | Mapping[str, tuple[float | None, ...]]]
    golden_round: ClassVar[int] = 4

    def _reference(self, frame: pl.DataFrame) -> object:
        lists = [frame[role].to_list() for role in type(self).inputs]
        return type(self).oracle(*lists, **type(self).params)

    def _reference_lanes(self, frame: pl.DataFrame) -> dict[str, list[float | None]]:
        reference = self._reference(frame)
        if isinstance(reference, Mapping):
            fields = cast("Mapping[str, list[float | None]]", reference)
            return {str(field): list(values) for field, values in fields.items()}
        if isinstance(reference, list):
            return {"out": cast("list[float | None]", reference)}
        return {"out": [cast("float | None", reference)]}

    def _assert_reference_agreement(self, frame: pl.DataFrame, rel_tol: float, abs_tol: float) -> None:
        expected = self._reference_lanes(frame)
        lanes = self._flat(frame)
        actual = {lane.name if len(lanes) > 1 else "out": lane.to_list() for lane in lanes}
        assert sorted(actual) == sorted(expected)
        for field, values in expected.items():
            assert_matches(actual[field], values, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_matches_reference(self) -> None:
        """Verifies agreement with the oracle on the deterministic probe frame, at the reference tier."""
        frame = probe_frame(type(self).inputs, self._probe_length())
        self._assert_reference_agreement(frame, RELATIVE_TOLERANCE_REFERENCE, ABSOLUTE_TOLERANCE_REFERENCE)

    def test_golden_master(self) -> None:
        """Verifies the frozen golden master, rounded expression-side so it can never flake cross-platform."""
        frame = pl.DataFrame(
            {role: pl.Series(list(values), dtype=pl.Float64) for role, values in type(self).golden_input.items()}
        )
        columns = (pl.col(role) for role in type(self).inputs)
        expr = type(self).factory(*columns, **{**type(self).params, **type(self).golden_params})
        out = frame.select(expr.alias("out"))
        schema = out.schema["out"]
        expected = type(self).golden_output
        if isinstance(schema, pl.Struct):
            assert isinstance(expected, Mapping)
            for field in schema.fields:
                lane = out["out"].struct.field(field.name).round(type(self).golden_round).to_list()
                assert_matches(lane, list(expected[field.name]))
        else:
            assert not isinstance(expected, Mapping)
            lane = out["out"].round(type(self).golden_round).to_list()
            assert_matches(lane, list(expected))


class ContractProperties(ContractCorrectness):
    """The property tier: oracle agreement under fuzzed inputs, clean and with missing data.

    Inherits :class:`ContractCorrectness` by design — the property rungs *are* oracle-agreement rungs, so a
    contract cannot compose the fuzz tier without declaring the oracle it fuzzes against.
    """

    def _cases(self, *, missing: bool) -> st.SearchStrategy[pl.DataFrame]:
        minimum = self._widest_warmup() + 4
        length = st.integers(min_value=minimum, max_value=minimum + 24)
        inputs = type(self).inputs
        if inputs == ("high", "low"):
            bars = coherent_hl_with_missing() if missing else coherent_hl()
            return length.flatmap(
                lambda n: st.lists(bars, min_size=n, max_size=n).map(
                    lambda rows: pl.DataFrame(dict(zip(("high", "low"), split_pairs(rows), strict=True)))
                )
            )
        if len(inputs) == 1:
            values = missing_data_floats() if missing else finite_floats()
            role = inputs[0]
            return length.flatmap(
                lambda n: st.lists(values, min_size=n, max_size=n).map(
                    lambda rows: pl.DataFrame({role: pl.Series(rows, dtype=pl.Float64)})
                )
            )
        msg = f"no fuzz strategy for inputs {inputs}"  # extended as the rollout reaches new input shapes
        raise TypeError(msg)

    # Hypothesis rungs are stamped per concrete class by the machinery (``_stamp_hypothesis_rungs``): a shared
    # ``@given`` method would be one function object executed by every contract, which Hypothesis rejects
    # (``differing_executors``) and which would cross-contaminate its example database between contracts.
    _RUNGS_HYPOTHESIS: ClassVar[Mapping[str, str]] = {
        "test_matches_reference_for_any_input": "_rung_matches_reference_for_any_input",
        "test_matches_reference_under_missing_data": "_rung_matches_reference_under_missing_data",
    }

    def _rung_matches_reference_for_any_input(self, data: st.DataObject) -> None:
        """Verifies oracle agreement over the fuzz domain, at the property tier."""
        frame = data.draw(self._cases(missing=False))
        self._assert_reference_agreement(frame, RELATIVE_TOLERANCE_PROPERTY, ABSOLUTE_TOLERANCE_PROPERTY)

    def _rung_matches_reference_under_missing_data(self, data: st.DataObject) -> None:
        """Verifies oracle agreement when the fuzz mixes interior nulls and NaNs into the input."""
        frame = data.draw(self._cases(missing=True))
        self._assert_reference_agreement(frame, RELATIVE_TOLERANCE_PROPERTY, ABSOLUTE_TOLERANCE_PROPERTY)
