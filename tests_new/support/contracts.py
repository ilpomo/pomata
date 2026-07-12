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
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import ClassVar

import polars as pl
from tests.support import count_leading_nulls

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

    # --- universal rungs: identical for every public function, so they live exactly once ---

    def _expression(self) -> pl.Expr:
        columns = (pl.col(role) for role in type(self).inputs)
        return type(self).factory(*columns, **type(self).params)

    def test_returns_expr(self) -> None:
        """Verifies the factory returns a free-standing ``pl.Expr`` without touching any data."""
        assert isinstance(self._expression(), pl.Expr)

    def test_output_lands_on_declared_column(self) -> None:
        """Verifies the output keeps the declared root name on distinctly-named input columns."""
        frame = probe_frame(type(self).inputs, 16)
        out = frame.select(self._expression())
        assert out.columns == [type(self).lands_on]


class ReducingContract(Contract):
    """A whole-series reduction: one output row, whatever the input length."""

    def test_reduces_to_one_row(self) -> None:
        """Verifies the expression reduces the probe frame to a single row."""
        frame = probe_frame(type(self).inputs, 16)
        assert frame.select(self._expression().alias("out")).height == 1


class SeriesContract(Contract):
    """An elementwise transform: a same-length ``Float64`` series."""

    def test_preserves_length_and_dtype(self) -> None:
        """Verifies the output is one same-length ``Float64`` column."""
        frame = probe_frame(type(self).inputs, 16)
        out = frame.select(self._expression().alias("out"))
        assert out.height == frame.height
        assert out.schema["out"] == pl.Float64


class StructContract(Contract):
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


class WindowedContract(Contract):
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
