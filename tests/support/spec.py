"""
The spec engine: a per-function contract is a frozen dataclass of pure data, plus the small engine the rungs delegate
to.

There is no metaprogramming here — no metaclass, no ``__init_subclass__``, no runtime function stamping. A :class:`Spec`
is a row of declarations: its required fields have no default, so a function *cannot* be declared without stating each,
and its conditional requirements (a struct names its fields, params imply raises, a scale claim is never an empty
tuple) are checked in a plain :meth:`Spec.__post_init__`. Deviations are declared as data — an ``oracle_adapter`` when
the oracle's signature is not a mirror of the factory's, a :class:`Deviant` when a documented input has a non-null
answer, a :class:`ScaleExempt` when a function is scale-exempt by design — never as a method override.

The rungs live in ``tests/test_ladder.py``; each is one module-level function parametrized over the applicable
spec subset. This module holds the data types and the engine (the deterministic probe frame, the expression builder,
the lane readers, the oracle bridge, the fuzz strategies, and the sizing helpers) they call.
"""

import enum
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import cast

import polars as pl
from hypothesis import strategies as st
from tests.support.bars import split_pairs, split_quads, split_triples
from tests.support.strategies import (
    apply_missing,
    coherent_hl,
    coherent_hl_with_missing,
    coherent_hlc,
    coherent_hlc_with_missing,
    coherent_hlcv,
    coherent_hlcv_with_missing,
    coherent_ohlc,
    coherent_ohlc_with_missing,
    finite_floats,
    missing_data_floats,
)

import pomata.indicators
import pomata.metrics
import pomata.pnl
from pomata._policy import POLICIES, NanPolicy, NullPolicy

SPEC_SCALAR = int | float | bool
SPEC_LANE = tuple[float | None, ...]


def _no_params() -> dict[str, SPEC_SCALAR]:
    return {}


_SPEC_FAMILIES: dict[str, ModuleType] = {
    "indicators": pomata.indicators,
    "metrics": pomata.metrics,
    "pnl": pomata.pnl,
}


class Shape(enum.Enum):
    """What one probe row observes — stated, not derived, so a rung can gate its applicability by hand."""

    REDUCING = "reducing"  # one output row, whatever the input length
    SERIES = "series"  # one same-length Float64 column
    STRUCT = "struct"  # one Struct column, fields declared and ordered


# The input-column roles the deterministic probe frame knows how to synthesize; a spec's ``inputs`` draw from these.
_SPEC_ROLE_BUILDERS: dict[str, Callable[[int], list[float]]] = {
    "high": lambda n: [float(i) + 1.5 for i in range(n)],
    "low": lambda n: [float(i) + 0.5 for i in range(n)],
    "open": lambda n: [float(i) + 0.9 for i in range(n)],
    "close": lambda n: [float(i) + 1.1 for i in range(n)],
    "volume": lambda n: [100.0 + float(i) for i in range(n)],
    "expr": lambda n: [float(i) + 1.0 for i in range(n)],
    # A non-monotone oscillating price series: a strictly-monotone probe saturates any RSI to a flat 100, which then
    # drives a stochastic-of-RSI to a 0/0 all-NaN output the flow rungs cannot read. This role gives such a function a
    # varied, well-defined baseline instead (rsi_stochastic).
    "wave": lambda n: [100.0 + 10.0 * math.sin(0.7 * float(i)) for i in range(n)],
    "price": lambda n: [float(i) + 10.0 for i in range(n)],
    # A strictly-positive compounding path whose step is small enough that the benchmark tier's frames (up to 1M
    # rows) stay far below the float64 ceiling (1.0001 ** n overflows only past ~7.1M rows; 1.02 ** n blows up at
    # ~35.7k).
    "equity_curve": lambda n: [100.0 * (1.0001 ** float(i)) for i in range(n)],
    "returns": lambda n: [0.01 if i % 2 == 0 else -0.005 for i in range(n)],
    "benchmark": lambda n: [0.008 if i % 2 == 0 else -0.004 for i in range(n)],
    "asset_returns": lambda n: [0.01 if i % 2 == 0 else -0.005 for i in range(n)],
    "weight": lambda n: [0.5 + 0.01 * float(i) for i in range(n)],
    # Alternating long/short so the short-only branches (cost_borrow's max(-q, 0)) meet real shorts on the probe:
    # an all-positive quantity would leave them identically zero and their scale axes vacuously green.
    "quantity": lambda n: [(-1.0) ** i * (10.0 + float(i % 3)) for i in range(n)],
    "cost": lambda n: [0.1 for _ in range(n)],
    "dividend_per_share": lambda n: [0.05 for _ in range(n)],
    "returns_gross": lambda n: [0.01 if i % 2 == 0 else -0.005 for i in range(n)],
    "funding_rate": lambda n: [0.0001 for _ in range(n)],
    "pnl_gross": lambda n: [10.0 + float(i) for i in range(n)],
}


@dataclass(frozen=True)
class Deviant:
    """A documented non-default answer, stated as data with its reason — never a silent special case in a rung."""

    expected: object
    reason: str


@dataclass(frozen=True)
class ScaleAxis:
    """
    One homogeneity claim: scaling the named ``roles`` by ``k`` scales the output by ``k ** degree``.

    The ``degree`` form is fixed by the spec's shape, so a reader is never left guessing which lanes a number
    covers: a single-lane output (SERIES / REDUCING) declares one ``int``, a STRUCT declares one degree per field
    (a mapping keyed exactly by its ``fields``) — a bare int on a struct and a mapping on a single lane are both
    rejected at construction (e.g. supertrend's ``line`` rides the price scale at degree 1 while its ``direction``
    flag is degree-0 invariant under the same input rescale).
    """

    roles: tuple[str, ...]
    degree: int | Mapping[str, int]


@dataclass(frozen=True)
class ScaleExempt:
    """A function that is neither scale-invariant nor homogeneous, declared exempt with its documented reason."""

    reason: str


@dataclass(frozen=True, kw_only=True)
class SpecPin:
    """
    One crafted-input case: fixed input lanes mapped to fixed output lanes, with the reason it exists. It is the data
    home for a hand-computed golden, a domain corner, or a signed-zero case — a fact a probe row or the reference
    oracle cannot express on its own.
    """

    label: str  # the case's short name; the pin's pytest id is ``{function}-{label}``
    inputs: Mapping[str, SPEC_LANE]  # the full input lanes, one per input role
    expected: SPEC_LANE | Mapping[str, SPEC_LANE]  # the expected output lanes (a per-field mapping for a struct)
    reason: str  # why the case is pinned
    params_override: Mapping[str, SPEC_SCALAR] = field(default_factory=_no_params)  # kwargs overriding ``params``
    # Compare the sign as well as the value: ``assert_matches`` reads ``-0.0`` and ``0.0`` as equal, so a case that
    # pins the sign of a zero sets this and the rung checks ``math.copysign`` on each pair.
    signed: bool = False
    # This pin witnesses the degenerate regime the spec's ``conditioning`` filter excludes from the property tiers:
    # a filter is only allowed together with at least one such pin (checked in ``__post_init__``), so no input regime
    # is ever silently asserted away — what the fuzz never sees, a fixed case must still demonstrate.
    covers_conditioning: bool = False
    # Expression-side rounding for the comparison, mirroring ``golden_round``: declared ONLY where the exact lanes
    # are platform-dependent (a transcendental pipeline evaluated on a degenerate input settles on libm-rounded
    # fixed points that differ across OS math libraries), with the reason naming that fact. ``None`` compares exact.
    round_to: int | None = None


@dataclass(frozen=True, kw_only=True)
class Spec:
    """
    One public function's whole contract, as data. The required fields have no default: a new function *cannot* be
    declared without stating each, so the language itself enforces completeness and no rung is ever silently skipped.
    """

    # --- required of every function (no defaults: the language is the completeness lock) ---
    factory: Callable[..., pl.Expr]
    inputs: tuple[str, ...]
    params: Mapping[str, SPEC_SCALAR]
    shape: Shape
    # A non-empty tuple of homogeneity axes, or a ``ScaleExempt`` for a function that is scale-exempt by design; an
    # empty tuple is rejected in ``__post_init__`` so "no scale claim" is always a deliberate, reasoned exemption.
    scale: tuple[ScaleAxis, ...] | ScaleExempt
    # The naive reference oracle and the frozen golden master (a per-field mapping for a struct).
    oracle: Callable[..., object]
    golden_input: Mapping[str, SPEC_LANE]
    golden_output: SPEC_LANE | Mapping[str, SPEC_LANE]

    # --- optional, with defaults ---
    # Exact leading-null count under ``params``: an int for a windowed series, a per-field mapping for a struct, or
    # ``None`` for a reduction or an unwindowed transform (there is nothing to warm up).
    warmup: int | Mapping[str, int] | None = None
    fields: tuple[str, ...] = ()  # required non-empty for a struct, empty otherwise
    # Validation counterexamples: each is (kwargs overriding ``params``, the ValueError match regex).
    raises: tuple[tuple[Mapping[str, SPEC_SCALAR], str], ...] = ()
    golden_params: Mapping[str, SPEC_SCALAR] = field(default_factory=_no_params)
    golden_round: int = 4
    lands_on: str = ""  # the landing column; defaults to the first input's root name
    # Rows past an interior missing bar beyond which the declared flow must have played out; ``-1`` derives it from
    # the warm-up and the widest window (declare a positive value only where an output is displaced further).
    flow_horizon: int = -1
    # A reason a function's interior-missing-bar flow is input-dependent and so cannot be expressed by the shared
    # policy rung: a directional-movement guard turns a fully-missing bar into neutral 0 movement (a full-bar null /
    # NaN is absorbed and the recursion continues at 0), while a single-column NaN on the driving leg still latches —
    # two behaviors one declared policy cannot hold. Non-empty exempts the function from the two flow rungs; its flow
    # is pinned as crafted cases and covered by the missing-data property tier instead. Empty = the flow rungs apply.
    flow_deviation: str = ""
    # A deviation stated as data: an oracle whose signature is not a mirror of the factory's (different kwarg names)
    # supplies a frame->result callable here. ``None`` means "mirror the signature" (positional inputs, params kwargs).
    oracle_adapter: "Callable[[Spec, pl.DataFrame], object] | None" = None
    # An optional Hypothesis filter on a fuzzed frame, applied through ``assume`` in the property tier (e.g. mama's
    # even-lag guard, the standardized moments' well-spread guard) — the input regimes where the impl and the oracle
    # cannot be expected to agree, excluded as data rather than silently in a rung.
    conditioning: "Callable[[pl.DataFrame], bool] | None" = None
    # The documented answer to an all-null input; ``None`` means the answer is all-null (the ordinary case).
    all_null: Deviant | None = None
    # Crafted-input cases (hand-computed goldens, domain corners, signed-zero pins): each maps fixed input lanes to
    # fixed output lanes, the data home for a fact a probe or an oracle cannot express.
    pins: tuple[SpecPin, ...] = ()
    # The public-function recomposition this factory must reproduce (a metamorphic identity), as a zero-argument
    # expression builder; ``None`` when the function has no such definition. Compared to the factory on the probe frame.
    component_expr: Callable[[], pl.Expr] | None = None
    # The oracle-agreement band, declared only where a one-pass rolling form cannot meet the tight default against its
    # two-pass oracle. ``None`` uses the tier default.
    oracle_rel_tol: float | None = None
    oracle_abs_tol: float | None = None
    # The kernel's polynomial cost degree in the ROW COUNT: ``t(n) ~ c * n**cost_degree`` (logarithmic factors ride
    # within a degree — one timed decade cannot resolve them, and no contract claims them). The scaling guard derives
    # its per-function bound from this declaration; ``1`` is the family norm and no current spec deviates. Distinct
    # from a :class:`ScaleAxis` ``degree``, which is homogeneity in the input VALUES, not cost in the row count.
    cost_degree: int = 1

    def __post_init__(self) -> None:
        """Conditional requirements, checked loudly at construction (import time) — the one obvious place they live."""
        self._check_inputs()
        self._check_fields()
        self._check_warmup()
        if self.params and not self.raises:
            msg = f"{self.name}: declares params but no raises counterexamples — the validation rung would be a no-op"
            raise ValueError(msg)
        if self.cost_degree < 1:
            msg = f"{self.name}: cost_degree must be >= 1, got {self.cost_degree}"
            raise ValueError(msg)
        self._check_scale()
        self._check_golden()
        self._check_pins()
        if self.name not in POLICIES:
            msg = f"{self.name}: no declared policy in pomata._policy (the name is derived from the factory)"
            raise ValueError(msg)
        if self.name not in _family_index():
            msg = f"{self.name}: the derived name is in no public __all__"
            raise ValueError(msg)

    def _check_inputs(self) -> None:
        unknown = sorted(role for role in self.inputs if role not in _SPEC_ROLE_BUILDERS)
        if not self.inputs or unknown:
            msg = f"{self.name}: inputs must be non-empty roles the probe frame can build; unknown: {unknown}"
            raise ValueError(msg)

    def _check_fields(self) -> None:
        if self.shape is Shape.STRUCT and not self.fields:
            msg = f"{self.name}: a struct must declare its ordered fields"
            raise ValueError(msg)
        if self.shape is not Shape.STRUCT and self.fields:
            msg = f"{self.name}: only a struct declares fields, got {self.fields}"
            raise ValueError(msg)

    def _check_warmup(self) -> None:
        if self.shape is Shape.REDUCING and self.warmup is not None:
            msg = f"{self.name}: a reduction has no warm-up; declare warmup=None"
            raise ValueError(msg)
        if self.shape is Shape.STRUCT and self.warmup is not None and not isinstance(self.warmup, Mapping):
            msg = f"{self.name}: a struct declares its warm-up per field — a bare int hides which lanes it covers"
            raise ValueError(msg)
        if isinstance(self.warmup, Mapping) and (
            self.shape is not Shape.STRUCT or set(self.warmup) != set(self.fields)
        ):
            msg = f"{self.name}: a per-field warm-up mapping is keyed by a struct's fields, and only a struct's"
            raise ValueError(msg)

    def _check_scale(self) -> None:
        if isinstance(self.scale, ScaleExempt):
            return
        if not self.scale:
            msg = f"{self.name}: an empty scale tuple is never allowed — declare ScaleExempt(reason) instead"
            raise ValueError(msg)
        for axis in self.scale:
            unknown = sorted(role for role in axis.roles if role not in self.inputs)
            if not axis.roles or unknown:
                msg = f"{self.name}: a scale axis names input roles; unknown or empty: {unknown or axis.roles}"
                raise ValueError(msg)
            if self.shape is Shape.STRUCT and (
                not isinstance(axis.degree, Mapping) or set(axis.degree) != set(self.fields)
            ):
                msg = f"{self.name}: a struct's scale axis declares one degree per field — a bare int hides them"
                raise ValueError(msg)
            if self.shape is not Shape.STRUCT and isinstance(axis.degree, Mapping):
                msg = f"{self.name}: only a struct's scale axis maps degrees per field; declare a single int"
                raise ValueError(msg)

    def _check_golden(self) -> None:
        if set(self.golden_input) != set(self.inputs):
            msg = f"{self.name}: golden_input keys {sorted(self.golden_input)} must match inputs {list(self.inputs)}"
            raise ValueError(msg)
        struct_golden = isinstance(self.golden_output, Mapping)
        if struct_golden != (self.shape is Shape.STRUCT):
            msg = f"{self.name}: golden_output is a per-field mapping iff the shape is a struct"
            raise ValueError(msg)

    def _check_pins(self) -> None:
        labels = [pin.label for pin in self.pins]
        if len(set(labels)) != len(labels):
            msg = f"{self.name}: pin labels must be unique, got {sorted(labels)}"
            raise ValueError(msg)
        for pin in self.pins:
            if set(pin.inputs) != set(self.inputs):
                msg = f"{self.name}: pin {pin.label!r} inputs {sorted(pin.inputs)} must match {list(self.inputs)}"
                raise ValueError(msg)
            if isinstance(pin.expected, Mapping) != (self.shape is Shape.STRUCT):
                msg = f"{self.name}: pin {pin.label!r} expected is a per-field mapping iff the shape is a struct"
                raise ValueError(msg)
        covering = [pin.label for pin in self.pins if pin.covers_conditioning]
        if self.conditioning is not None and not covering:
            msg = (
                f"{self.name}: a conditioning filter excludes an input regime from the property tiers, so at least "
                "one pin must witness that regime (covers_conditioning=True) — no exclusion without a fixed case"
            )
            raise ValueError(msg)
        if self.conditioning is None and covering:
            msg = f"{self.name}: pins {covering} claim to cover a conditioning filter, but none is declared"
            raise ValueError(msg)

    # --- derived, never declared: read off the factory and the package registries ---

    @property
    def name(self) -> str:
        """The function's name, read off the factory — the key the id, the policy, and the family all derive from."""
        return self.factory.__name__

    @property
    def family(self) -> str:
        """The family the function belongs to, from the public ``__all__`` tuples."""
        return _family_index()[self.name]

    @property
    def null_policy(self) -> NullPolicy:
        """The declared interior-``null`` policy, from :mod:`pomata._policy`."""
        return POLICIES[self.name][0]

    @property
    def nan_policy(self) -> NanPolicy:
        """The declared interior-``NaN`` policy, from :mod:`pomata._policy`."""
        return POLICIES[self.name][1]

    @property
    def spec_id(self) -> str:
        """The pytest id for this spec — the function's name."""
        return self.name

    @property
    def landing(self) -> str:
        """The column the output lands on: the declared ``lands_on`` or, by default, the first input's root name."""
        return self.lands_on or self.inputs[0]


def _family_index() -> dict[str, str]:
    return {name: family for family, module in _SPEC_FAMILIES.items() for name in module.__all__}


def spec_id(spec: Spec) -> str:
    """The pytest id of a spec — its function name — as a module function for ``parametrize(ids=...)``."""
    return spec.spec_id


# --- the engine the rungs delegate to ---


def probe_frame(inputs: tuple[str, ...], length: int) -> pl.DataFrame:
    """A well-conditioned deterministic frame, one distinctly-named ``Float64`` column per declared input role."""
    return pl.DataFrame({role: pl.Series(_SPEC_ROLE_BUILDERS[role](length), dtype=pl.Float64) for role in inputs})


def build_expr(spec: Spec, **overrides: SPEC_SCALAR) -> pl.Expr:
    """The factory applied to its declared input columns under ``params`` (with optional per-call overrides)."""
    columns = (pl.col(role) for role in spec.inputs)
    return spec.factory(*columns, **{**spec.params, **overrides})


def lane_series(out: pl.DataFrame) -> list[pl.Series]:
    """Every scalar lane of a computed ``out`` column — a struct's fields expanded, so no lane is ever skipped."""
    schema = out.schema["out"]
    if isinstance(schema, pl.Struct):
        return [out["out"].struct.field(f.name) for f in schema.fields]
    return [out["out"]]


def flat(spec: Spec, frame: pl.DataFrame) -> list[pl.Series]:
    """The output lanes of the spec's expression applied to ``frame``."""
    return lane_series(frame.select(build_expr(spec).alias("out")))


def actual_lanes(spec: Spec, frame: pl.DataFrame) -> dict[str, list[float | None]]:
    """The expression's output as one named lane per line: struct field names for a struct, else ``{"out": ...}``."""
    lanes = flat(spec, frame)
    if len(lanes) > 1:
        return {lane.name: lane.to_list() for lane in lanes}
    return {"out": lanes[0].to_list()}


def reference_lanes(spec: Spec, frame: pl.DataFrame) -> dict[str, list[float | None]]:
    """The oracle's output as one named lane per line, matching :func:`actual_lanes`'s naming."""
    if spec.oracle_adapter is not None:
        result = spec.oracle_adapter(spec, frame)
    else:  # the signature-mirror: positional input lists, params as kwargs
        lists = [frame[role].to_list() for role in spec.inputs]
        result = spec.oracle(*lists, **spec.params)
    if isinstance(result, Mapping):
        mapping = cast("Mapping[str, Sequence[float | None]]", result)
        return {str(name): list(values) for name, values in mapping.items()}
    if isinstance(result, list):
        return {"out": cast("list[float | None]", result)}
    return {"out": [cast("float | None", result)]}


def widest_warmup(spec: Spec) -> int:
    """The widest declared leading-null count (0 for a reduction or an unwindowed transform), used to size frames."""
    if spec.warmup is None:
        return 0
    if isinstance(spec.warmup, Mapping):
        return max(spec.warmup.values())
    return spec.warmup


def widest_window(spec: Spec) -> int:
    """The widest ``window*`` integer parameter (1 when the function has none) — a term of the default flow horizon."""
    windows = [value for key, value in spec.params.items() if key.startswith("window") and isinstance(value, int)]
    return max(windows) if windows else 1


def horizon(spec: Spec) -> int:
    """Rows past an interior missing bar beyond which the declared flow must have played out."""
    return spec.flow_horizon if spec.flow_horizon >= 0 else widest_warmup(spec) + widest_window(spec) + 2


def probe_length(spec: Spec) -> int:
    """A probe length long enough that the flow rung's post-horizon tail is never empty."""
    return widest_warmup(spec) + 3 + horizon(spec) + 8


def _finite(low: float, high: float) -> st.SearchStrategy[float]:
    """Finite floats in ``[low, high]`` — the bounded element domain a multi-input column draws from."""
    return st.floats(min_value=low, max_value=high, allow_nan=False, allow_infinity=False)


# Per-role element domains for the multi-input fuzz vocabulary: each column of an input frame is drawn independently
# from the domain its role lives in — a signed quantity (long and short), a positive price, a bounded weight, a modest
# return or funding rate, a non-negative cost or dividend — so a multi-input factory meets its oracle on
# well-conditioned inputs.
_FUZZ_ELEMENT: dict[str, st.SearchStrategy[float]] = {
    # Signed and bounded away from zero on both sides, so the short-only branches are fuzzed too.
    "quantity": st.one_of(_finite(1e-3, 1e6), _finite(-1e6, -1e-3)),
    "price": _finite(1e-3, 1e6),
    "weight": _finite(-1.5, 1.5),
    "asset_returns": _finite(-0.5, 0.5),
    "returns_gross": _finite(-0.5, 0.5),
    "funding_rate": _finite(-0.5, 0.5),
    "dividend_per_share": _finite(0.0, 1e3),
    "cost": _finite(0.0, 1e6),
    "pnl_gross": _finite(-1e6, 1e6),
    # The two legs of a relative metric: modest returns bounded away from zero (|r| in [0.01, 0.5]) so an embedded
    # regression's variance stays well-conditioned, both legs stay commensurate, and a capture ratio's geometric power
    # never lands in the near-one catastrophic-cancellation band.
    "returns": st.one_of(_finite(0.01, 0.5), _finite(-0.5, -0.01)),
    "benchmark": st.one_of(_finite(0.01, 0.5), _finite(-0.5, -0.01)),
}

# The multi-input shapes the vocabulary supports, read off the pnl and benchmark-relative metrics factory signatures;
# every role appears in the probe-frame builders too, so each shape can back a real spec. Anything outside this closed
# set raises below.
_FUZZ_SHAPES: frozenset[tuple[str, ...]] = frozenset(
    {
        ("quantity", "price"),
        ("quantity", "price", "funding_rate"),
        ("quantity", "dividend_per_share"),
        ("weight", "asset_returns"),
        ("returns_gross", "cost"),
        ("pnl_gross", "cost"),
        ("returns", "benchmark"),
    }
)


# The equity-curve shape draws a positive growth path, not independent draws: a compounded series of small steps stays
# strictly positive (a growth factor is always > 0), so a drawdown-family metric and its oracle agree on the domain
# they are defined on, and its magnitude stays modest so an annualizing power never overflows.
_EQUITY_STEP = st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False)


def _cumulative_growth(steps: list[float]) -> list[float]:
    """A strictly-positive equity path: the running product of ``1 + step`` from a unit start (each step in ±0.1)."""
    path: list[float] = []
    level = 1.0
    for step in steps:
        level *= 1.0 + step
        path.append(level)
    return path


def _equity_frames(length: st.SearchStrategy[int], *, missing: bool) -> st.SearchStrategy[pl.DataFrame]:
    """Frames of one positive ``equity_curve`` column; when ``missing``, null / NaN are punched into the grown path."""

    def column(n: int) -> st.SearchStrategy[list[float | None]]:
        path = st.lists(_EQUITY_STEP, min_size=n, max_size=n).map(_cumulative_growth)
        if not missing:
            return cast("st.SearchStrategy[list[float | None]]", path)
        masks = st.lists(st.sampled_from((0, 1, 2)), min_size=n, max_size=n)
        return st.tuples(path, masks).map(
            lambda drawn: [apply_missing(value, choice) for value, choice in zip(drawn[0], drawn[1], strict=True)]
        )

    return length.flatmap(
        lambda n: column(n).map(lambda values: pl.DataFrame({"equity_curve": pl.Series(values, dtype=pl.Float64)}))
    )


def _independent_frame(
    roles: tuple[str, ...], length: st.SearchStrategy[int], *, missing: bool
) -> st.SearchStrategy[pl.DataFrame]:
    """Frames whose columns are drawn independently, each from its role's domain, mixing null / NaN when ``missing``."""

    def column(role: str) -> st.SearchStrategy[float | None]:
        element = _FUZZ_ELEMENT[role]
        return st.one_of(st.none(), st.just(math.nan), element) if missing else element

    return length.flatmap(
        lambda n: st.tuples(*(st.lists(column(role), min_size=n, max_size=n) for role in roles)).map(
            lambda drawn: pl.DataFrame(
                {role: pl.Series(values, dtype=pl.Float64) for role, values in zip(roles, drawn, strict=True)}
            )
        )
    )


def _bars_to_frames[T](
    roles: tuple[str, ...],
    bars: st.SearchStrategy[T],
    length: st.SearchStrategy[int],
    split: Callable[[list[T]], tuple[list[float | None], ...]],
) -> st.SearchStrategy[pl.DataFrame]:
    """A frame strategy: draw a list of coherent bars, then unzip it into the declared per-role columns."""
    return length.flatmap(
        lambda n: st.lists(bars, min_size=n, max_size=n).map(
            lambda rows: pl.DataFrame(dict(zip(roles, split(rows), strict=True)))
        )
    )


def _coherent_bar_frames(
    inputs: tuple[str, ...], length: st.SearchStrategy[int], *, missing: bool
) -> st.SearchStrategy[pl.DataFrame] | None:
    """Frames of coherent OHLC-family bars for the price-bar input shapes, or ``None`` when the shape is not one."""
    if inputs == ("high", "low"):
        return _bars_to_frames(inputs, coherent_hl_with_missing() if missing else coherent_hl(), length, split_pairs)
    if inputs == ("high", "low", "close"):
        return _bars_to_frames(
            inputs, coherent_hlc_with_missing() if missing else coherent_hlc(), length, split_triples
        )
    if inputs == ("high", "low", "close", "volume"):
        return _bars_to_frames(
            inputs, coherent_hlcv_with_missing() if missing else coherent_hlcv(), length, split_quads
        )
    if inputs == ("open", "high", "low", "close"):
        return _bars_to_frames(
            inputs, coherent_ohlc_with_missing() if missing else coherent_ohlc(), length, split_quads
        )
    return None


def fuzz_frames(spec: Spec, *, missing: bool) -> st.SearchStrategy[pl.DataFrame]:
    """A Hypothesis strategy of well-formed input frames for the property tier, keyed on the spec's input shape."""
    minimum = widest_warmup(spec) + 4
    length = st.integers(min_value=minimum, max_value=minimum + 24)
    bar_frames = _coherent_bar_frames(spec.inputs, length, missing=missing)
    if bar_frames is not None:
        return bar_frames
    if spec.inputs == ("equity_curve",):
        return _equity_frames(length, missing=missing)
    if len(spec.inputs) == 1:
        role = spec.inputs[0]
        if role == "returns":
            # A modest return domain bounded away from zero (|r| in [0.01, 1.0]): a one-pass rolling moment stays
            # well-conditioned against its two-pass oracle (a subnormal-magnitude or near-zero draw would round the two
            # apart).
            finite = st.one_of(_finite(0.01, 1.0), _finite(-1.0, -0.01))
            values = st.one_of(st.none(), st.just(math.nan), finite) if missing else finite
        elif role == "price":
            # Strictly positive prices in a modest band ([1.0, 1e3]): an indicator that divides by a moving average of
            # the price (a percentage oscillator) or forms a one-pass rolling sum over it (an adaptive mean) stays
            # well-conditioned against its two-pass oracle here, where a symmetric or near-zero draw would round the two
            # apart — a positive-only, modest-magnitude domain.
            positive = _finite(1.0, 1e3)
            values = st.one_of(st.none(), st.just(math.nan), positive) if missing else positive
        else:
            values = missing_data_floats() if missing else finite_floats()
        return length.flatmap(
            lambda n: st.lists(values, min_size=n, max_size=n).map(
                lambda rows: pl.DataFrame({role: pl.Series(rows, dtype=pl.Float64)})
            )
        )
    if spec.inputs == ("price", "volume"):
        # A modest positive band ([1, 1e3]) for both a price and its volume: a one-pass rolling volume-weighted sum
        # stays well-conditioned against its two-pass oracle here, where a wide product range would drop a small
        # window's contribution into the accumulated sum's rounding.
        element = _finite(1.0, 1e3)
        column = st.one_of(st.none(), st.just(math.nan), element) if missing else element
        return length.flatmap(
            lambda n: st.tuples(st.lists(column, min_size=n, max_size=n), st.lists(column, min_size=n, max_size=n)).map(
                lambda drawn: pl.DataFrame(
                    {"price": pl.Series(drawn[0], dtype=pl.Float64), "volume": pl.Series(drawn[1], dtype=pl.Float64)}
                )
            )
        )
    if spec.inputs in _FUZZ_SHAPES:
        return _independent_frame(spec.inputs, length, missing=missing)
    msg = f"{spec.name}: no fuzz strategy for inputs {spec.inputs}"  # extend when a spec introduces a new input shape
    raise TypeError(msg)


# The bit-constant battery: every column one repeated constant — the exact regime where the shipped kernels pin a
# zero dispersion to exactly zero (the float-residue closures tests/README.md documents) while a naive two-pass mean can
# keep a rounding residue. The constants are values whose k-copy Python mean rounds AWAY from the constant at some
# series
# lengths (three copies of 0.8 average to 0.8000000000000001; a power of two would divide back exactly at every
# length and never exercise the regime), and the lengths sweep the warm-up neighborhood because that rounding is
# length-dependent (seven copies of 0.8 average back exactly; three do not).
BIT_CONSTANT_VALUES: tuple[float, ...] = (0.8, 0.3, 0.7, 0.1)
BIT_CONSTANT_EXTRA_LENGTHS: tuple[int, ...] = (2, 3, 4, 8)

# The OHLC-family roles a bit-constant assignment must keep coherent (``low <= open, close <= high``): an incoherent
# constant bar would be out-of-domain input (see tests/support/strategies.py), not a regime the rung may probe.
_BAR_PRICE_ROLES: frozenset[str] = frozenset({"open", "high", "low", "close"})


def _coherent_bar_assignment(roles: tuple[str, ...], values: dict[str, float]) -> dict[str, float]:
    """
    Reassign the price legs of a bar-shaped role set by rank — high the largest, low the smallest — so the constant
    bar stays coherent; non-bar roles keep their drawn constant.
    """
    price_roles = [role for role in roles if role in _BAR_PRICE_ROLES]
    if len(price_roles) < 2:
        return values
    ranked = sorted((values[role] for role in price_roles), reverse=True)
    values["high"] = ranked[0]
    values["low"] = ranked[-1]
    interior = [role for role in price_roles if role not in ("high", "low")]
    values.update(zip(interior, ranked[1:-1], strict=True))
    return values


def bit_constant_frames(spec: Spec) -> list[pl.DataFrame]:
    """
    The spec's bit-constant probe frames: constants rotated across roles (distinct per role, bar-coherent), lengths
    swept around the warm-up, each also in a leading-null variant.
    """
    frames: list[pl.DataFrame] = []
    lengths = sorted({max(2, widest_warmup(spec) + extra) for extra in BIT_CONSTANT_EXTRA_LENGTHS})
    for shift in range(len(BIT_CONSTANT_VALUES)):
        assigned = {
            role: BIT_CONSTANT_VALUES[(index + shift) % len(BIT_CONSTANT_VALUES)]
            for index, role in enumerate(spec.inputs)
        }
        assigned = _coherent_bar_assignment(spec.inputs, assigned)
        for length in lengths:
            for leading_null in (False, True):
                columns: dict[str, pl.Series] = {}
                for role in spec.inputs:
                    values: list[float | None] = [assigned[role]] * length
                    if leading_null:
                        values[0] = None
                    columns[role] = pl.Series(values, dtype=pl.Float64)
                frames.append(pl.DataFrame(columns))
    return frames
