"""
The declaration: a per-function contract stated as one frozen dataclass of pure data, plus the small engine the rungs
delegate to.

A contributor never builds a :class:`Declaration` directly — a per-family suite function (``suite_pnl`` and its twins)
does, filling the family-specific enums and registering the result. What that function assembles is this row of
declarations: its required fields have no default, so a function *cannot* be declared without stating each, and its
conditional requirements (a struct names its fields, a golden's lanes match its inputs, an oracle is named for the
function it checks) are checked in a plain :meth:`Declaration.__post_init__`. There is no metaprogramming: no metaclass,
no ``__init_subclass__``, no runtime function stamping.

The rungs live in :mod:`tests.support.rungs`; this module holds the data types and the engine (the expression
builder, the lane readers, the oracle bridge, the sizing helpers) they and the synthesis builders call.
"""

import enum
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import polars as pl

from tests.support.frames import KNOWN_ROLES

ScalarParam = int | float | bool
Lane = tuple[float | None, ...]
# The reflective factory / oracle signatures: the ``...`` mirrors each public function's own parameter list, which the
# closed input vocabulary and per-function params make concrete at the call site. Named here (the one module the strict
# ``disallow-any-explicit`` gate exempts) so the family suite functions can reuse them without re-spelling the ``...``.
FactoryExpr = Callable[..., pl.Expr]
OracleFn = Callable[..., object]


def _no_params() -> dict[str, ScalarParam]:
    return {}


class Shape(enum.Enum):
    """What one probe row observes — stated, not derived, so a rung can gate its applicability by hand."""

    REDUCING = "reducing"  # one output row, whatever the input length
    SERIES = "series"  # one same-length Float64 column
    STRUCT = "struct"  # one Struct column, fields declared and ordered


@dataclass(frozen=True)
class ScaleAxis:
    """
    One homogeneity claim: scaling the named ``roles`` by ``k`` scales the output by ``k ** degree``.

    The ``degree`` form is fixed by the declaration's shape, so a reader is never left guessing which lanes a number
    covers: a single-lane output (SERIES / REDUCING) declares one ``int``, a STRUCT declares one degree per field (a
    mapping keyed exactly by its ``fields``) — a bare int on a struct and a mapping on a single lane are both rejected
    at construction.
    """

    roles: tuple[str, ...]
    degree: int | Mapping[str, int]


@dataclass(frozen=True)
class ScaleExempt:
    """A function that is neither scale-invariant nor homogeneous, declared exempt with its documented reason."""

    reason: str


@dataclass(frozen=True)
class Deviant:
    """A documented non-default answer for a degenerate regime, stated as data with its reason — never a silent rung."""

    expected: object
    reason: str


@dataclass(frozen=True, kw_only=True)
class Golden:
    """The frozen golden master: fixed input lanes mapped to fixed output lanes (a per-field mapping for a struct)."""

    inputs: Mapping[str, Lane]  # the full input lanes, one per input role
    output: Lane | Mapping[str, Lane]  # the expected output lanes (a per-field mapping for a struct)
    params: Mapping[str, ScalarParam] = field(default_factory=_no_params)  # kwargs overriding the declared params
    round_to: int = 4  # expression-side rounding so the master can never flake cross-platform


@dataclass(frozen=True, kw_only=True)
class Pin:
    """
    One crafted-input case: fixed input lanes mapped to fixed output lanes, with the reason it exists — the data home
    for a hand-computed value the synthesis and the oracle cannot derive on their own.
    """

    label: str  # the case's short name; the pin's pytest id is ``{function}-{label}``
    inputs: Mapping[str, Lane]  # the full input lanes, one per input role
    expected: Lane | Mapping[str, Lane]  # the expected output lanes (a per-field mapping for a struct)
    reason: str  # why the case is pinned
    params_override: Mapping[str, ScalarParam] = field(default_factory=_no_params)  # kwargs overriding the params
    # Compare the sign as well as the value: ``assert_matches`` reads ``-0.0`` and ``0.0`` as equal, so a case that
    # pins the sign of a zero sets this and the rung checks ``math.copysign`` on each pair.
    signed: bool = False
    # Expression-side rounding for the comparison: declared ONLY where the exact lanes are platform-dependent (a
    # transcendental pipeline on a degenerate input settles on libm-rounded fixed points). ``None`` compares exact.
    round_to: int | None = None
    # This pin witnesses the degenerate regime a ``conditioning`` filter excludes from the property tiers: a filter is
    # only allowed together with at least one such pin (checked in ``__post_init__``), so no input regime is ever
    # silently asserted away.
    covers_conditioning: bool = False


@dataclass(frozen=True, kw_only=True)
class Declaration:
    """
    One public function's whole testing contract, as data. The required fields have no default: a new function *cannot*
    be declared without stating each, so the language itself enforces completeness and no rung is ever silently skipped.
    """

    # --- required of every function (no defaults: the language is the completeness lock) ---
    family: str  # the family the function belongs to; routes it to the family registry
    factory: FactoryExpr  # the ``pl.Expr`` factory under test; ``name`` is read off it
    inputs: tuple[str, ...]  # ordered input column roles, drawn from the probe-frame vocabulary
    params: Mapping[str, ScalarParam]  # the default keyword arguments the factory is exercised under
    shape: Shape
    behavior_null: enum.Enum  # what an interior ``null`` does to the output (family dialect)
    behavior_nan: enum.Enum  # what an interior ``NaN`` does to the output (family dialect)
    oracle: OracleFn  # the naive reference, named ``reference_{name}``, mirroring the factory signature
    # A non-empty tuple of homogeneity axes, or a ``ScaleExempt`` for a function that is scale-exempt by design; an
    # empty tuple is rejected in ``__post_init__`` so "no scale claim" is always a deliberate, reasoned exemption.
    scaling: tuple[ScaleAxis, ...] | ScaleExempt

    # --- optional, with defaults ---
    # The pnl dialect's three axes, present only for a family that declares them (the pnl family fills each). A family
    # without them (metrics) leaves them ``None``: none gates a rung except ``nonfinite``, whose ``None`` skips the
    # IEEE-flow probes — a family that does not contract ``±inf`` flow (a metric's reduction over an infinite input is
    # an implementation-defined artifact the naive oracle does not model) simply declares no such contract.
    space: enum.Enum | None = None  # the units the output lives in (e.g. cash flow vs returns flow)
    sign: enum.Enum | None = None  # the sign convention the payoff follows
    nonfinite: enum.Enum | None = None  # how the function carries ``±inf`` inputs; drives the IEEE-flow probes
    # Exact leading-null count under ``params``: an int for a windowed series, a per-field mapping for a struct, or
    # ``None`` for a reduction or an unwindowed transform (there is nothing to warm up).
    warmup: int | Mapping[str, int] | None = None
    fields: tuple[str, ...] = ()  # required non-empty for a struct, empty otherwise
    # Validation counterexamples: each is (kwargs overriding ``params``, the ValueError match regex).
    raises: tuple[tuple[Mapping[str, ScalarParam], str], ...] = ()
    golden: Golden | None = None  # the frozen golden master; the recommended hand-computed anchor
    # Crafted-input cases: the data home for exact values the synthesis and the oracle cannot derive.
    pins: tuple[Pin, ...] = ()
    # The recomposition identity: a zero-arg callable returning the ``pl.Expr`` that rebuilds this function out of other
    # public functions (a ratio metric as its numerator over its denominator, an oscillator as a difference of two
    # lines). When set, the recomposition rung holds the factory's output equal to the recomposition's on the probe
    # frame, lane by lane; ``None`` for a function with no such identity.
    recomposition: Callable[[], pl.Expr] | None = None
    # The documented answer to a degenerate regime (currently the all-null input); ``None`` means the ordinary answer.
    deviant: Deviant | None = None
    degenerate: enum.Enum | None = None  # the declared degenerate-denominator regime (family dialect); ``None`` if none
    # The annualization convention the output follows (family dialect); drives the closed-form annualization rung, which
    # is a no-op when this is ``None``.
    annualization: enum.Enum | None = None
    # The reducing / series twin a rolling function rolls per trailing window (the metrics twins); ``None`` for a
    # non-rolling function. When set, the twin-coherence rung holds this function's row ``i`` to the twin reduced over
    # the trailing window ending at ``i``, and the family harness inherits the twin's behavior where unstated.
    rolling_of: "Declaration | None" = None
    # The name of the window-length parameter (a key in ``params``): required with ``rolling_of`` — the coherence rung
    # slices by it and excludes it from the twin's parameters — and read by the in-window null shape. ``None`` when the
    # function is not windowed under a named parameter.
    window: str | None = None
    # An optional Hypothesis filter on a fuzzed frame, applied through ``assume`` in the property tier — the input
    # regimes where the impl and the oracle cannot be expected to agree, excluded as data rather than silently.
    conditioning: Callable[[pl.DataFrame], bool] | None = None
    # The oracle-agreement band, declared only where a one-pass rolling form cannot meet the tight default against its
    # two-pass oracle. ``None`` uses the tier default.
    oracle_rel_tol: float | None = None
    oracle_abs_tol: float | None = None
    reference: str = ""  # the literature citation for the definition (for the generated docstring)
    wikipedia: str = ""  # the encyclopedic reference URL (for the generated docstring)
    # The TA-Lib relation (the indicators dialect), read by the differential tier to partition the public surface:
    # a matching twin is compared bar for bar, a documented divergence and a no-equivalent are accounted for but not
    # compared. ``None`` for a family with no TA-Lib comparison (pnl, metrics).
    talib: enum.Enum | None = None
    # The reason a function documents a divergence from, or has no, TA-Lib twin: the pure data a non-``talib`` reader
    # still needs, carried on the declaration itself. Non-empty exactly for the divergence / no-equivalent relations,
    # empty for a matching twin (enforced in ``__post_init__``).
    talib_reason: str = ""
    # The recursion's seeding convention (the indicators dialect): metadata for the generated docstring, recorded but
    # not read by any rung. ``None`` where the family declares none.
    seeding: enum.Enum | None = None
    # A reason a function's interior-missing-bar flow is input-dependent and so cannot be held by a single behavior
    # shape: a directional-movement guard turns a fully-missing bar into neutral 0 movement (absorbed, the recurrence
    # continues at 0) while a single-column NaN on the driving leg still latches — two behaviors one declared shape
    # cannot hold. Non-empty skips the two structural flow rungs; the flow is pinned and covered by the missing-data
    # property tier instead. Empty = the flow rungs apply.
    flow_deviation: str = ""
    # Rows past an interior missing bar beyond which the declared flow must have played out; ``-1`` derives it from the
    # warm-up and the widest window. A positive value is declared only where an output is displaced further (a long
    # contracting recursion — the parabolic SAR cold start, the Ehlers Fisher pipeline).
    flow_horizon: int = -1

    def __post_init__(self) -> None:
        """Conditional requirements, checked loudly at construction (import time) — the one obvious place they live."""
        self._check_inputs()
        self._check_oracle_name()
        self._check_fields()
        self._check_warmup()
        self._check_scaling()
        self._check_golden()
        self._check_pins()
        self._check_deviant()
        self._check_rolling()
        self._check_talib()
        if self.params and not self.raises:
            msg = f"{self.name}: declares params but no raises counterexamples — the validation rung would be a no-op"
            raise ValueError(msg)

    def _check_inputs(self) -> None:
        unknown = sorted(role for role in self.inputs if role not in KNOWN_ROLES)
        if not self.inputs or unknown:
            msg = f"{self.name}: inputs must be non-empty roles the probe frame can build; unknown: {unknown}"
            raise ValueError(msg)

    def _check_oracle_name(self) -> None:
        expected = f"reference_{self.name}"
        if self.oracle.__name__ != expected:
            msg = f"{self.name}: the oracle must be named {expected!r}, got {self.oracle.__name__!r}"
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

    def _check_scaling(self) -> None:
        if isinstance(self.scaling, ScaleExempt):
            return
        if not self.scaling:
            msg = f"{self.name}: an empty scale tuple is never allowed — declare ScaleExempt(reason) instead"
            raise ValueError(msg)
        for axis in self.scaling:
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
        if self.golden is None:
            return
        if set(self.golden.inputs) != set(self.inputs):
            msg = f"{self.name}: golden inputs {sorted(self.golden.inputs)} must match inputs {list(self.inputs)}"
            raise ValueError(msg)
        struct_golden = isinstance(self.golden.output, Mapping)
        if struct_golden != (self.shape is Shape.STRUCT):
            msg = f"{self.name}: a golden output is a per-field mapping iff the shape is a struct"
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

    def _check_deviant(self) -> None:
        if self.deviant is not None and not self.deviant.reason.strip():
            msg = f"{self.name}: a deviant must carry a non-empty reason"
            raise ValueError(msg)

    def _check_rolling(self) -> None:
        if self.window is not None and self.window not in self.params:
            msg = f"{self.name}: window={self.window!r} must name a parameter in params {sorted(self.params)}"
            raise ValueError(msg)
        if self.rolling_of is None:
            return
        if self.window is None:
            msg = f"{self.name}: a rolling twin (rolling_of={self.rolling_of.name}) must name its window parameter"
            raise ValueError(msg)
        if self.rolling_of.inputs != self.inputs:
            msg = (
                f"{self.name}: a rolling twin shares its twin's inputs, so the window slice can drive both — "
                f"{list(self.rolling_of.inputs)} != {list(self.inputs)}"
            )
            raise ValueError(msg)

    def _check_talib(self) -> None:
        if self.talib is None:
            if self.talib_reason:
                msg = f"{self.name}: a talib_reason needs a talib relation to justify"
                raise ValueError(msg)
            return
        documented = self.talib.name in ("DOCUMENTED_DIVERGENCE", "NO_EQUIVALENT")
        if documented and not self.talib_reason.strip():
            msg = f"{self.name}: talib={self.talib.name} must carry a non-empty talib_reason"
            raise ValueError(msg)
        if not documented and self.talib_reason:
            msg = f"{self.name}: talib={self.talib.name} is a matching twin and takes no talib_reason"
            raise ValueError(msg)

    # --- derived, never declared: read off the factory ---

    @property
    def name(self) -> str:
        """The function's name, read off the factory — the key the id, the registry, and the oracle name derive from."""
        return self.factory.__name__

    @property
    def landing(self) -> str:
        """The column the output lands on: the first input's root name."""
        return self.inputs[0]


# --- the engine the rungs and the synthesis builders delegate to ---


def build_expr(declaration: Declaration, **overrides: ScalarParam) -> pl.Expr:
    """The factory applied to its declared input columns under ``params`` (with optional per-call overrides)."""
    columns = (pl.col(role) for role in declaration.inputs)
    return declaration.factory(*columns, **{**declaration.params, **overrides})


def lane_series(out: pl.DataFrame) -> list[pl.Series]:
    """Every scalar lane of a computed ``out`` column — a struct's fields expanded, so no lane is ever skipped."""
    schema = out.schema["out"]
    if isinstance(schema, pl.Struct):
        return [out["out"].struct.field(inner.name) for inner in schema.fields]
    return [out["out"]]


def flat(declaration: Declaration, frame: pl.DataFrame) -> list[pl.Series]:
    """The output lanes of the declaration's expression applied to ``frame``."""
    return lane_series(frame.select(build_expr(declaration).alias("out")))


def actual_lanes(declaration: Declaration, frame: pl.DataFrame) -> dict[str, list[float | None]]:
    """The expression's output as one named lane per line: struct field names for a struct, else ``{"out": ...}``."""
    lanes = flat(declaration, frame)
    if len(lanes) > 1:
        return {lane.name: lane.to_list() for lane in lanes}
    return {"out": lanes[0].to_list()}


def reference_lanes(declaration: Declaration, frame: pl.DataFrame) -> dict[str, list[float | None]]:
    """The oracle's output as one named lane per line, matching :func:`actual_lanes`'s naming."""
    lists = [frame[role].to_list() for role in declaration.inputs]
    result = declaration.oracle(*lists, **declaration.params)
    if isinstance(result, Mapping):
        mapping = cast("Mapping[str, Sequence[float | None]]", result)
        return {str(name): list(values) for name, values in mapping.items()}
    if isinstance(result, list):
        return {"out": cast("list[float | None]", result)}
    return {"out": [cast("float | None", result)]}


def widest_warmup(declaration: Declaration) -> int:
    """The widest declared leading-null count (0 for a reduction or an unwindowed transform), used to size frames."""
    if declaration.warmup is None:
        return 0
    if isinstance(declaration.warmup, Mapping):
        return max(declaration.warmup.values())
    return declaration.warmup


def widest_window(declaration: Declaration) -> int:
    """The widest ``window*`` integer parameter (1 when the function has none) — a term of the flow horizon."""
    windows = [
        value for key, value in declaration.params.items() if key.startswith("window") and isinstance(value, int)
    ]
    return max(windows) if windows else 1


def window_length(declaration: Declaration) -> int:
    """The declared window length: the value of the named ``window`` param, else the widest ``window*`` param."""
    if declaration.window is not None:
        return int(declaration.params[declaration.window])
    return widest_window(declaration)


def horizon(declaration: Declaration) -> int:
    """Rows past an interior missing bar beyond which the declared flow must have played out."""
    if declaration.flow_horizon >= 0:
        return declaration.flow_horizon
    return widest_warmup(declaration) + widest_window(declaration) + 2


def probe_length(declaration: Declaration) -> int:
    """A probe length long enough that a flow rung's post-horizon tail is never empty."""
    return widest_warmup(declaration) + 3 + horizon(declaration) + 8
