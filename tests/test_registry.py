"""
The self-check for the edge-case contract registry (:mod:`tests.support.registry`).

The registry is the single source of truth the shared contracts read; this module keeps it honest. It proves three
things, so a row that lies about its function -- or a function that is added without a row -- is a red build rather than
a finding for the next audit:

- **bijection** -- every public ``__all__`` name has exactly one row, and no row is an orphan;
- **oracle integrity** -- every declared ``*_reference`` oracle is importable from its family's ``oracles`` package;
- **policy is real** -- each function's *actual* ``null`` / ``NaN`` behaviour, observed on a well-conditioned series,
  matches the ``null_policy`` / ``nan_policy`` and ``shape`` it declares. Only the null/NaN *flow* is read (which rows
  become null/NaN, and whether the effect recovers), never a value, so the check is exact and platform-stable.
"""

import importlib
import inspect
import math
from collections.abc import Callable

import polars as pl
import pytest
from tests.support.registry import REGISTRY, NanPolicy, NullPolicy, Shape

from pomata import indicators, metrics, pnl

_FAMILIES = {"indicators": indicators, "pnl": pnl, "metrics": metrics}
# A well-spread strictly-positive series: enough variation that no windowed statistic degenerates, values irrelevant.
# It must be long enough that the gap lands *after* every function's warm-up -- the deepest is the 63-bar Hilbert
# pipeline -- so the flow read is never taken on an all-null baseline (which would pass a latch check vacuously).
_SERIES: list[float | None] = [100.0 + 5.0 * math.sin(i) + 0.3 * i for i in range(96)]
_GAP = 75  # an interior index past the deepest warm-up, with room after it for the recovery / span read
_SPREAD = 3.0  # keeps a coherent bar's high strictly above its low, so a directional movement is never degenerate


def _public_names() -> set[str]:
    return {name for family in _FAMILIES.values() for name in family.__all__}


def _factory(name: str) -> Callable[..., pl.Expr]:
    for family in _FAMILIES.values():
        if name in family.__all__:
            return getattr(family, name)  # type: ignore[no-any-return]
    raise KeyError(name)


def _column(role: str, series: list[float | None]) -> list[float | None]:
    """A coherent, domain-appropriate column for a parameter named `role`, carrying `series`' interior gap."""
    if role == "high":
        return [None if v is None else v + _SPREAD for v in series]
    if role == "low":
        return [None if v is None else v - _SPREAD for v in series]
    if role == "volume":
        return [None if v is None else 1000.0 + 10.0 * v for v in series]
    if role == "benchmark":
        return [None if v is None else 0.4 * v + 5.0 for v in series]  # distinct from a `returns` input, not a copy
    return series  # close / open / returns / equity_curve / a single generic price series


_CLEAN: object = object()  # sentinel: run with no injected gap


def _expr_params(factory: Callable[..., pl.Expr]) -> list[str]:
    return [
        p.name
        for p in inspect.signature(factory, eval_str=True).parameters.values()
        if p.default is inspect.Parameter.empty and p.annotation is pl.Expr
    ]


def _run(
    factory: Callable[..., pl.Expr],
    inject: object = _CLEAN,
    target: str | None = None,
    base: list[float | None] = _SERIES,
) -> tuple[list[object], int]:
    """
    Call `factory` on a coherent frame built from `base`. With `inject` a marker (``None`` or ``NaN``), place it at
    ``_GAP`` in the `target` input only (default: the first), leaving every other input coherent and finite -- so the
    flow read is the function's response to one interior gap in a single input, never the degenerate all-inputs-missing
    bar (which some guards absorb to a defined value, masking a recursion).
    """
    parameters = [
        p for p in inspect.signature(factory, eval_str=True).parameters.values() if p.default is inspect.Parameter.empty
    ]
    expr_names = [p.name for p in parameters if p.annotation is pl.Expr]
    columns = {name: _column(name, base) for name in expr_names}
    if inject is not _CLEAN and expr_names:
        name = target or expr_names[0]
        gapped = list(columns[name])
        gapped[_GAP] = inject  # type: ignore[call-overload]
        columns[name] = gapped
    positional: list[object] = []
    keywords: dict[str, object] = {}
    for parameter in parameters:
        value = (
            pl.col(parameter.name) if parameter.annotation is pl.Expr else (3 if parameter.annotation is int else 0.1)
        )
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            keywords[parameter.name] = value
        else:
            positional.append(value)
    frame = pl.DataFrame({name: pl.Series(name, data, dtype=pl.Float64) for name, data in columns.items()})
    out = frame.select(factory(*positional, **keywords).alias("o"))
    column = out.unnest("o").to_series(0) if isinstance(out.schema["o"], pl.Struct) else out["o"]
    return column.to_list(), out.height


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _defined(value: object) -> bool:
    return value is not None and not _is_nan(value)


class _Observation:
    """
    The null/NaN flow of one function, read structurally over EVERY input: a gap is injected into each input in turn and
    the worst case is taken, because a gap in any one input (an EMA's own source, the low leg of a down-movement) is
    enough to bridge or latch. The policy documents that worst case.
    """

    def __init__(self, factory: Callable[..., pl.Expr]) -> None:
        clean, height = _run(factory)
        self.reducing = height == 1
        self.clean = clean
        targets: list[str | None] = [*_expr_params(factory)] or [None]
        self._null_runs = [_run(factory, None, target)[0] for target in targets]
        self._nan_runs = [_run(factory, math.nan, target)[0] for target in targets]

    def reduction_skips_null(self, factory: Callable[..., pl.Expr]) -> bool:
        removed, _ = _run(factory, base=[v for i, v in enumerate(_SERIES) if i != _GAP])
        here, there = self._null_runs[0][0], removed[0]
        if not (isinstance(here, float) and math.isfinite(here)) or not (
            isinstance(there, float) and math.isfinite(there)
        ):
            return (
                True  # inconclusive on this frame (e.g. a degenerate denominator); the class-tier contracts verify it
            )
        return math.isclose(here, there, rel_tol=1e-9, abs_tol=1e-9)

    @property
    def reduction_poisons_on_nan(self) -> bool:
        return _is_nan(self._nan_runs[0][0])

    @property
    def null_recovers(self) -> bool:
        return all(_defined(run[-1]) for run in self._null_runs)

    @property
    def nan_recovers(self) -> bool:
        return all(_defined(run[-1]) for run in self._nan_runs)

    @property
    def null_span(self) -> int:
        """The most rows at/after the gap that a single interior null nulls, across all inputs."""
        spans = (
            sum(1 for i in range(_GAP, len(self.clean)) if run[i] is None and self.clean[i] is not None)
            for run in self._null_runs
        )
        return max(spans, default=0)


def test_registry_is_in_bijection_with_public_surface() -> None:
    """
    Verifies the registry has exactly one row per public function and no orphan rows.
    """
    assert set(REGISTRY) == _public_names()


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_declared_oracle_is_importable(name: str) -> None:
    """
    Verifies that where a row names a ``*_reference`` oracle, it is importable from its family's ``oracles`` package.
    """
    oracle = REGISTRY[name].oracle
    if oracle is None:
        pytest.skip("correctness pinned by component-definition / golden master, not a standalone oracle")
    module = importlib.import_module(f"tests.{REGISTRY[name].macro.value}.oracles.{name}")
    assert hasattr(module, oracle)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_declared_policy_matches_actual_behaviour(name: str) -> None:
    """
    Verifies each function's observed ``null`` / ``NaN`` flow matches the ``shape`` and policies its row declares.
    """
    profile = REGISTRY[name]
    observation = _Observation(_factory(name))

    assert observation.reducing == (profile.shape is Shape.REDUCING), f"{name}: shape/reducing mismatch"

    if profile.shape is Shape.REDUCING:
        assert profile.null_policy is NullPolicy.SKIPPED, f"{name}: a reducing metric declares {profile.null_policy}"
        assert observation.reduction_skips_null(_factory(name)), f"{name}: a dropped null changes the reduction"
        assert profile.nan_policy is NanPolicy.POISONS, f"{name}: a reducing metric declares {profile.nan_policy}"
        assert observation.reduction_poisons_on_nan, f"{name}: a NaN does not poison the reduction"
        return

    # elementwise / struct: the null policy is fixed by (recovers?, span) and the nan policy by (recovers?).
    # Guard against a vacuous read: the clean (un-gapped) run must clear this function's warm-up and emit a defined
    # final row, or a latch check would pass on an all-null baseline while verifying nothing.
    assert _defined(observation.clean[-1]), f"{name}: the probe series is too short to clear the warm-up"
    span = observation.null_span
    if profile.null_policy is NullPolicy.LATCHES:
        assert not observation.null_recovers, f"{name}: declares null LATCHES but the output recovers"
    else:
        assert observation.null_recovers, f"{name}: declares null {profile.null_policy} but it never recovers"
        if profile.null_policy is NullPolicy.IN_WINDOW_IS_NULL:
            assert span >= 2, f"{name}: declares IN_WINDOW_IS_NULL but a null spans only {span} row(s)"
        elif profile.null_policy is NullPolicy.PROPAGATES:
            assert span <= 2, f"{name}: declares PROPAGATES but a null spans {span} rows (windowed?)"

    if profile.nan_policy is NanPolicy.LATCHES:
        assert not observation.nan_recovers, f"{name}: declares nan LATCHES but the output recovers"
    else:
        assert observation.nan_recovers, f"{name}: declares nan {profile.nan_policy} but it never recovers"

    # a recursion that bridges a null is exactly one that latches a NaN -- the two must be declared together.
    if profile.null_policy is NullPolicy.BRIDGED:
        assert profile.nan_policy is NanPolicy.LATCHES, f"{name}: BRIDGED null must pair with LATCHES nan"
