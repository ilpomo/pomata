"""
Regime synthesis: build the degenerate-input probes a rung needs straight from the declaration, plus the property-tier
fuzz strategy.

The old suite *recognized* a regime from a hand-written fixed case whose name contained a marker. This inverts it: the
declaration says a function carries ``±inf`` per IEEE flow or propagates an interior ``null``, and these builders
*construct* the input that provokes the regime — an infinite lane per input, a poisoned interior bar, an all-null frame,
a single row, an empty frame. Each returns a :class:`Probe` carrying the tiny frame, a one-line description, and a
copy-pasteable reproduction snippet, so a failing rung can print exactly what it fed the function. The frames are
deterministic and small (a handful of rows), with the distinctly-named per-role columns the probe-frame builders give.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl
from hypothesis import strategies as st

from tests_new.support.declaration import Declaration, probe_length, widest_warmup
from tests_new.support.frames import probe_frame
from tests_new.support.strategies import finite_floats, missing_data_floats

# A tiny extra tail past the warm-up: an interior injection at ``widest_warmup + 2`` still leaves rows to observe the
# flow play out, while the whole frame stays within the handful-of-rows contract the pnl dialect keeps.
_PROBE_TAIL = 5
_ALL_NULL_ROWS = 4


@dataclass(frozen=True)
class Probe:
    """One synthesized input case: the tiny frame, a one-line description of the regime, and a repro snippet."""

    frame: pl.DataFrame
    description: str
    snippet: tuple[str, ...]


def _probe_length(declaration: Declaration) -> int:
    """A tiny probe length: the warm-up plus a short tail, so an interior injection still has rows to play out."""
    return widest_warmup(declaration) + _PROBE_TAIL


def _injection(declaration: Declaration) -> int:
    """The interior bar a regime is injected at — past the warm-up, with a tail behind it."""
    return widest_warmup(declaration) + 2


def _lane_literal(values: Sequence[float | None]) -> str:
    """Render a lane as a Python list literal, spelling out ``None`` / ``nan`` / ``±inf`` so the snippet round-trips."""
    parts: list[str] = []
    for value in values:
        if value is None:
            parts.append("None")
        elif math.isnan(value):
            parts.append('float("nan")')
        elif math.isinf(value):
            parts.append('float("inf")' if value > 0.0 else 'float("-inf")')
        else:
            parts.append(repr(value))
    return "[" + ", ".join(parts) + "]"


def _call_repr(declaration: Declaration) -> str:
    """The factory call as source text: ``name(pl.col("a"), pl.col("b"), param=value)``."""
    positional = ", ".join(f'pl.col("{role}")' for role in declaration.inputs)
    keywords = "".join(f", {name}={value!r}" for name, value in declaration.params.items())
    return f"{declaration.name}({positional}{keywords})"


def _snippet(declaration: Declaration, frame: pl.DataFrame) -> tuple[str, ...]:
    """A three-line copy-pasteable reproduction: the import, the factory import, and the framed call."""
    columns = ", ".join(f'"{role}": {_lane_literal(frame[role].to_list())}' for role in declaration.inputs)
    return (
        "import polars as pl",
        f"from pomata.{declaration.family} import {declaration.name}",
        f"pl.DataFrame({{{columns}}}).select({_call_repr(declaration)})",
    )


def describe(declaration: Declaration, frame: pl.DataFrame, description: str) -> Probe:
    """Wrap an already-built frame (deterministic, golden, pinned, or rescaled) into a :class:`Probe` for a message."""
    return Probe(frame, description, _snippet(declaration, frame))


def _poison(declaration: Declaration, length: int, injection: int, *, value: float | None) -> pl.DataFrame:
    """A clean probe frame with every input column set to ``value`` (``None`` or ``NaN``) at one interior bar."""
    clean = probe_frame(declaration.inputs, length)
    return clean.with_columns(
        pl.when(pl.int_range(pl.len()) == injection)
        .then(pl.lit(value, dtype=pl.Float64))
        .otherwise(pl.col(role))
        .alias(role)
        for role in declaration.inputs
    )


def frame_null_interior(declaration: Declaration) -> Probe:
    """A probe with a ``null`` punched into every input at one interior bar — the interior-null flow regime."""
    injection = _injection(declaration)
    frame = _poison(declaration, _probe_length(declaration), injection, value=None)
    return Probe(frame, f"a null in every input at row {injection}", _snippet(declaration, frame))


def frame_nan_interior(declaration: Declaration) -> Probe:
    """A probe with a ``NaN`` punched into every input at one interior bar — the interior-NaN flow regime."""
    injection = _injection(declaration)
    frame = _poison(declaration, _probe_length(declaration), injection, value=math.nan)
    return Probe(frame, f"a NaN in every input at row {injection}", _snippet(declaration, frame))


@dataclass(frozen=True)
class FlowProbe:
    """An interior-injection probe paired with its clean twin and the injected row, for the structural flow checks."""

    probe: Probe
    frame_clean: pl.DataFrame
    row: int


def _flow_pair(declaration: Declaration, *, value: float | None, regime: str) -> FlowProbe:
    """The clean frame and its poisoned twin, injected at one interior bar past the warm-up.

    The length comes from :func:`probe_length`, which is sized so the post-horizon tail is never empty — the
    structural recovery claims would otherwise pass vacuously.
    """
    length = probe_length(declaration)
    injection = _injection(declaration)
    clean = probe_frame(declaration.inputs, length)
    frame = _poison(declaration, length, injection, value=value)
    probe = Probe(frame, f"a {regime} in every input at row {injection}", _snippet(declaration, frame))
    return FlowProbe(probe, clean, injection)


def frame_flow_null(declaration: Declaration) -> FlowProbe:
    """The interior-``null`` flow pair: the poisoned probe, its clean twin, and the injected row."""
    return _flow_pair(declaration, value=None, regime="null")


def frame_flow_nan(declaration: Declaration) -> FlowProbe:
    """The interior-``NaN`` flow pair: the poisoned probe, its clean twin, and the injected row."""
    return _flow_pair(declaration, value=math.nan, regime="NaN")


def frame_infinite_input(declaration: Declaration) -> list[Probe]:
    """One probe per input role: ``+inf`` then ``-inf`` at two interior bars of that role — the IEEE-flow regime."""
    length = _probe_length(declaration)
    positive = _injection(declaration)
    negative = positive + 1
    probes: list[Probe] = []
    for role in declaration.inputs:
        clean = probe_frame(declaration.inputs, length)
        frame = clean.with_columns(
            pl.when(pl.int_range(pl.len()) == positive)
            .then(pl.lit(math.inf, dtype=pl.Float64))
            .when(pl.int_range(pl.len()) == negative)
            .then(pl.lit(-math.inf, dtype=pl.Float64))
            .otherwise(pl.col(role))
            .alias(role)
        )
        probes.append(
            Probe(frame, f"+inf then -inf in {role!r} at rows {positive}, {negative}", _snippet(declaration, frame))
        )
    return probes


def frame_all_null(declaration: Declaration) -> Probe:
    """A probe whose every input column is entirely ``null`` — the all-missing regime."""
    frame = pl.DataFrame({role: pl.Series([None] * _ALL_NULL_ROWS, dtype=pl.Float64) for role in declaration.inputs})
    return Probe(frame, "every input all-null", _snippet(declaration, frame))


def frame_single_row(declaration: Declaration) -> Probe:
    """A one-row probe — the shortest well-conditioned input."""
    frame = probe_frame(declaration.inputs, 1)
    return Probe(frame, "a single row", _snippet(declaration, frame))


def frame_empty(declaration: Declaration) -> Probe:
    """A zero-row probe — the empty-frame regime."""
    frame = probe_frame(declaration.inputs, 0)
    return Probe(frame, "an empty frame", _snippet(declaration, frame))


# --- the property-tier fuzz strategy ---


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
    # The two legs of a relative metric: modest returns bounded away from zero so an embedded regression's variance
    # stays well-conditioned and a capture ratio's geometric power never lands in the near-one cancellation band.
    "returns": st.one_of(_finite(0.01, 0.5), _finite(-0.5, -0.01)),
    "benchmark": st.one_of(_finite(0.01, 0.5), _finite(-0.5, -0.01)),
}

# The multi-input shapes the vocabulary supports, read off the pnl and benchmark-relative factory signatures; every
# role appears in the probe-frame builders too, so each shape can back a real declaration. Anything outside this closed
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


def _single_input_frame(role: str, length: st.SearchStrategy[int], *, missing: bool) -> st.SearchStrategy[pl.DataFrame]:
    """A one-column frame drawn from the role's domain: modest returns, positive prices, or the general finite band."""
    if role == "returns":
        # A modest return domain bounded away from zero: a running sum stays well-conditioned against its two-pass
        # oracle (a subnormal-magnitude or near-zero draw would round the two apart).
        finite: st.SearchStrategy[float] = st.one_of(_finite(0.01, 1.0), _finite(-1.0, -0.01))
        values: st.SearchStrategy[float | None] = st.one_of(st.none(), st.just(math.nan), finite) if missing else finite
    elif role == "price":
        # Strictly positive prices in a modest band: a one-bar log or simple return stays well-defined and
        # well-conditioned here, where a symmetric or near-zero draw would round the two paths apart.
        positive = _finite(1.0, 1e3)
        values = st.one_of(st.none(), st.just(math.nan), positive) if missing else positive
    else:
        values = missing_data_floats() if missing else finite_floats()
    return length.flatmap(
        lambda n: st.lists(values, min_size=n, max_size=n).map(
            lambda rows: pl.DataFrame({role: pl.Series(rows, dtype=pl.Float64)})
        )
    )


def fuzz_frames(declaration: Declaration, *, missing: bool) -> st.SearchStrategy[pl.DataFrame]:
    """
    A Hypothesis strategy of well-formed input frames for the property tier, keyed on the declaration's input shape.

    Covers the single-input shapes and the multi-input pnl / benchmark-relative shapes; an unlisted shape raises, so the
    closed vocabulary can never silently under-test a new function.
    """
    minimum = widest_warmup(declaration) + 4
    length = st.integers(min_value=minimum, max_value=minimum + 24)
    if len(declaration.inputs) == 1:
        return _single_input_frame(declaration.inputs[0], length, missing=missing)
    if declaration.inputs in _FUZZ_SHAPES:
        return _independent_frame(declaration.inputs, length, missing=missing)
    msg = f"{declaration.name}: no fuzz strategy for inputs {declaration.inputs}"  # extend when a new shape lands
    raise TypeError(msg)
