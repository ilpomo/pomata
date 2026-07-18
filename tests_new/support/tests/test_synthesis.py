"""
Self-tests of :mod:`tests_new.support.synthesis` — the regime probe builders and the fuzz strategy.

These pin that the synthesized probes are deterministic and tiny, that a poisoned bar lands where the description says,
that an infinite-input probe is built per role, and that the fuzz vocabulary refuses an unsupported input shape rather
than silently under-testing it.
"""

import enum
import math

import polars as pl
import pytest
from hypothesis import strategies as st

from tests_new.support.declaration import Declaration, ScaleAxis, Shape
from tests_new.support.synthesis import (
    frame_all_null,
    frame_empty,
    frame_infinite_input,
    frame_nan_interior,
    frame_null_interior,
    frame_single_row,
    fuzz_frames,
)


class _Behavior(enum.Enum):
    PROPAGATES = "propagates"


class _Space(enum.Enum):
    CASH = "cash"


class _Sign(enum.Enum):
    LONG_SHORT = "long_short"


class _NonFinite(enum.Enum):
    IEEE_FLOW = "ieee_flow"


def one_leg(price: pl.Expr) -> pl.Expr:
    """A single-input stand-in factory."""
    return price


def reference_one_leg(price: list[float | None]) -> list[float | None]:
    """The stand-in oracle for ``one_leg``."""
    return list(price)


def two_legs(quantity: pl.Expr, price: pl.Expr) -> pl.Expr:
    """A two-input stand-in factory."""
    return quantity * price


def reference_two_legs(quantity: list[float | None], price: list[float | None]) -> list[float | None]:
    """The stand-in oracle for ``two_legs``."""
    return [None if q is None or p is None else q * p for q, p in zip(quantity, price, strict=True)]


_ONE = Declaration(
    family="pnl",
    factory=one_leg,
    inputs=("price",),
    params={},
    shape=Shape.SERIES,
    behavior_null=_Behavior.PROPAGATES,
    behavior_nan=_Behavior.PROPAGATES,
    space=_Space.CASH,
    sign=_Sign.LONG_SHORT,
    nonfinite=_NonFinite.IEEE_FLOW,
    oracle=reference_one_leg,
    scaling=(ScaleAxis(roles=("price",), degree=0),),
)

_TWO = Declaration(
    family="pnl",
    factory=two_legs,
    inputs=("quantity", "price"),
    params={},
    shape=Shape.SERIES,
    behavior_null=_Behavior.PROPAGATES,
    behavior_nan=_Behavior.PROPAGATES,
    space=_Space.CASH,
    sign=_Sign.LONG_SHORT,
    nonfinite=_NonFinite.IEEE_FLOW,
    oracle=reference_two_legs,
    scaling=(ScaleAxis(roles=("quantity",), degree=1),),
)

_UNSUPPORTED = Declaration(
    family="pnl",
    factory=two_legs,
    inputs=("high", "low"),
    params={},
    shape=Shape.SERIES,
    behavior_null=_Behavior.PROPAGATES,
    behavior_nan=_Behavior.PROPAGATES,
    space=_Space.CASH,
    sign=_Sign.LONG_SHORT,
    nonfinite=_NonFinite.IEEE_FLOW,
    oracle=reference_two_legs,
    scaling=(ScaleAxis(roles=("high",), degree=1),),
)


class TestRegimeProbes:
    """The degenerate-regime builders are deterministic, tiny, and inject where they say."""

    def test_null_interior_is_deterministic_and_placed(self) -> None:
        """The null-interior probe repeats byte for byte and nulls the declared injection row."""
        first = frame_null_interior(_ONE)
        second = frame_null_interior(_ONE)
        assert first.frame.equals(second.frame)
        assert first.frame.height <= 8
        column = first.frame["price"].to_list()
        assert column[2] is None  # widest_warmup (0) + 2

    def test_nan_interior_places_a_nan(self) -> None:
        """The NaN-interior probe puts a ``NaN`` at the injection row."""
        column = frame_nan_interior(_ONE).frame["price"].to_list()
        value = column[2]
        assert value is not None
        assert math.isnan(value)

    def test_infinite_input_is_one_probe_per_role(self) -> None:
        """The infinite-input builder yields one probe per input role, each carrying ``+inf`` and ``-inf``."""
        probes = frame_infinite_input(_TWO)
        assert len(probes) == 2
        quantity_probe = probes[0]
        column = quantity_probe.frame["quantity"].to_list()
        assert math.inf in column
        assert -math.inf in column

    def test_all_null_is_entirely_null(self) -> None:
        """The all-null probe has every input column entirely ``null``."""
        frame = frame_all_null(_TWO).frame
        assert all(value is None for value in frame["quantity"].to_list())
        assert all(value is None for value in frame["price"].to_list())

    def test_single_row_and_empty(self) -> None:
        """The single-row probe has one row and the empty probe has none."""
        assert frame_single_row(_ONE).frame.height == 1
        assert frame_empty(_ONE).frame.height == 0

    def test_snippet_round_trips(self) -> None:
        """The reproduction snippet imports the factory and frames the same data it was built from."""
        probe = frame_null_interior(_ONE)
        assert probe.snippet[0] == "import polars as pl"
        assert probe.snippet[1] == "from pomata.pnl import one_leg"
        assert 'one_leg(pl.col("price"))' in probe.snippet[2]
        assert "None" in probe.snippet[2]


class TestFuzzFrames:
    """The fuzz vocabulary backs the supported shapes and refuses the rest."""

    def test_supported_shapes_return_a_strategy(self) -> None:
        """A single-input and a supported multi-input shape both return a strategy."""
        assert isinstance(fuzz_frames(_ONE, missing=False), st.SearchStrategy)
        assert isinstance(fuzz_frames(_TWO, missing=True), st.SearchStrategy)

    def test_unsupported_shape_raises(self) -> None:
        """An input shape outside the closed vocabulary raises rather than under-testing silently."""
        with pytest.raises(TypeError, match=r"no fuzz strategy for inputs"):
            fuzz_frames(_UNSUPPORTED, missing=False)
