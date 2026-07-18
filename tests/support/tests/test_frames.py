"""
Self-tests of :mod:`tests.support.frames` — the materialize adapters, the probe-frame builders, and the splitters.

These pin the test infrastructure: the adapters must round-trip the ``null`` / ``NaN`` / finite distinction through
Polars unchanged, and the probe frame must give one distinctly-named ``Float64`` column per role so a multi-input
factory is caught by construction rather than passing on identical inputs.
"""

import math

import polars as pl

from tests.support.frames import (
    COLUMN_X,
    KNOWN_ROLES,
    apply_expr,
    count_leading_nulls,
    materialize,
    probe_frame,
    split_pairs,
)


class TestMaterialize:
    """The adapters build a ``Float64`` frame, evaluate the expression, and return the output (null / NaN kept)."""

    def test_apply_expr_preserves_length_null_and_nan(self) -> None:
        """The single-input adapter keeps length and round-trips ``None`` and ``NaN`` distinctly."""
        out = apply_expr([1.0, None, math.nan, 4.0], pl.col(COLUMN_X) * 2.0)
        assert len(out) == 4
        assert out[0] == 2.0
        assert out[1] is None
        assert out[2] is not None
        assert math.isnan(out[2])
        assert out[3] == 8.0

    def test_materialize_reads_named_columns(self) -> None:
        """The multi-input adapter wires each named column to its ``pl.col(name)``."""
        out = materialize({"high": [3.0, 4.0], "low": [1.0, 2.0]}, pl.col("high") - pl.col("low"))
        assert out == [2.0, 2.0]

    def test_materialize_preserves_null_and_nan(self) -> None:
        """A ``None`` stays ``None`` and a ``NaN`` stays ``NaN`` through the frame coercion."""
        out = materialize({"high": [None, math.nan, 3.0]}, pl.col("high"))
        assert out[0] is None
        assert out[1] is not None
        assert math.isnan(out[1])
        assert out[2] == 3.0


class TestProbeFrame:
    """The probe frame gives one distinctly-named ``Float64`` column per declared role."""

    def test_distinct_named_columns(self) -> None:
        """Each role builds its own named column, and the columns are not identical."""
        frame = probe_frame(("high", "low"), 5)
        assert frame.columns == ["high", "low"]
        assert frame.height == 5
        assert frame["high"].to_list() != frame["low"].to_list()

    def test_zero_length_is_empty(self) -> None:
        """A zero-length probe frame has the columns but no rows."""
        frame = probe_frame(("price",), 0)
        assert frame.columns == ["price"]
        assert frame.height == 0

    def test_every_role_is_known(self) -> None:
        """Every role the builders know is exposed in the known-role set."""
        assert "quantity" in KNOWN_ROLES
        assert "mystery" not in KNOWN_ROLES


class TestHelpers:
    """The leading-null count and the bar splitter."""

    def test_count_leading_nulls(self) -> None:
        """The leading ``None`` run is counted up to the first non-null value."""
        assert count_leading_nulls([None, None, 1.0, None]) == 2
        assert count_leading_nulls([1.0, None]) == 0

    def test_split_pairs(self) -> None:
        """A list of 2-tuples unzips into two aligned columns."""
        assert split_pairs([(1.0, 2.0), (3.0, 4.0)]) == ([1.0, 3.0], [2.0, 4.0])
