"""
Meta-tests for ``tests_new.support.frames`` — the materialize adapters.

These pin the test infrastructure: the materialize adapters must round-trip the ``null`` / ``NaN`` / finite distinction
through Polars unchanged, so an indicator test compares like with like.
"""

import math

import polars as pl
from tests_new.support import CLOSE, COLUMN_X, HIGH, LOW, apply_expr, materialize, materialize_struct


class TestMaterialize:
    """
    The adapters build a ``Float64`` frame, evaluate the expression, and return the output (preserving null / NaN).
    """

    def test_apply_expr_preserves_length_null_and_nan(self) -> None:
        """
        Verifies that the single-input adapter keeps length and round-trips ``None`` and ``NaN`` distinctly.
        """
        out = apply_expr([1.0, None, math.nan, 4.0], pl.col(COLUMN_X) * 2.0)
        assert len(out) == 4
        assert out[0] == 2.0
        assert out[1] is None
        assert out[2] is not None
        assert math.isnan(out[2])
        assert out[3] == 8.0

    def test_materialize_reads_named_columns(self) -> None:
        """
        Verifies that the multi-input adapter wires each named column to its ``pl.col(name)``.
        """
        out = materialize({HIGH: [3.0, 4.0], LOW: [1.0, 2.0]}, pl.col(HIGH) - pl.col(LOW))
        assert out == [2.0, 2.0]

    def test_materialize_matches_apply_expr_on_one_column(self) -> None:
        """
        Verifies that ``materialize`` and ``apply_expr`` agree on the single-column case (``apply_expr`` delegates).
        """
        expr = pl.col(COLUMN_X) + 1.0
        assert materialize({COLUMN_X: [1.0, 2.0, 3.0]}, expr) == apply_expr([1.0, 2.0, 3.0], expr)

    def test_materialize_preserves_null_and_nan(self) -> None:
        """
        Verifies that a ``None`` stays ``None`` and a ``NaN`` stays ``NaN`` through the frame coercion.
        """
        out = materialize({HIGH: [None, math.nan, 3.0]}, pl.col(HIGH))
        assert out[0] is None
        assert out[1] is not None
        assert math.isnan(out[1])
        assert out[2] == 3.0

    def test_materialize_struct_splits_each_field(self) -> None:
        """
        Verifies that the struct adapter returns one aligned list per requested field, in order.
        """
        expr = pl.struct(total=pl.col(HIGH) + pl.col(CLOSE), diff=pl.col(HIGH) - pl.col(CLOSE))
        out = materialize_struct({HIGH: [3.0, 5.0], CLOSE: [1.0, 2.0]}, expr, ("total", "diff"))
        assert out == {"total": [4.0, 7.0], "diff": [2.0, 3.0]}
