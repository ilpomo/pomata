"""
Tests for ``pomata.metrics.total_return_rolling`` — the rolling (windowed) twin of :func:`pomata.metrics.total_return`.

``total_return_rolling`` is single-input and WINDOWED-SERIES-VALUED (an equity series → a series the same length, one
value per trailing window), so tests use the shared ``apply_expr`` helper; ``assert_matches`` and the naive
``total_return_rolling_reference`` oracle (the window's last equity over its first, less one) are shared across the
suite. It is an ENDPOINT quantity: a window's value depends only on its first and last equity, so a ``null`` / ``NaN``
at either endpoint propagates while an interior ``null`` / ``NaN`` is spanned and does not affect the result.

The ladder is the canonical one: contract (type / length-preserving / lazy-eager / ``.over`` per-group warm-up), edge
(validation / empty / warm-up / endpoint null / interior null spanned / NaN), correctness (vs the closed-form reference
and a frozen golden master), and properties (reference agreement for any input and under missing data). Categories are
split into classes; cross-cutting categories use markers.
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import total_return_rolling_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_REFERENCE,
    WINDOW_MAX,
    apply_expr,
    assert_matches,
    positive_missing_data,
)

from pomata.metrics import total_return_rolling

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- total_return_rolling is WINDOWED and series-valued. Facts (mirroring the windowed indicators):
#   1. shape   length-preserving: one output row per input row; the first ``window - 1`` rows are warm-up ``null``
#   2. domain  positive equity (a unit-start growth curve); the missing variant mixes null / NaN
#   3. window  window_min = 2 (a windowed return needs two endpoints) .. WINDOW_MAX
# Each case carries (window - 1) warm-up rows + a window of defined output, so no example is all warm-up. The endpoint
# operation is exact (a ratio less one), so the property tiers compare to the oracle within the reference band.
# ----------------------------------------------------------------------------------------------------------------------
_EQUITY = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T], window_min: int = 2) -> tuple[list[T], int]:
    """A (series, window) pair sized so every example has a window of defined output past the warm-up."""
    window = draw(st.integers(min_value=window_min, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


class TestTotalReturnRollingContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(total_return_rolling(pl.col(COLUMN_X), 3), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the metric maps a series to a ``Float64`` series of the same length.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 1.1, 1.05, 1.2, 1.15], dtype=pl.Float64)})
        result = frame.select(total_return_rolling(pl.col(COLUMN_X), 3).alias("t"))
        assert result.height == frame.height
        assert result.schema["t"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 1.1, 1.05, 1.2, 1.15], dtype=pl.Float64)})
        expr = total_return_rolling(pl.col(COLUMN_X), 3).alias("t")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` each group warms up independently and the window never spans a boundary.
        """
        group_a = [1.0, 1.1, 1.05, 1.2]
        group_b = [1.0, 0.95, 1.05, 1.02]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * len(group_a) + ["b"] * len(group_b), COLUMN_X: group_a + group_b})
        grouped = frame.select(total_return_rolling(pl.col(COLUMN_X), 2).over(GROUP_KEY).alias("t"))["t"].to_list()
        expected = total_return_rolling_reference(group_a, 2) + total_return_rolling_reference(group_b, 2)
        assert_matches(grouped, expected, rel_tol=RELATIVE_TOLERANCE_REFERENCE)


class TestTotalReturnRollingEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_window_below_two_raises(self) -> None:
        """
        Verifies that ``window < 2`` raises ``ValueError`` (a windowed return needs two endpoints).
        """
        with pytest.raises(ValueError, match="window must be >= 2"):
            total_return_rolling(pl.col(COLUMN_X), 1)

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields an empty result.
        """
        assert apply_expr([], total_return_rolling(pl.col(COLUMN_X), 3)) == []

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are ``null`` and the rest match the reference.
        """
        values = [1.0, 1.1, 1.05, 1.2, 1.15]
        assert_matches(
            apply_expr(values, total_return_rolling(pl.col(COLUMN_X), 3)),
            total_return_rolling_reference(values, 3),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_endpoint_null_is_null(self) -> None:
        """
        Verifies that a ``null`` at a window endpoint yields ``null`` (the result depends on both endpoints).
        """
        assert_matches(apply_expr([1.0, 1.1, None], total_return_rolling(pl.col(COLUMN_X), 3)), [None, None, None])

    def test_interior_null_is_spanned(self) -> None:
        """
        Verifies that an interior ``null`` is spanned, not propagated: only the two endpoints determine the result.
        """
        values = [1.0, None, 1.2]
        result = apply_expr(values, total_return_rolling(pl.col(COLUMN_X), 3))
        assert_matches(result, total_return_rolling_reference(values, 3))
        assert result[:2] == [None, None]
        assert result[2] == pytest.approx(0.2)

    def test_endpoint_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` at a window endpoint propagates to ``NaN``.
        """
        values = [1.0, 1.1, math.nan]
        assert_matches(
            apply_expr(values, total_return_rolling(pl.col(COLUMN_X), 3)),
            total_return_rolling_reference(values, 3),
        )


class TestTotalReturnRollingCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative equity curve.
        """
        values = [1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4]
        assert_matches(
            apply_expr(values, total_return_rolling(pl.col(COLUMN_X), 4)),
            total_return_rolling_reference(values, 4),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the rolling total return over a window of three.
        """
        values = [1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25]
        assert_matches(
            apply_expr(values, total_return_rolling(pl.col(COLUMN_X), 3).round(4)),
            [None, None, 0.05, 0.0909, 0.0952, 0.0833, 0.087],
        )


class TestTotalReturnRollingProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_EQUITY))
    def test_matches_reference_for_any_input(self, case: tuple[list[float], int]) -> None:
        """
        Verifies that, for any positive equity series and window, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, total_return_rolling(pl.col(COLUMN_X), window)),
            total_return_rolling_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(positive_missing_data(high=1e4)))
    def test_matches_reference_under_missing_data(self, case: tuple[list[float | None], int]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, total_return_rolling(pl.col(COLUMN_X), window)),
            total_return_rolling_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
