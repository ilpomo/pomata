"""
Tests for ``pomata.metrics.cagr_rolling`` — the rolling (windowed) twin of :func:`pomata.metrics.cagr`.

``cagr_rolling`` is single-input and WINDOWED-SERIES-VALUED (an equity series → a series the same length, one value per
trailing window), so tests use the shared ``apply_expr`` helper; ``assert_matches`` and the naive
``cagr_rolling_reference`` oracle (the window's endpoint ratio annualized) are shared across the suite. As an endpoint
quantity it depends only on the first and last equity of each window: a ``null`` at either endpoint yields ``null``, an
interior ``null`` is spanned, and a ``NaN`` at an endpoint propagates.

The ladder is the canonical one: contract (type / length-preserving / lazy-eager / ``.over`` per-group warm-up), edge
(validation / empty / warm-up / endpoint null / interior null / NaN), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement for any input and under missing data). Categories are split
into classes; cross-cutting categories use markers.
"""

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import cagr_rolling_reference
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

from pomata.metrics import cagr_rolling

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- cagr_rolling is WINDOWED and series-valued. Facts (mirroring the windowed siblings):
#   1. shape   length-preserving: one output row per input row; the first ``window - 1`` rows are warm-up ``null``
#   2. domain  positive equity in [0.1, 10] so the annualizing power ``** (P / window)`` stays finite; the missing
#              variant mixes positive finite values with null / NaN
#   3. window  window_min = 2 (an endpoint ratio needs two observations) .. WINDOW_MAX
# Each case carries (window - 1) warm-up rows + a window of defined output, so no example is all warm-up. The endpoint
# / max operations are exact, so no per-window conditioning guard is needed. PERIODS is drawn from a small annualizing
# set (NOT 252) so the power stays in a sane range.
# ----------------------------------------------------------------------------------------------------------------------
PERIODS = 4
_EQUITY = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
_PERIODS = st.sampled_from([4, 12, 52])


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T], window_min: int = 2) -> tuple[list[T], int]:
    """A (series, window) pair sized so every example has a window of defined output past the warm-up."""
    window = draw(st.integers(min_value=window_min, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


class TestCagrRollingContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(cagr_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the metric maps a series to a ``Float64`` series of the same length.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 1.1, 1.05, 1.2, 1.15], dtype=pl.Float64)})
        result = frame.select(cagr_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS).alias("c"))
        assert result.height == frame.height
        assert result.schema["c"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 1.1, 1.05, 1.2, 1.15], dtype=pl.Float64)})
        expr = cagr_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS).alias("c")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` each group warms up independently and the window never spans a boundary.
        """
        group_a = [1.0, 1.1, 1.05, 1.2]
        group_b = [1.0, 1.2, 1.1, 1.3]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * len(group_a) + ["b"] * len(group_b), COLUMN_X: group_a + group_b})
        grouped = frame.select(cagr_rolling(pl.col(COLUMN_X), 2, periods_per_year=PERIODS).over(GROUP_KEY).alias("c"))[
            "c"
        ].to_list()
        expected = cagr_rolling_reference(group_a, 2, PERIODS) + cagr_rolling_reference(group_b, 2, PERIODS)
        assert_matches(grouped, expected, rel_tol=RELATIVE_TOLERANCE_REFERENCE)


class TestCagrRollingEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_window_below_two_raises(self) -> None:
        """
        Verifies that ``window < 2`` raises ``ValueError`` (an endpoint ratio needs two observations).
        """
        with pytest.raises(ValueError, match="window must be >= 2"):
            cagr_rolling(pl.col(COLUMN_X), 1, periods_per_year=PERIODS)

    def test_periods_per_year_below_one_raises(self) -> None:
        """
        Verifies that ``periods_per_year < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="periods_per_year must be >= 1"):
            cagr_rolling(pl.col(COLUMN_X), 3, periods_per_year=0)

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields an empty result.
        """
        assert apply_expr([], cagr_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)) == []

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are ``null`` and the rest match the reference.
        """
        values = [1.0, 1.1, 1.05, 1.2, 1.15]
        assert_matches(
            apply_expr(values, cagr_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)),
            cagr_rolling_reference(values, 3, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_endpoint_null_is_null(self) -> None:
        """
        Verifies that a ``null`` at either window endpoint yields ``null`` for the windows that touch it.
        """
        values = [None, 1.1, 1.05, 1.2, None]
        assert_matches(
            apply_expr(values, cagr_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)),
            cagr_rolling_reference(values, 3, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_interior_null_is_spanned(self) -> None:
        """
        Verifies that an interior ``null`` does not affect the result (the metric reads only the window's endpoints).
        """
        values = [1.0, None, 1.05, None, 1.15]
        assert_matches(
            apply_expr(values, cagr_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)),
            cagr_rolling_reference(values, 3, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_endpoint_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` at a window endpoint propagates to ``NaN`` for the windows that touch it.
        """
        values = [1.0, 1.1, float("nan"), 1.2, 1.15]
        assert_matches(
            apply_expr(values, cagr_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)),
            cagr_rolling_reference(values, 3, PERIODS),
        )


class TestCagrRollingCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative equity curve.
        """
        values = [1.05, 1.1, 1.08, 1.15, 1.2, 1.18, 1.25, 1.3]
        assert_matches(
            apply_expr(values, cagr_rolling(pl.col(COLUMN_X), 4, periods_per_year=PERIODS)),
            cagr_rolling_reference(values, 4, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the rolling compound annual growth rate over a window of three.
        """
        values = [1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25]
        assert_matches(
            apply_expr(values, cagr_rolling(pl.col(COLUMN_X), 3, periods_per_year=4).round(4)),
            [None, None, 0.0672, 0.123, 0.129, 0.1126, 0.1176],
        )


class TestCagrRollingProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_EQUITY), periods=_PERIODS)
    def test_matches_reference_for_any_input(self, case: tuple[list[float], int], periods: int) -> None:
        """
        Verifies that, for any positive equity series and window, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, cagr_rolling(pl.col(COLUMN_X), window, periods_per_year=periods)),
            cagr_rolling_reference(values, window, periods),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(positive_missing_data(high=1e4)), periods=_PERIODS)
    def test_matches_reference_under_missing_data(self, case: tuple[list[float | None], int], periods: int) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, cagr_rolling(pl.col(COLUMN_X), window, periods_per_year=periods)),
            cagr_rolling_reference(values, window, periods),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
