"""
Tests for ``pomata.metrics.skewness_rolling`` — the rolling (windowed) twin of :func:`pomata.metrics.skewness`.

``skewness_rolling`` is single-input and WINDOWED-SERIES-VALUED (a return series → a series the same length, one value
per trailing window), so tests use the shared ``apply_expr`` helper; ``assert_matches`` and the naive
``skewness_rolling_reference`` oracle (the reducing :func:`skewness` recomputed over each window) are shared across the
suite. A window holding any ``null`` is ``null`` (it must hold ``window`` non-null values); a ``NaN`` inside a window
propagates.

The ladder is the canonical one: contract (type / length-preserving / lazy-eager / ``.over`` per-group warm-up), edge
(validation / empty / warm-up / null-in-window / NaN / constant), correctness (vs the closed-form reference and a frozen
golden master), and properties (reference agreement for any input and under missing data). Categories are split into
classes; cross-cutting categories use markers.
"""

import math

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import skewness_rolling_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    WINDOW_MAX,
    apply_expr,
    assert_matches,
    windows_well_conditioned,
)

from pomata.metrics import skewness_rolling

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- skewness_rolling is WINDOWED and series-valued (a standardized third moment per window). Facts:
#   1. shape   length-preserving: one output row per input row; the first ``window - 1`` rows are warm-up ``null``
#   2. domain  magnitude-bounded returns (``|r|`` in [0.01, 1], sign-varied) so a window never mixes a subnormal with
#              an ``O(1)`` value (that mix makes the one-pass cube cancel catastrophically); missing mixes null / NaN
#   3. window  window_min = 3 (skewness is degenerate -- identically 0 -- for two points) .. WINDOW_MAX
# The one-pass rolling moments diverge from the two-pass oracle on ill-conditioned windows, so the property tiers
# require every window to be well-conditioned (variance a real fraction of the magnitude); agreement is a 1e-6 band.
# ----------------------------------------------------------------------------------------------------------------------
_VALUE = st.one_of(
    st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
)
_VALUE_MISSING = st.one_of(st.none(), st.just(math.nan), _VALUE)


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T], window_min: int = 3) -> tuple[list[T], int]:
    """A (series, window) pair sized so every example has a window of defined output past the warm-up."""
    window = draw(st.integers(min_value=window_min, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


class TestSkewnessRollingContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(skewness_rolling(pl.col(COLUMN_X), 3), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the metric maps a series to a ``Float64`` series of the same length.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.03, -0.01, 0.02], dtype=pl.Float64)})
        result = frame.select(skewness_rolling(pl.col(COLUMN_X), 3).alias("s"))
        assert result.height == frame.height
        assert result.schema["s"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.03, -0.01, 0.02], dtype=pl.Float64)})
        expr = skewness_rolling(pl.col(COLUMN_X), 3).alias("s")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` each group warms up independently and the window never spans a boundary.
        """
        group_a = [0.01, -0.02, 0.03, -0.01]
        group_b = [0.02, -0.05, 0.01, -0.01]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * len(group_a) + ["b"] * len(group_b), COLUMN_X: group_a + group_b})
        grouped = frame.select(skewness_rolling(pl.col(COLUMN_X), 3).over(GROUP_KEY).alias("s"))["s"].to_list()
        expected = skewness_rolling_reference(group_a, 3) + skewness_rolling_reference(group_b, 3)
        assert_matches(grouped, expected, rel_tol=RELATIVE_TOLERANCE_REFERENCE)


class TestSkewnessRollingEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_window_below_two_raises(self) -> None:
        """
        Verifies that ``window < 2`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 2"):
            skewness_rolling(pl.col(COLUMN_X), 1)

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields an empty result.
        """
        assert apply_expr([], skewness_rolling(pl.col(COLUMN_X), 3)) == []

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are ``null`` and the rest match the reference.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, skewness_rolling(pl.col(COLUMN_X), 3)),
            skewness_rolling_reference(values, 3),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_null_in_window_is_null(self) -> None:
        """
        Verifies that a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        """
        values = [0.01, None, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, skewness_rolling(pl.col(COLUMN_X), 3)),
            skewness_rolling_reference(values, 3),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` inside a window propagates to ``NaN`` for the windows that touch it.
        """
        values = [0.01, math.nan, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, skewness_rolling(pl.col(COLUMN_X), 3)),
            skewness_rolling_reference(values, 3),
        )

    def test_constant_window_is_nan(self) -> None:
        """
        Verifies that a constant window has zero variance, so the skewness is ``NaN`` -- the value is not exactly
        representable, so the one-pass central moments leave a residue that must be guarded, not surfaced as a finite.
        """
        assert_matches(
            apply_expr([0.3, 0.3, 0.3, 0.3], skewness_rolling(pl.col(COLUMN_X), 3)),
            [None, None, math.nan, math.nan],
        )


class TestSkewnessRollingCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018]
        assert_matches(
            apply_expr(values, skewness_rolling(pl.col(COLUMN_X), 4)),
            skewness_rolling_reference(values, 4),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the rolling skewness over a window of four.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015]
        assert_matches(
            apply_expr(values, skewness_rolling(pl.col(COLUMN_X), 4).round(4)),
            [None, None, None, 0.278, 0.0, 0.0, 0.6568],
        )


class TestSkewnessRollingProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_VALUE))
    def test_matches_reference_for_any_input(self, case: tuple[list[float], int]) -> None:
        """
        Verifies that, for any well-conditioned series and window, the implementation matches the naive reference.
        """
        values, window = case
        assume(windows_well_conditioned(values, window))
        assert_matches(
            apply_expr(values, skewness_rolling(pl.col(COLUMN_X), window)),
            skewness_rolling_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_VALUE_MISSING))
    def test_matches_reference_under_missing_data(self, case: tuple[list[float | None], int]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window = case
        assume(windows_well_conditioned(values, window))
        assert_matches(
            apply_expr(values, skewness_rolling(pl.col(COLUMN_X), window)),
            skewness_rolling_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
