"""
Tests for ``pomata.metrics.value_at_risk_rolling`` — the rolling (windowed) twin of
:func:`pomata.metrics.value_at_risk`.

``value_at_risk_rolling`` is single-input and WINDOWED-SERIES-VALUED (a return series → a series the same length, one
value per trailing window), so tests use the shared ``apply_expr`` helper; ``assert_matches`` and the naive
``value_at_risk_rolling_reference`` oracle (the reducing :func:`value_at_risk` recomputed over each window) are shared
across the suite. The rolling null/NaN policy differs from the reducing one: a window holding any ``null`` is ``null``
(it must hold ``window`` non-null values), and a ``NaN`` inside a window propagates.

The ladder is the canonical one: contract (type / length-preserving / lazy-eager / ``.over`` per-group warm-up), edge
(validation / empty / warm-up / null-in-window / NaN / sign convention), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement for any input and under missing data). Categories are split
into classes; cross-cutting categories use markers.
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import value_at_risk_rolling_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_REFERENCE,
    WINDOW_MAX,
    apply_expr,
    assert_matches,
    missing_data_floats,
    streaming_abs_tol,
    subnormal_safe_floats,
)

from pomata.metrics import value_at_risk_rolling


def _abs_tol(values: Sequence[float | None]) -> float:
    """The magnitude-relative absolute tolerance for a historical quantile on the returns' own scale."""
    return streaming_abs_tol(values)


# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- value_at_risk_rolling is WINDOWED and series-valued. Facts (mirroring the windowed indicators):
#   1. shape   length-preserving: one output row per input row; the first ``window - 1`` rows are warm-up ``null``
#   2. domain  subnormal_safe_floats: finite returns floored off subnormal so the empirical quantile is
#              well-conditioned; the missing variant mixes null / NaN
#   3. window  window_min = 1 (a single-observation window is the trivial quantile) .. WINDOW_MAX
# Each case carries (window - 1) warm-up rows + a window of defined output, so no example is all warm-up. A historical
# quantile is an order statistic, so the implementation matches the oracle to a tight reference band.
# ----------------------------------------------------------------------------------------------------------------------
CONFIDENCE = 0.95


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T], window_min: int = 1) -> tuple[list[T], int]:
    """A (series, window) pair sized so every example has a window of defined output past the warm-up."""
    window = draw(st.integers(min_value=window_min, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


class TestValueAtRiskRollingContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(value_at_risk_rolling(pl.col(COLUMN_X), 3, confidence=CONFIDENCE), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the metric maps a series to a ``Float64`` series of the same length.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.03, -0.01, 0.02], dtype=pl.Float64)})
        result = frame.select(value_at_risk_rolling(pl.col(COLUMN_X), 3, confidence=CONFIDENCE).alias("v"))
        assert result.height == frame.height
        assert result.schema["v"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.03, -0.01, 0.02], dtype=pl.Float64)})
        expr = value_at_risk_rolling(pl.col(COLUMN_X), 3, confidence=CONFIDENCE).alias("v")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` each group warms up independently and the window never spans a boundary.
        """
        group_a = [0.01, -0.02, 0.03, -0.01]
        group_b = [0.02, -0.05, 0.01, -0.01]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * len(group_a) + ["b"] * len(group_b), COLUMN_X: group_a + group_b})
        grouped = frame.select(
            value_at_risk_rolling(pl.col(COLUMN_X), 2, confidence=CONFIDENCE).over(GROUP_KEY).alias("v")
        )["v"].to_list()
        expected = value_at_risk_rolling_reference(group_a, 2) + value_at_risk_rolling_reference(group_b, 2)
        assert_matches(grouped, expected, rel_tol=RELATIVE_TOLERANCE_REFERENCE)


class TestValueAtRiskRollingEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            value_at_risk_rolling(pl.col(COLUMN_X), 0, confidence=CONFIDENCE)

    def test_confidence_out_of_range_raises(self) -> None:
        """
        Verifies that a ``confidence`` outside the open interval ``(0, 1)`` raises ``ValueError``.
        """
        for invalid in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="confidence must be in the open interval"):
                value_at_risk_rolling(pl.col(COLUMN_X), 3, confidence=invalid)

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields an empty result.
        """
        assert apply_expr([], value_at_risk_rolling(pl.col(COLUMN_X), 3, confidence=CONFIDENCE)) == []

    def test_warm_up_is_null(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are ``null`` and the rest match the reference.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, value_at_risk_rolling(pl.col(COLUMN_X), 3, confidence=CONFIDENCE)),
            value_at_risk_rolling_reference(values, 3),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_null_in_window_is_null(self) -> None:
        """
        Verifies that a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        """
        values = [0.01, None, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, value_at_risk_rolling(pl.col(COLUMN_X), 3, confidence=CONFIDENCE)),
            value_at_risk_rolling_reference(values, 3),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` inside a window propagates to ``NaN`` for the windows that touch it.
        """
        values = [0.01, math.nan, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, value_at_risk_rolling(pl.col(COLUMN_X), 3, confidence=CONFIDENCE)),
            value_at_risk_rolling_reference(values, 3),
        )

    def test_sign_convention_is_signed_quantile(self) -> None:
        """
        Verifies the sign convention: the result is the signed return quantile (negative for a loss).
        """
        values = [-0.05, -0.04, -0.03, -0.02, -0.01]
        result = apply_expr(values, value_at_risk_rolling(pl.col(COLUMN_X), 3, confidence=CONFIDENCE))
        assert all(value is None or value < 0.0 for value in result)
        assert_matches(result, value_at_risk_rolling_reference(values, 3), rel_tol=RELATIVE_TOLERANCE_REFERENCE)


class TestValueAtRiskRollingCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018]
        assert_matches(
            apply_expr(values, value_at_risk_rolling(pl.col(COLUMN_X), 4, confidence=CONFIDENCE)),
            value_at_risk_rolling_reference(values, 4),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the 5th-percentile return of each trailing window of four.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015]
        assert_matches(
            apply_expr(values, value_at_risk_rolling(pl.col(COLUMN_X), 4).round(4)),
            [None, None, None, -0.0185, -0.0185, -0.0085, -0.0142],
        )


class TestValueAtRiskRollingProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(subnormal_safe_floats(bound=1e3)))
    def test_matches_reference_for_any_input(self, case: tuple[list[float], int]) -> None:
        """
        Verifies that, for any well-conditioned series and window, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, value_at_risk_rolling(pl.col(COLUMN_X), window, confidence=CONFIDENCE)),
            value_at_risk_rolling_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=_abs_tol(values),
        )

    @given(case=_cases(missing_data_floats(min_magnitude=1e-3)))
    def test_matches_reference_under_missing_data(self, case: tuple[list[float | None], int]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, value_at_risk_rolling(pl.col(COLUMN_X), window, confidence=CONFIDENCE)),
            value_at_risk_rolling_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=_abs_tol(values),
        )
