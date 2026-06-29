"""
Tests for ``pomata.metrics.sortino_ratio_rolling`` — the rolling (windowed) twin of
:func:`pomata.metrics.sortino_ratio`.

``sortino_ratio_rolling`` is single-input and WINDOWED-SERIES-VALUED (a return series → a series the same length, one
value per trailing window), so tests use the shared ``apply_expr`` helper; ``assert_matches`` and the naive
``sortino_ratio_rolling_reference`` oracle (the reducing :func:`sortino_ratio` recomputed over each window) are shared
across the suite. A window holding any ``null`` is ``null`` (it must hold ``window`` non-null values); a ``NaN`` inside
a window propagates.

The ladder is the canonical one: contract (type / length-preserving / lazy-eager / ``.over`` per-group warm-up), edge
(validation / empty / warm-up / null-in-window / NaN / no-downside), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement for any input and under missing data). Categories are split
into classes; cross-cutting categories use markers.
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import sortino_ratio_rolling_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    WINDOW_MAX,
    apply_expr,
    assert_matches,
)

from pomata.metrics import sortino_ratio_rolling

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- sortino_ratio_rolling is WINDOWED and series-valued (a mean over a downside deviation per window).
# Facts:
#   1. shape   length-preserving: one output row per input row; the first ``window - 1`` rows are warm-up ``null``
#   2. domain  magnitude-bounded returns (``|r|`` in [0.01, 1], sign-varied): the Sortino ratio is scale-invariant, and
#              same-magnitude values keep the one-pass sliding downside mean free of cross-window cancellation; missing
#              mixes null / NaN
#   3. window  window_min = 1 (the downside deviation uses a population mean, defined for one observation) .. WINDOW_MAX
# The denominator (the downside deviation) is unbounded toward zero, so the property tiers skip the tiny-but-non-zero
# regime where the one-pass sliding sum cannot track the two-pass oracle; the dimensionless ratio then agrees to 1e-6.
# ----------------------------------------------------------------------------------------------------------------------
PERIODS = 252
_DOWNSIDE_FLOOR = 1e-3
_VALUE = st.one_of(
    st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
)
_VALUE_MISSING = st.one_of(st.none(), st.just(math.nan), _VALUE)


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T], window_min: int = 1) -> tuple[list[T], int]:
    """A (series, window) pair sized so every example has a window of defined output past the warm-up."""
    window = draw(st.integers(min_value=window_min, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


def _windows_conditioned(values: Sequence[float | None], window: int) -> bool:
    """
    Whether every trailing window's downside deviation is either zero (no downside) or a real fraction of the magnitude.

    Sortino is a ratio whose denominator (the downside deviation about zero) is unbounded toward zero; a tiny-but-non-
    zero downside left by the one-pass sliding sum (once a large value exits) is the one regime where it cannot track
    the two-pass oracle, so the property tiers skip it while keeping the well-defined no-downside (``+inf``) case.
    """
    for index in range(window - 1, len(values)):
        finite = [
            value for value in values[index - window + 1 : index + 1] if value is not None and not math.isnan(value)
        ]
        if not finite:
            continue
        downside_square = sum(value * value for value in finite if value < 0.0) / len(finite)
        scale = max(abs(value) for value in finite) or 1.0
        if 0.0 < downside_square < (scale * _DOWNSIDE_FLOOR) ** 2:
            return False
    return True


class TestSortinoRollingContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(sortino_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the metric maps a series to a ``Float64`` series of the same length.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.03, -0.01, 0.02], dtype=pl.Float64)})
        result = frame.select(sortino_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS).alias("s"))
        assert result.height == frame.height
        assert result.schema["s"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.03, -0.01, 0.02], dtype=pl.Float64)})
        expr = sortino_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS).alias("s")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` each group warms up independently and the window never spans a boundary.
        """
        group_a = [0.01, -0.02, 0.03, -0.01]
        group_b = [0.02, -0.05, 0.01, -0.01]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * len(group_a) + ["b"] * len(group_b), COLUMN_X: group_a + group_b})
        grouped = frame.select(
            sortino_ratio_rolling(pl.col(COLUMN_X), 2, periods_per_year=PERIODS).over(GROUP_KEY).alias("s")
        )["s"].to_list()
        expected = sortino_ratio_rolling_reference(group_a, 2, PERIODS) + sortino_ratio_rolling_reference(
            group_b, 2, PERIODS
        )
        assert_matches(grouped, expected, rel_tol=RELATIVE_TOLERANCE_REFERENCE)


class TestSortinoRollingEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            sortino_ratio_rolling(pl.col(COLUMN_X), 0, periods_per_year=PERIODS)

    def test_periods_per_year_below_one_raises(self) -> None:
        """
        Verifies that ``periods_per_year < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="periods_per_year must be >= 1"):
            sortino_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=0)

    def test_non_finite_risk_free_rate_raises(self) -> None:
        """
        Verifies that a non-finite ``risk_free_rate`` raises ``ValueError``.
        """
        for invalid in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="risk_free_rate must be a finite number"):
                sortino_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS, risk_free_rate=invalid)

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields an empty result.
        """
        assert apply_expr([], sortino_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)) == []

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are ``null`` and the rest match the reference.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, sortino_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)),
            sortino_ratio_rolling_reference(values, 3, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_null_in_window_is_null(self) -> None:
        """
        Verifies that a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        """
        values = [0.01, None, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, sortino_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)),
            sortino_ratio_rolling_reference(values, 3, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` inside a window propagates to ``NaN`` for the windows that touch it.
        """
        values = [0.01, math.nan, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, sortino_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)),
            sortino_ratio_rolling_reference(values, 3, PERIODS),
        )

    def test_no_downside_is_inf(self) -> None:
        """
        Verifies that a window with no downside has zero downside deviation with a positive mean, so the ratio is
        ``+inf``.
        """
        assert_matches(
            apply_expr([0.01, 0.02, 0.03, 0.04], sortino_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=PERIODS)),
            [None, None, math.inf, math.inf],
        )


class TestSortinoRollingCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, 0.01, -0.02, 0.018]
        assert_matches(
            apply_expr(
                values, sortino_ratio_rolling(pl.col(COLUMN_X), 4, periods_per_year=PERIODS, risk_free_rate=0.02)
            ),
            sortino_ratio_rolling_reference(values, 4, PERIODS, 0.02),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the daily-annualized rolling Sortino over a window of three.
        """
        values = [0.03, -0.01, 0.02, -0.015, 0.025, -0.005, 0.02]
        assert_matches(
            apply_expr(values, sortino_ratio_rolling(pl.col(COLUMN_X), 3, periods_per_year=252).round(4)),
            [None, None, 36.6606, -2.542, 18.3303, 2.8983, 73.3212],
        )


class TestSortinoRollingProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_VALUE))
    def test_matches_reference_for_any_input(self, case: tuple[list[float], int]) -> None:
        """
        Verifies that, for any well-conditioned series and window, the implementation matches the naive reference.
        """
        values, window = case
        assume(_windows_conditioned(values, window))
        assert_matches(
            apply_expr(values, sortino_ratio_rolling(pl.col(COLUMN_X), window, periods_per_year=PERIODS)),
            sortino_ratio_rolling_reference(values, window, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_VALUE_MISSING))
    def test_matches_reference_under_missing_data(self, case: tuple[list[float | None], int]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window = case
        assume(_windows_conditioned(values, window))
        assert_matches(
            apply_expr(values, sortino_ratio_rolling(pl.col(COLUMN_X), window, periods_per_year=PERIODS)),
            sortino_ratio_rolling_reference(values, window, PERIODS),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
