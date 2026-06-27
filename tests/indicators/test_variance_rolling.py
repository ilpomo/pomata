"""
Tests for ``pomata.indicators.variance_rolling`` — rolling variance over a window.

``variance_rolling`` is single-input and windowed, so tests use the shared ``apply_expr`` helper to materialize the
factory over a one-column ``Float64`` frame; ``assert_matches`` and the naive ``variance_rolling_reference`` oracle are
shared across the suite.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` independence), edge (warm-up / window
boundaries / single-row / null / NaN), correctness (vs the closed-form reference incl. ``ddof`` and a frozen golden
master), and properties (reference agreement incl. ``ddof`` and missing data, degree-2 scale-homogeneity, and
large-magnitude stability). Categories are split into classes; cross-cutting categories use markers (see
``tests/README.md``).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.indicators.oracles import variance_rolling_reference
from tests.support import (
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_SCALE,
    SUBNORMAL_FLOOR,
    VARIANCE_TOLERANCE_FACTOR,
    WINDOW_MAX,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    input_scale,
    missing_data_floats,
    subnormal_safe_floats,
)

from pomata.indicators import variance_rolling

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (the window must hold ``window`` non-null values before a result is emitted)
#   2. memory  the oracle shares pomata's seeding, so the property holds from the first defined row (M = 0); each
#              example carries D in [window, 2 * window] defined values -- one window of output, never all warm-up
#   3. domain  subnormal_safe_floats(bound): finite values floored away from the subnormal-square underflow (the squared
#              deviation must stay representable); ``bound`` is the safe magnitude, widened per test below
# Windows span ``window_min`` .. WINDOW_MAX; variance needs window >= ddof + 1 (a variance over ``ddof`` points is
# undefined), which ``_ddof_cases`` guarantees. Repetitions N are the shared CI profile (tests/conftest.py); override
# per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](
    draw: st.DrawFn,
    values: st.SearchStrategy[T],
    window_min: int = 1,
) -> tuple[list[T], int]:
    """
    A (series, window) pair sized from the facts above: ``window`` over its regimes, length = warm-up + a window of
    defined values, so every example has output to check (never an all-warm-up series, the waste a ``window`` decoupled
    from the length would cause).
    """
    window = draw(st.integers(min_value=window_min, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


@st.composite
def _ddof_cases[T](draw: st.DrawFn, values: st.SearchStrategy[T]) -> tuple[list[T], int, int]:
    """
    A :func:`_cases` pair plus a ``ddof`` in {0, 1}, with ``window >= ddof + 1`` guaranteed.
    """
    ddof = draw(st.integers(min_value=0, max_value=1))
    series, window = draw(_cases(values, window_min=ddof + 1))
    return series, window, ddof


class TestVarianceRollingContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(variance_rolling(pl.col(COLUMN_X), 3), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 2.0, 3.0, 4.0, 5.0])})
        result = frame.select(variance_rolling(pl.col(COLUMN_X), 3).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 2.0, 3.0, 4.0, 5.0])})
        result_eager = frame.select(variance_rolling(pl.col(COLUMN_X), 3).alias("y"))
        result_lazy = frame.lazy().select(variance_rolling(pl.col(COLUMN_X), 3).alias("y")).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the window resets per group and never spans group boundaries.
        """
        frame = pl.DataFrame({GROUP_KEY: ["a", "a", "a", "b", "b", "b"], COLUMN_X: [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]})
        result = frame.select(variance_rolling(pl.col(COLUMN_X), 2).over(GROUP_KEY).alias("y"))["y"].to_list()
        assert_matches(result, [None, 0.25, 0.25, None, 25.0, 25.0])


class TestVarianceRollingEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            variance_rolling(pl.col(COLUMN_X), 0)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are null (warm-up) and the first full window is defined.
        """
        result = apply_expr([1.0, 2.0, 3.0, 4.0, 5.0], variance_rolling(pl.col(COLUMN_X), 3))
        assert result[:2] == [None, None]
        assert result[2] is not None

    def test_window_one_is_zero(self) -> None:
        """
        Verifies that ``window == 1`` has no spread: the variance is ``0`` at every row, with no warm-up.
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], variance_rolling(pl.col(COLUMN_X), 1)), [0.0, 0.0, 0.0])

    def test_window_equals_length(self) -> None:
        """
        Verifies the single defined value (the variance of the whole series) when ``window`` equals the series length.
        """
        assert_matches(apply_expr([1.0, 3.0], variance_rolling(pl.col(COLUMN_X), 2)), [None, 1.0])

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series: ``window == 1`` is ``0`` (no spread), a larger window is warm-up.
        """
        assert_matches(apply_expr([42.0], variance_rolling(pl.col(COLUMN_X), 1)), [0.0])
        assert_matches(apply_expr([42.0], variance_rolling(pl.col(COLUMN_X), 3)), [None])

    def test_null_propagates(self) -> None:
        """
        Verifies that a ``null`` inside the window yields ``null`` there, and the value returns once the window clears.
        """
        result = apply_expr([1.0, None, 3.0, 4.0], variance_rolling(pl.col(COLUMN_X), 2))
        assert_matches(result, [None, None, None, 0.25])

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` inside the window yields ``NaN`` there (``null`` still takes precedence over ``NaN``).
        """
        result = apply_expr([1.0, math.nan, 3.0, 4.0], variance_rolling(pl.col(COLUMN_X), 2))
        assert_matches(result, [None, math.nan, math.nan, 0.25])

    def test_ddof_at_or_above_window_raises(self) -> None:
        """
        Verifies that a non-positive divisor (``ddof >= window``) raises ``ValueError`` rather than returning a silent
        all-null line: ``ddof == window`` and ``ddof > window`` are both rejected.
        """
        with pytest.raises(ValueError, match="ddof must be < window"):
            variance_rolling(pl.col(COLUMN_X), 1, ddof=1)
        with pytest.raises(ValueError, match="ddof must be < window"):
            variance_rolling(pl.col(COLUMN_X), 2, ddof=2)
        with pytest.raises(ValueError, match="ddof must be < window"):
            variance_rolling(pl.col(COLUMN_X), 2, ddof=5)

    def test_window_exceeds_length(self) -> None:
        """
        Verifies the whole output is null when ``window`` exceeds the series length (no full window ever forms).
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], variance_rolling(pl.col(COLUMN_X), 5)), [None, None, None])

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields an empty result.
        """
        assert_matches(apply_expr([], variance_rolling(pl.col(COLUMN_X), 3)), [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series stays null (no window ever holds the required non-null values).
        """
        assert_matches(apply_expr([None, None, None], variance_rolling(pl.col(COLUMN_X), 2)), [None, None, None])


class TestVarianceRollingCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows (population, ``ddof=0``).
        """
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        for window in (1, 2, 3, 4, 5):
            result = apply_expr(values, variance_rolling(pl.col(COLUMN_X), window))
            assert_matches(result, variance_rolling_reference(values, window))

    def test_matches_reference_sample_ddof(self) -> None:
        """
        Verifies agreement with the reference for the sample variance (``ddof=1``) across several windows.
        """
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        for window in (2, 3, 4, 5):
            result = apply_expr(values, variance_rolling(pl.col(COLUMN_X), window, ddof=1))
            assert_matches(result, variance_rolling_reference(values, window, ddof=1))

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: population variance(window=2) over [2, 4, 4, 8] == [None, 1, 0, 4].
        """
        assert_matches(apply_expr([2.0, 4.0, 4.0, 8.0], variance_rolling(pl.col(COLUMN_X), 2)), [None, 1.0, 0.0, 4.0])

    def test_sample_ddof_golden(self) -> None:
        """
        Verifies the frozen sample reference: variance(window=3, ddof=1) over [1, 3, 5] == [None, None, 4].
        """
        result = apply_expr([1.0, 3.0, 5.0], variance_rolling(pl.col(COLUMN_X), 3, ddof=1))
        assert_matches(result, [None, None, 4.0])


class TestVarianceRollingProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_ddof_cases(subnormal_safe_floats(bound=1e6)))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], int, int],
    ) -> None:
        """
        Verifies that, for any series, window, and ``ddof`` in {0, 1}, the implementation matches the naive reference.
        """
        values, window, ddof = case
        assert_matches(
            apply_expr(values, variance_rolling(pl.col(COLUMN_X), window, ddof=ddof)),
            variance_rolling_reference(values, window, ddof=ddof),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(values) ** 2 * VARIANCE_TOLERANCE_FACTOR,
        )

    @given(
        case=_cases(subnormal_safe_floats()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(
        self,
        case: tuple[list[float], int],
        exponent: int,
    ) -> None:
        """
        Verifies that variance is homogeneous of degree 2: ``variance(k * x) == k**2 * variance(x)``. ``k`` is a power
        of two so the rescaling is lossless and cannot introduce a sub-ULP drift into the squared deviations.
        """
        k = 2.0**exponent
        values, window = case
        scaled_values = [value * k for value in values]
        result_base = apply_expr(values, variance_rolling(pl.col(COLUMN_X), window))
        result_scaled = apply_expr(scaled_values, variance_rolling(pl.col(COLUMN_X), window))
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=2)

    @given(case=_cases(missing_data_floats(min_magnitude=SUBNORMAL_FLOOR)))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[float | None], int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, variance_rolling(pl.col(COLUMN_X), window)),
            variance_rolling_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(values) ** 2 * VARIANCE_TOLERANCE_FACTOR,
        )

    @given(
        case=_cases(st.floats(min_value=1e-3, max_value=1.0, allow_nan=False, allow_infinity=False)),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: tuple[list[float], int],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme positive magnitudes the implementation stays finite where the reference is and agrees.
        """
        values, window = case
        scaled_values = [value * scale for value in values]
        assert_matches(
            apply_expr(scaled_values, variance_rolling(pl.col(COLUMN_X), window)),
            variance_rolling_reference(scaled_values, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(scaled_values) ** 2 * VARIANCE_TOLERANCE_FACTOR,
        )
