"""
Tests for ``pomata.indicators.sma`` — Simple Moving Average, the golden-template indicator.

``sma`` is single-input and windowed, so tests use the shared ``apply_expr`` helper to materialize the factory over a
one-column ``Float64`` frame; ``assert_matches`` and the naive ``sma_reference`` oracle are shared across the suite.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` parity), edge (warm-up / window
boundaries / single-row / null / NaN), correctness (vs the closed-form reference and a frozen golden master), and
properties (reference agreement for any input and scale-homogeneity). Categories are split into classes; cross-cutting
categories elsewhere use markers (see ``tests/README.md``).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import sma_reference
from tests.support import (
    COLUMN_X,
    EXACT_TOLERANCE_FACTOR,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_SCALE,
    WINDOW_MAX,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    input_scale,
    missing_data_floats,
)

from pomata.indicators import sma

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (the window must hold ``window`` non-null values before a result is emitted)
#   2. memory  the oracle is windowed like pomata, so the property holds from the first defined row (M = 0); each
#              example carries D in [window, 2 * window] defined values -- one window of output, never all warm-up
#   3. domain  finite floats over the test's regime (any-input / scale / missing-data / large-magnitude), widened per
#              test below
# Windows span ``window_min`` .. WINDOW_MAX. Repetitions N are the shared CI profile (tests/conftest.py); override
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


class TestSmaContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the window resets per group and never spans group boundaries.
        """
        frame = pl.DataFrame({GROUP_KEY: ["a", "a", "a", "b", "b", "b"], COLUMN_X: [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]})
        result = frame.select(sma(pl.col(COLUMN_X), 2).over(GROUP_KEY).alias("y"))["y"].to_list()
        assert_matches(result, [None, 1.5, 2.5, None, 15.0, 25.0])


class TestSmaEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            sma(pl.col(COLUMN_X), 0)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are null (warm-up) and the first full window is defined.
        """
        result = apply_expr([1.0, 2.0, 3.0, 4.0, 5.0], sma(pl.col(COLUMN_X), 3))
        assert result[:2] == [None, None]
        assert result[2] is not None

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output.
        """
        assert_matches(
            apply_expr([None, None, None, None, None], sma(pl.col(COLUMN_X), 3)), [None, None, None, None, None]
        )

    def test_window_one_is_identity(self) -> None:
        """
        Verifies that ``window == 1`` reproduces the input with no warm-up.
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], sma(pl.col(COLUMN_X), 1)), [1.0, 2.0, 3.0])

    def test_window_equals_length(self) -> None:
        """
        Verifies the single defined value (the mean of the whole series) when ``window`` equals the series length.
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], sma(pl.col(COLUMN_X), 3)), [None, None, 2.0])

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a window exceeding the series length yields an all-null output.
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], sma(pl.col(COLUMN_X), 5)), [None, None, None])

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series: ``window == 1`` returns the value, a larger window is all warm-up.
        """
        assert_matches(apply_expr([42.0], sma(pl.col(COLUMN_X), 1)), [42.0])
        assert_matches(apply_expr([42.0], sma(pl.col(COLUMN_X), 3)), [None])

    def test_null_propagates(self) -> None:
        """
        Verifies that a ``null`` inside the window yields ``null`` there, and the value returns once the window clears.
        """
        assert_matches(apply_expr([1.0, None, 3.0, 4.0], sma(pl.col(COLUMN_X), 2)), [None, None, None, 3.5])

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` inside the window yields ``NaN`` there (``null`` still takes precedence over ``NaN``).
        """
        assert_matches(apply_expr([1.0, math.nan, 3.0, 4.0], sma(pl.col(COLUMN_X), 2)), [None, math.nan, math.nan, 3.5])


class TestSmaCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several windows.
        """
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        for window in (1, 2, 3, 4, 5):
            assert_matches(apply_expr(values, sma(pl.col(COLUMN_X), window)), sma_reference(values, window))

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: SMA(window=3) over [2, 4, 6, 8, 10] == [None, None, 4, 6, 8].
        """
        assert_matches(apply_expr([2.0, 4.0, 6.0, 8.0, 10.0], sma(pl.col(COLUMN_X), 3)), [None, None, 4.0, 6.0, 8.0])


class TestSmaProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], int],
    ) -> None:
        """
        Verifies that, for any series and window, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, sma(pl.col(COLUMN_X), window)),
            sma_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(values) * EXACT_TOLERANCE_FACTOR,
        )

    @given(
        case=_cases(st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(
        self,
        case: tuple[list[float], int],
        exponent: int,
    ) -> None:
        """
        Verifies that SMA is homogeneous of degree 1: ``sma(k * x) == k * sma(x)``. ``k`` is a power of two so the
        rescaling is lossless and cannot introduce a sub-ULP drift between the two sides.
        """
        k = 2.0**exponent
        values, window = case
        scaled_values = [value * k for value in values]
        result_base = apply_expr(values, sma(pl.col(COLUMN_X), window))
        result_scaled = apply_expr(scaled_values, sma(pl.col(COLUMN_X), window))
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=1)

    @given(case=_cases(missing_data_floats()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[float | None], int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, sma(pl.col(COLUMN_X), window)),
            sma_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(values) * EXACT_TOLERANCE_FACTOR,
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
            apply_expr(scaled_values, sma(pl.col(COLUMN_X), window)),
            sma_reference(scaled_values, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(scaled_values) * EXACT_TOLERANCE_FACTOR,
        )
