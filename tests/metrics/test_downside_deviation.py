"""
Tests for ``pomata.metrics.downside_deviation`` — the annualized dispersion of returns below a threshold.

``downside_deviation`` is single-input and REDUCING (a return series → one scalar), so tests read the single output row
of ``apply_expr``; ``assert_matches`` and the naive ``downside_deviation_reference`` oracle are shared across the suite.
At ``threshold = 0`` it is degree-1 homogeneous in the returns, so it carries the scale-homogeneity and large-magnitude
tiers (run at the default threshold).

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (validation / empty / single-row / no-downside / null / NaN), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement incl. missing data, degree-1 scale-homogeneity,
large-magnitude stability). Categories are split into classes; cross-cutting categories use markers.
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.metrics.oracles import downside_deviation_reference
from tests.support import (
    COLUMN_X,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    STREAMING_TOLERANCE_FACTOR,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    input_scale,
    missing_data_floats,
    streaming_abs_tol,
    subnormal_safe_floats,
)

from pomata.metrics import downside_deviation

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- downside_deviation is windowless and REDUCING (M = 0); a case is just a return series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  subnormal_safe_floats(bound): finite returns floored away from the subnormal-square underflow
#   3. scale   degree-1 homogeneous at threshold 0 -> scale-homogeneity + large-magnitude tiers
# PERIODS is a representative annualization. Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
PERIODS = 252


@st.composite
def _cases[T](draw: st.DrawFn, returns: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A return series sized from the facts above."""
    return draw(st.lists(returns, min_size=min_size, max_size=SERIES_MAX))


def _abs_tol(values: Sequence[float | None]) -> float:
    """The magnitude-relative absolute tolerance for the annualized downside deviation."""
    return streaming_abs_tol(values, periods=PERIODS)


class TestDownsideDeviationContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestDownsideDeviationEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_periods_per_year_below_one_raises(self) -> None:
        """
        Verifies that ``periods_per_year < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="periods_per_year must be >= 1"):
            downside_deviation(pl.col(COLUMN_X), periods_per_year=0)

    def test_non_finite_threshold_raises(self) -> None:
        """
        Verifies that a non-finite ``threshold`` raises ``ValueError``.
        """
        for invalid in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="threshold must be a finite number"):
                downside_deviation(pl.col(COLUMN_X), periods_per_year=PERIODS, threshold=invalid)

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series resolves to its annualized shortfall (the population RMS is defined for one
        observation).
        """
        values = [-0.02]
        assert_matches(
            apply_expr(values, downside_deviation(pl.col(COLUMN_X), periods_per_year=PERIODS)),
            [downside_deviation_reference(values, PERIODS)],
            abs_tol=_abs_tol(values),
        )

    def test_no_downside_is_zero(self) -> None:
        """
        Verifies that returns all at or above the threshold have zero downside, so the deviation is ``0``.
        """
        result = apply_expr([0.01, 0.02, 0.0, 0.03], downside_deviation(pl.col(COLUMN_X), periods_per_year=PERIODS))
        assert_matches(result, [0.0])

    def test_null_skipped(self) -> None:
        """
        Verifies that ``null`` returns are skipped (excluded from the downside deviation), matching the reference.
        """
        values = [0.01, None, -0.02, 0.03, None]
        assert_matches(
            apply_expr(values, downside_deviation(pl.col(COLUMN_X), periods_per_year=PERIODS)),
            [downside_deviation_reference(values, PERIODS)],
            abs_tol=_abs_tol(values),
        )

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN return poisons the result to NaN.
        """
        values = [0.01, math.nan, -0.02]
        assert_matches(apply_expr(values, downside_deviation(pl.col(COLUMN_X), periods_per_year=PERIODS)), [math.nan])


class TestDownsideDeviationCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative return series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018]
        assert_matches(
            apply_expr(values, downside_deviation(pl.col(COLUMN_X), periods_per_year=PERIODS)),
            [downside_deviation_reference(values, PERIODS)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=_abs_tol(values),
        )

    def test_matches_reference_with_threshold(self) -> None:
        """
        Verifies agreement with the reference when a non-zero minimum acceptable return shifts the shortfall.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018]
        assert_matches(
            apply_expr(values, downside_deviation(pl.col(COLUMN_X), periods_per_year=PERIODS, threshold=0.01)),
            [downside_deviation_reference(values, PERIODS, threshold=0.01)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=_abs_tol(values),
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the RMS of the [0, -0.04, 0, -0.06, 0] shortfalls, annualized over 252 periods.
        """
        values = [0.02, -0.04, 0.01, -0.06, 0.03]
        assert_matches(
            apply_expr(values, downside_deviation(pl.col(COLUMN_X), periods_per_year=252).round(4)), [0.5119]
        )


class TestDownsideDeviationProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(subnormal_safe_floats(bound=1e3)))
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any return series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, downside_deviation(pl.col(COLUMN_X), periods_per_year=PERIODS)),
            [downside_deviation_reference(case, PERIODS)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=_abs_tol(case),
        )

    @given(case=_cases(missing_data_floats(min_magnitude=1e-3), min_size=0))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, downside_deviation(pl.col(COLUMN_X), periods_per_year=PERIODS)),
            [downside_deviation_reference(case, PERIODS)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=_abs_tol(case),
        )

    @given(case=_cases(subnormal_safe_floats(bound=1e3)), exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]))
    def test_scale_homogeneity(self, case: list[float], exponent: int) -> None:
        """
        Verifies degree-1 homogeneity at threshold 0: ``downside_deviation(k * r) == k * downside_deviation(r)`` for
        positive powers of two ``k``.
        """
        k = 2.0**exponent
        base = apply_expr(case, downside_deviation(pl.col(COLUMN_X), periods_per_year=PERIODS))
        scaled = apply_expr(
            [value * k for value in case], downside_deviation(pl.col(COLUMN_X), periods_per_year=PERIODS)
        )
        assert_scale_homogeneous(scaled, base, k=k, degree=1)

    @given(case=_cases(subnormal_safe_floats(bound=1e3)), scale=st.sampled_from([1e-6, 1e6, 1e9]))
    def test_matches_reference_at_large_magnitude(self, case: list[float], scale: float) -> None:
        """
        Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees.
        """
        scaled = [value * scale for value in case]
        assert_matches(
            apply_expr(scaled, downside_deviation(pl.col(COLUMN_X), periods_per_year=PERIODS)),
            [downside_deviation_reference(scaled, PERIODS)],
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(scaled) * math.sqrt(PERIODS) * STREAMING_TOLERANCE_FACTOR,
        )
