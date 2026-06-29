"""
Tests for ``pomata.metrics.volatility`` — the annualized sample standard deviation of a return series.

``volatility`` is single-input and REDUCING (a return series → one scalar), so tests use the shared ``apply_expr``
helper to materialize the factory over a one-column ``Float64`` frame and read the single output row; ``assert_matches``
and the naive ``volatility_reference`` oracle are shared across the suite. It is degree-1 homogeneous in the returns
(``volatility(k * r) == |k| * volatility(r)``), so it carries the scale-homogeneity and large-magnitude tiers.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (validation / empty / single-row / null / NaN / flat), correctness (vs the closed-form reference and a frozen
golden master), and properties (reference agreement incl. missing data, degree-1 scale-homogeneity, large-magnitude
stability). Categories are split into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import volatility_reference
from tests.support import (
    COLUMN_X,
    GROUP_KEY,
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

from pomata.metrics import volatility

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# metric's. volatility is windowless and REDUCING (M = 0); a case is just the return series. Its three facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  subnormal_safe_floats(bound): finite returns floored away from the subnormal-square underflow (the
#              squared deviation must stay representable); ``bound`` is widened in the scale / magnitude tiers
#   3. scale   degree-1 homogeneous in the returns -> scale-homogeneity + large-magnitude tiers (not scale-invariant)
# PERIODS is a representative annualization; the result annualizes by sqrt(PERIODS). The std is one-pass in Polars and
# two-pass in the oracle, so the sqrt amplifies relative error as variance -> 0: comparisons use a STREAMING-sized abs
# tolerance. Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
PERIODS = 252


@st.composite
def _cases[T](draw: st.DrawFn, returns: st.SearchStrategy[T], min_size: int = 2) -> list[T]:
    """
    A return series sized from the facts above. volatility is windowless, so a case is just the series; ``min_size``
    defaults to two (the sample standard deviation needs two observations to be defined).
    """
    return draw(st.lists(returns, min_size=min_size, max_size=SERIES_MAX))


def _abs_tol(values: Sequence[float | None]) -> float:
    """The magnitude-relative absolute tolerance for the annualized std (sized to the output's sqrt-of-time scale)."""
    return streaming_abs_tol(values, periods=PERIODS)


class TestVolatilityContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(volatility(pl.col(COLUMN_X), periods_per_year=PERIODS), pl.Expr)

    def test_reduces_to_scalar(self) -> None:
        """
        Verifies that the metric reduces a series to one ``Float64`` row.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.015, 0.005], dtype=pl.Float64)})
        result = frame.select(volatility(pl.col(COLUMN_X), periods_per_year=PERIODS).alias("v"))
        assert result.height == 1
        assert result.schema["v"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [0.01, -0.02, 0.015, 0.005], dtype=pl.Float64)})
        expr = volatility(pl.col(COLUMN_X), periods_per_year=PERIODS).alias("v")
        result_eager = frame.select(expr)
        result_lazy = frame.lazy().select(expr).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the volatility is computed per group (broadcast) and never spans boundaries.
        """
        group_a = [0.01, -0.02, 0.015, 0.005, -0.01]
        group_b = [0.02, 0.01, -0.03, 0.0, 0.01]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * 5 + ["b"] * 5, COLUMN_X: group_a + group_b})
        expr = volatility(pl.col(COLUMN_X), periods_per_year=PERIODS).over(GROUP_KEY)
        grouped = frame.select(expr.alias("v"))["v"].to_list()
        expected_a = volatility_reference(group_a, PERIODS)
        expected_b = volatility_reference(group_b, PERIODS)
        assert_matches(grouped, [expected_a] * 5 + [expected_b] * 5, abs_tol=_abs_tol(group_a + group_b))


class TestVolatilityEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_periods_per_year_below_one_raises(self) -> None:
        """
        Verifies that ``periods_per_year < 1`` raises ``ValueError``.
        """
        for invalid in (0, -1, -252):
            with pytest.raises(ValueError, match="periods_per_year must be >= 1"):
                volatility(pl.col(COLUMN_X), periods_per_year=invalid)

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields ``null`` (no observations).
        """
        assert_matches(apply_expr([], volatility(pl.col(COLUMN_X), periods_per_year=PERIODS)), [None])

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series yields ``null`` (the sample standard deviation needs two observations).
        """
        assert_matches(apply_expr([0.05], volatility(pl.col(COLUMN_X), periods_per_year=PERIODS)), [None])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields ``null``.
        """
        assert_matches(apply_expr([None, None, None], volatility(pl.col(COLUMN_X), periods_per_year=PERIODS)), [None])

    def test_null_skipped(self) -> None:
        """
        Verifies that ``null`` returns are skipped (excluded from the standard deviation), matching the reference.
        """
        values = [0.01, None, 0.02, 0.03, None]
        assert_matches(
            apply_expr(values, volatility(pl.col(COLUMN_X), periods_per_year=PERIODS)),
            [volatility_reference(values, PERIODS)],
            abs_tol=_abs_tol(values),
        )

    def test_nan_poisons(self) -> None:
        """
        Verifies that a ``NaN`` return propagates, yielding ``NaN``.
        """
        values = [0.01, math.nan, 0.02, 0.03]
        assert_matches(apply_expr(values, volatility(pl.col(COLUMN_X), periods_per_year=PERIODS)), [math.nan])

    def test_flat_returns_zero(self) -> None:
        """
        Verifies that a constant return series has zero dispersion, so the volatility is ``0``.
        """
        assert_matches(
            apply_expr([0.01, 0.01, 0.01, 0.01], volatility(pl.col(COLUMN_X), periods_per_year=PERIODS).round(10)),
            [0.0],
        )


class TestVolatilityCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative return series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018]
        assert_matches(
            apply_expr(values, volatility(pl.col(COLUMN_X), periods_per_year=PERIODS)),
            [volatility_reference(values, PERIODS)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=_abs_tol(values),
        )

    def test_golden_master(self) -> None:
        """
        Verifies frozen references: the un-annualized sample std of ``[0.1, -0.1, 0.2, -0.2]`` is ``sqrt(0.1 / 3)`` and
        its daily annualization scales that by ``sqrt(252)``.
        """
        values = [0.1, -0.1, 0.2, -0.2]
        assert_matches(apply_expr(values, volatility(pl.col(COLUMN_X), periods_per_year=1).round(4)), [0.1826])
        assert_matches(apply_expr(values, volatility(pl.col(COLUMN_X), periods_per_year=252).round(4)), [2.8983])


class TestVolatilityProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(subnormal_safe_floats(bound=1e3)))
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any return series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, volatility(pl.col(COLUMN_X), periods_per_year=PERIODS)),
            [volatility_reference(case, PERIODS)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=_abs_tol(case),
        )

    @given(case=_cases(missing_data_floats(min_magnitude=1e-3), min_size=0))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, volatility(pl.col(COLUMN_X), periods_per_year=PERIODS)),
            [volatility_reference(case, PERIODS)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=_abs_tol(case),
        )

    @given(case=_cases(subnormal_safe_floats(bound=1e3)), exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]))
    def test_scale_homogeneity(self, case: list[float], exponent: int) -> None:
        """
        Verifies degree-1 homogeneity: ``volatility(k * r) == k * volatility(r)`` for positive powers of two ``k``
        (a lossless rescaling that cannot perturb the dispersion).
        """
        k = 2.0**exponent
        base = apply_expr(case, volatility(pl.col(COLUMN_X), periods_per_year=PERIODS))
        scaled = apply_expr([value * k for value in case], volatility(pl.col(COLUMN_X), periods_per_year=PERIODS))
        assert_scale_homogeneous(scaled, base, k=k, degree=1)

    @given(case=_cases(subnormal_safe_floats(bound=1e3)), scale=st.sampled_from([1e-6, 1e6, 1e9]))
    def test_matches_reference_at_large_magnitude(self, case: list[float], scale: float) -> None:
        """
        Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees.
        """
        scaled = [value * scale for value in case]
        assert_matches(
            apply_expr(scaled, volatility(pl.col(COLUMN_X), periods_per_year=PERIODS)),
            [volatility_reference(scaled, PERIODS)],
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(scaled) * math.sqrt(PERIODS) * STREAMING_TOLERANCE_FACTOR,
        )
