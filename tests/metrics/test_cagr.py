"""
Tests for ``pomata.metrics.cagr`` — the compound annual growth rate of an equity curve.

``cagr`` is single-input and REDUCING (an equity series → one scalar), so tests read the single output row of
``apply_expr``; ``assert_matches`` and the naive ``cagr_reference`` oracle are shared across the suite. It reads a
growth-factor series normalized to a unit start and annualizes its total growth, so it is neither scale-homogeneous nor
scale-invariant — its correctness is pinned by the reference, a golden master, and the metamorphic identity linking it
to :func:`total_return`.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (validation / empty / single-row / null / NaN poison), correctness (vs the closed-form reference and a frozen
golden master), and properties (reference agreement incl. missing data, the cagr/total-return identity). Categories are
split into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import cagr_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    apply_expr,
    assert_matches,
)

from pomata.metrics import cagr, total_return

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- cagr is windowless and REDUCING (M = 0); a case is just a positive equity series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  modest positive growth factors in [0.1, 10] so the annualizing power stays finite (a final ** (P / N)
#              with a tiny N and a large P overflows for an unbounded equity); the missing variant mixes null / NaN
#   3. scale   neither (a normalized growth factor) -> no scale tier; pinned by the reference + golden + the identity
# PERIODS is drawn from a realistic set of annualization factors. Repetitions N are the shared CI profile.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
PERIODS = 252
_EQUITY = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
_EQUITY_MISSING = st.one_of(st.none(), st.just(math.nan), _EQUITY)
_PERIODS = st.sampled_from([1, 4, 12, 52, 252])


@st.composite
def _cases[T](draw: st.DrawFn, equities: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A positive equity series sized from the facts above."""
    return draw(st.lists(equities, min_size=min_size, max_size=SERIES_MAX))


class TestCagrContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(cagr(pl.col(COLUMN_X), periods_per_year=PERIODS), pl.Expr)

    def test_reduces_to_scalar(self) -> None:
        """
        Verifies that the metric reduces a series to one ``Float64`` row.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.1, 1.21, 1.3], dtype=pl.Float64)})
        result = frame.select(cagr(pl.col(COLUMN_X), periods_per_year=PERIODS).alias("c"))
        assert result.height == 1
        assert result.schema["c"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.1, 1.21, 1.3], dtype=pl.Float64)})
        expr = cagr(pl.col(COLUMN_X), periods_per_year=PERIODS).alias("c")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the rate is computed per group (broadcast) and never spans boundaries.
        """
        group_a = [1.1, 1.21, 1.331, 1.4641]
        group_b = [1.0, 1.1, 1.2]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * len(group_a) + ["b"] * len(group_b), COLUMN_X: group_a + group_b})
        grouped = frame.select(cagr(pl.col(COLUMN_X), periods_per_year=4).over(GROUP_KEY).alias("c"))["c"].to_list()
        expected_a = cagr_reference(group_a, 4)
        expected_b = cagr_reference(group_b, 4)
        assert_matches(grouped, [expected_a] * len(group_a) + [expected_b] * len(group_b))


class TestCagrEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_periods_per_year_below_one_raises(self) -> None:
        """
        Verifies that ``periods_per_year < 1`` raises ``ValueError``.
        """
        for invalid in (0, -1):
            with pytest.raises(ValueError, match="periods_per_year must be >= 1"):
                cagr(pl.col(COLUMN_X), periods_per_year=invalid)

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields ``null``.
        """
        assert_matches(apply_expr([], cagr(pl.col(COLUMN_X), periods_per_year=PERIODS)), [None])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields ``null``.
        """
        assert_matches(apply_expr([None, None], cagr(pl.col(COLUMN_X), periods_per_year=PERIODS)), [None])

    def test_leading_null_uses_last_defined(self) -> None:
        """
        Verifies that leading warm-up nulls are skipped; the result uses the last defined equity.
        """
        values = [None, 1.1, 1.21]
        assert_matches(
            apply_expr(values, cagr(pl.col(COLUMN_X), periods_per_year=PERIODS)), [cagr_reference(values, PERIODS)]
        )

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN equity poisons the result to NaN.
        """
        assert_matches(apply_expr([1.1, math.nan, 1.2], cagr(pl.col(COLUMN_X), periods_per_year=PERIODS)), [math.nan])

    def test_single_period_annualizes(self) -> None:
        """
        Verifies that a single observation annualizes its growth over one period (``final ** periods_per_year - 1``).
        """
        values = [1.01]
        assert_matches(
            apply_expr(values, cagr(pl.col(COLUMN_X), periods_per_year=4)),
            [cagr_reference(values, 4)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )


class TestCagrCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative equity curve.
        """
        values = [1.05, 1.1, 1.08, 1.15, 1.2, 1.18, 1.25]
        assert_matches(
            apply_expr(values, cagr(pl.col(COLUMN_X), periods_per_year=PERIODS)),
            [cagr_reference(values, PERIODS)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: two years of annual equity (1.1, 1.21) grow at a constant 10% per year.
        """
        result = apply_expr([1.1, 1.21], cagr(pl.col(COLUMN_X), periods_per_year=1).round(4))
        assert_matches(result, [0.1])


class TestCagrProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_EQUITY), periods=_PERIODS)
    def test_matches_reference_for_any_input(self, case: list[float], periods: int) -> None:
        """
        Verifies that, for any positive equity series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, cagr(pl.col(COLUMN_X), periods_per_year=periods)),
            [cagr_reference(case, periods)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_EQUITY_MISSING, min_size=0), periods=_PERIODS)
    def test_matches_reference_under_missing_data(self, case: list[float | None], periods: int) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, cagr(pl.col(COLUMN_X), periods_per_year=periods)),
            [cagr_reference(case, periods)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_EQUITY))
    def test_cagr_equals_total_return_over_one_year(self, case: list[float]) -> None:
        """
        Verifies the metamorphic relation to :func:`total_return`: annualizing over a span of exactly one year
        (``periods_per_year`` equal to the number of observations) makes the compound annual growth rate equal the
        total return. Using ``periods_per_year == N`` keeps the annualizing exponent at one, so the check is numerically
        stable even for large losses.
        """
        cagr_value = apply_expr(case, cagr(pl.col(COLUMN_X), periods_per_year=len(case)))[0]
        total = apply_expr(case, total_return(pl.col(COLUMN_X)))[0]
        assert cagr_value is not None
        assert total is not None
        assert math.isclose(
            cagr_value, total, rel_tol=RELATIVE_TOLERANCE_PROPERTY, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE
        )
