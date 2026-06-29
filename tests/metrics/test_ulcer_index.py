"""
Tests for ``pomata.metrics.ulcer_index`` — the root-mean-square depth of an equity curve's drawdowns.

``ulcer_index`` is single-input and REDUCING (an equity series → one scalar), so tests read the single output row of
``apply_expr``; ``assert_matches`` and the naive ``ulcer_index_reference`` oracle are shared across the suite. It is
invariant under a positive rescaling of the equity, so it carries a scale-invariance tier.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (empty / single-row / null / NaN poison / monotonic), correctness (vs the closed-form reference and a frozen
golden master), and properties (reference agreement incl. missing data, scale invariance). Categories are split into
classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.metrics.oracles import ulcer_index_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    apply_expr,
    assert_matches,
)

from pomata.metrics import ulcer_index

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- ulcer_index is windowless and REDUCING (M = 0); a case is just a positive equity series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  positive equities (a growth factor is > 0); the missing variant mixes null / NaN
#   3. scale   invariant under a positive rescale (the peak ratio cancels) -> scale-invariance tier
# Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
_EQUITY = st.floats(min_value=1e-2, max_value=1e4, allow_nan=False, allow_infinity=False)
_EQUITY_MISSING = st.one_of(st.none(), st.just(math.nan), _EQUITY)


@st.composite
def _cases[T](draw: st.DrawFn, equities: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A positive equity series sized from the facts above."""
    return draw(st.lists(equities, min_size=min_size, max_size=SERIES_MAX))


class TestUlcerIndexContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(ulcer_index(pl.col(COLUMN_X)), pl.Expr)

    def test_reduces_to_scalar(self) -> None:
        """
        Verifies that the metric reduces a series to one ``Float64`` row.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 1.1, 1.05, 1.2], dtype=pl.Float64)})
        result = frame.select(ulcer_index(pl.col(COLUMN_X)).alias("u"))
        assert result.height == 1
        assert result.schema["u"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 1.1, 1.05, 1.2], dtype=pl.Float64)})
        expr = ulcer_index(pl.col(COLUMN_X)).alias("u")
        assert_frame_equal(frame.select(expr), frame.lazy().select(expr).collect())

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the index is computed per group (broadcast) and never spans boundaries.
        """
        group_a = [1.0, 1.1, 1.05, 1.2, 0.9, 1.0]
        group_b = [2.0, 1.8, 2.2, 2.0]
        frame = pl.DataFrame({GROUP_KEY: ["a"] * len(group_a) + ["b"] * len(group_b), COLUMN_X: group_a + group_b})
        grouped = frame.select(ulcer_index(pl.col(COLUMN_X)).over(GROUP_KEY).alias("u"))["u"].to_list()
        expected_a = ulcer_index_reference(group_a)
        expected_b = ulcer_index_reference(group_b)
        assert_matches(grouped, [expected_a] * len(group_a) + [expected_b] * len(group_b))


class TestUlcerIndexEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields ``null``.
        """
        assert_matches(apply_expr([], ulcer_index(pl.col(COLUMN_X))), [None])

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series has no drawdown, so the Ulcer Index is ``0``.
        """
        assert_matches(apply_expr([1.0], ulcer_index(pl.col(COLUMN_X))), [0.0])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields ``null``.
        """
        assert_matches(apply_expr([None, None], ulcer_index(pl.col(COLUMN_X))), [None])

    def test_monotonic_rise_is_zero(self) -> None:
        """
        Verifies that a never-declining curve has all-zero drawdowns, so the Ulcer Index is ``0``.
        """
        assert_matches(apply_expr([1.0, 1.1, 1.2, 1.3], ulcer_index(pl.col(COLUMN_X))), [0.0])

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN equity poisons the result to NaN.
        """
        assert_matches(apply_expr([1.0, 1.1, math.nan, 0.9, 1.2], ulcer_index(pl.col(COLUMN_X))), [math.nan])

    def test_null_skipped(self) -> None:
        """
        Verifies that null equities are skipped (excluded from the mean), matching the reference.
        """
        values = [1.0, None, 1.2, 0.9, None, 1.1]
        assert_matches(apply_expr(values, ulcer_index(pl.col(COLUMN_X))), [ulcer_index_reference(values)])


class TestUlcerIndexCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative equity curve.
        """
        values = [1.0, 1.05, 1.2, 1.1, 1.3, 0.95, 1.0, 1.4]
        assert_matches(
            apply_expr(values, ulcer_index(pl.col(COLUMN_X))),
            [ulcer_index_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference over a six-bar equity curve (the RMS of its [0, 0, -0.0455, 0, -0.25, -0.1667]
        drawdowns).
        """
        result = apply_expr([1.0, 1.1, 1.05, 1.2, 0.9, 1.0], ulcer_index(pl.col(COLUMN_X)).round(4))
        assert_matches(result, [0.1241])


class TestUlcerIndexProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_EQUITY))
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any positive equity series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, ulcer_index(pl.col(COLUMN_X))),
            [ulcer_index_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_EQUITY_MISSING, min_size=0))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, ulcer_index(pl.col(COLUMN_X))),
            [ulcer_index_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_EQUITY), exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]))
    def test_scale_invariance(self, case: list[float], exponent: int) -> None:
        """
        Verifies that a positive rescale of the equity leaves the Ulcer Index unchanged (powers of two, lossless).
        """
        k = 2.0**exponent
        base = apply_expr(case, ulcer_index(pl.col(COLUMN_X)))
        scaled = apply_expr([value * k for value in case], ulcer_index(pl.col(COLUMN_X)))
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_PROPERTY, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
