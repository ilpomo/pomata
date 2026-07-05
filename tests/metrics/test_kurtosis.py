"""
Tests for ``pomata.metrics.kurtosis`` — the excess (Fisher) kurtosis of a return series.

``kurtosis`` is single-input and REDUCING (a return series → one scalar), so tests read the single output row of
``apply_expr``; ``assert_matches`` and the naive ``kurtosis_reference`` oracle (the population moments) are shared
across the suite. It is scale-invariant (a standardized moment), so it carries a scale-invariance tier; being a ratio
of moments it is conditioning-sensitive, so the property comparisons use the scale band.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (empty / single-row / constant / null / NaN), correctness (vs the closed-form reference and a frozen golden
master), and properties (reference agreement incl. missing data, scale invariance). Categories are split into classes.
"""

import math

import polars as pl
from hypothesis import assume, given
from hypothesis import strategies as st
from tests.metrics.oracles import kurtosis_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    STANDARDIZED_MOMENT_FLOOR,
    apply_expr,
    assert_matches,
    missing_data_floats,
    standardized_moment_floats,
    well_spread,
)

from pomata.metrics import kurtosis

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- kurtosis is windowless and REDUCING (M = 0); a case is just a return series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  standardized_moment_floats: spread returns floored off subnormal (m2**2 underflows there); +null/NaN
#   3. scale   invariant (a standardized moment) -> scale-invariance tier; conditioning-sensitive -> the scale band
# Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50


@st.composite
def _cases[T](draw: st.DrawFn, returns: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A return series sized from the facts above."""
    return draw(st.lists(returns, min_size=min_size, max_size=SERIES_MAX))


class TestKurtosisContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestKurtosisEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_single_row_is_nan(self) -> None:
        """
        Verifies that a one-element series has zero variance, so the standardized moment is NaN.
        """
        assert_matches(apply_expr([0.05], kurtosis(pl.col(COLUMN_X))), [math.nan])

    def test_constant_is_nan(self) -> None:
        """
        Verifies that a constant series has zero variance, so the kurtosis is NaN.
        """
        assert_matches(apply_expr([0.01, 0.01, 0.01], kurtosis(pl.col(COLUMN_X))), [math.nan])

    def test_subnormal_magnitude_is_nan(self) -> None:
        """
        Verifies that a subnormal-magnitude series, whose ``m2 ** 2`` underflows to zero, yields NaN -- the degenerate
        the property tier floors away from, pinned deterministically here.
        """
        assert_matches(apply_expr([0.0, 1e-160, 2e-160], kurtosis(pl.col(COLUMN_X))), [math.nan])

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN return poisons the result to NaN.
        """
        assert_matches(apply_expr([0.01, math.nan, 0.02, 0.03], kurtosis(pl.col(COLUMN_X))), [math.nan])

    def test_null_skipped(self) -> None:
        """
        Verifies that null returns are skipped, matching the reference.
        """
        values = [0.01, None, 0.02, -0.03, 0.04, None, -0.01]
        assert_matches(
            apply_expr(values, kurtosis(pl.col(COLUMN_X))),
            [kurtosis_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_SCALE,
        )


class TestKurtosisCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative return series.
        """
        values = [0.01, -0.02, 0.015, -0.03, 0.005, -0.01, 0.02]
        assert_matches(
            apply_expr(values, kurtosis(pl.col(COLUMN_X))),
            [kurtosis_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: a short, spread-out return series is platykurtic (negative excess kurtosis).
        """
        values = [0.01, -0.02, 0.015, -0.03, 0.005, -0.01, 0.02]
        assert_matches(apply_expr(values, kurtosis(pl.col(COLUMN_X)).round(4)), [-1.3223])


class TestKurtosisProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(standardized_moment_floats(bound=1e3)))
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any well-conditioned return series, the implementation matches the naive reference.
        """
        assume(well_spread(case))
        assert_matches(
            apply_expr(case, kurtosis(pl.col(COLUMN_X))),
            [kurtosis_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(missing_data_floats(min_magnitude=STANDARDIZED_MOMENT_FLOOR), min_size=0))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for well-conditioned inputs freely mixing null / NaN / finite, the implementation matches the
        naive reference.
        """
        assume(well_spread(case))
        assert_matches(
            apply_expr(case, kurtosis(pl.col(COLUMN_X))),
            [kurtosis_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(standardized_moment_floats(bound=1e3), min_size=2),
        exponent=st.sampled_from([-4, -2, -1, 1, 2, 4]),
    )
    def test_scale_invariance(self, case: list[float], exponent: int) -> None:
        """
        Verifies that a positive rescale of the returns leaves the kurtosis unchanged (a standardized moment), using
        powers of two so the rescaling is lossless.
        """
        k = 2.0**exponent
        base = apply_expr(case, kurtosis(pl.col(COLUMN_X)))
        scaled = apply_expr([value * k for value in case], kurtosis(pl.col(COLUMN_X)))
        assert_matches(scaled, base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
