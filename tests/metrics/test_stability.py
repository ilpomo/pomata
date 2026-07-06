"""
Tests for ``pomata.metrics.stability`` — the R-squared of the cumulative log-return path against time.

``stability`` is single-input and REDUCING (a return series → one scalar), so tests read the single output row of
``apply_expr``; ``assert_matches`` and the naive ``stability_reference`` oracle (the squared correlation of the
cumulative log returns with the time index) are shared across the suite. It is a coefficient of determination, so its
defining property is that it lies in ``[0, 1]``.

The ladder is the canonical one: contract (type / reduces-to-scalar / lazy-eager / ``.over`` per-group independence),
edge (empty / single-row / constant / flat / out-of-domain / null / NaN), correctness (vs the closed-form reference and
a frozen golden master), and properties (reference agreement incl. missing data, the unit-interval bound). Categories
are split into classes; cross-cutting categories use markers.
"""

import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from tests.metrics.oracles import stability_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    apply_expr,
    assert_matches,
)

from pomata.metrics import stability

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- stability is windowless and REDUCING (M = 0); a case is just a return series. Facts:
#   1. shape   reducing: the output is one scalar (one row in ``select``; one per group under ``.over``)
#   2. domain  positive returns so the cumulative log is monotone and well-conditioned; the degenerate flat path and the
#              out-of-domain (<= -1) cases are pinned in the edge tier; the missing variant mixes null / NaN
#   3. scale   a coefficient of determination -> the defining property is the unit-interval bound [0, 1]
# Repetitions N are the shared CI profile (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
_RETURNS = st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False)
_RETURNS_MISSING = st.one_of(st.none(), st.just(math.nan), _RETURNS)


@st.composite
def _cases[T](draw: st.DrawFn, returns: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """A return series sized from the facts above."""
    return draw(st.lists(returns, min_size=min_size, max_size=SERIES_MAX))


class TestStabilityContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestStabilityEdge:
    """
    Boundaries and null / NaN handling.
    """

    def test_single_row_is_null(self) -> None:
        """
        Verifies that a one-element series yields ``null`` (the regression needs at least two points).
        """
        assert_matches(apply_expr([0.01], stability(pl.col(COLUMN_X))), [None])

    def test_constant_is_one(self) -> None:
        """
        Verifies that a constant non-zero series has a perfectly linear cumulative log, so the R-squared is ``1``.
        """
        assert_matches(apply_expr([0.01, 0.01, 0.01, 0.01], stability(pl.col(COLUMN_X))), [1.0])

    def test_flat_path_is_nan(self) -> None:
        """
        Verifies that an all-zero series has a flat (zero-variance) cumulative log, so the R-squared is ``NaN``.
        """
        assert_matches(apply_expr([0.0, 0.0, 0.0], stability(pl.col(COLUMN_X))), [math.nan])

    def test_out_of_domain_is_nan(self) -> None:
        """
        Verifies that a return at or below ``-1`` makes the cumulative log undefined, so the result is ``NaN``.
        """
        assert_matches(apply_expr([0.02, -1.5, 0.01], stability(pl.col(COLUMN_X))), [math.nan])

    def test_nan_poisons(self) -> None:
        """
        Verifies that a NaN return poisons the result to NaN.
        """
        assert_matches(apply_expr([0.01, math.nan, 0.02, 0.03], stability(pl.col(COLUMN_X))), [math.nan])

    def test_null_skipped(self) -> None:
        """
        Verifies that null returns are skipped and the time index runs over the retained observations.
        """
        values = [0.01, None, 0.012, 0.009, None, 0.011, 0.013]
        assert_matches(
            apply_expr(values, stability(pl.col(COLUMN_X))),
            [stability_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )


class TestStabilityCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative return series.
        """
        values = [0.012, -0.008, 0.02, 0.005, 0.015, -0.002, 0.018, 0.01]
        assert_matches(
            apply_expr(values, stability(pl.col(COLUMN_X))),
            [stability_reference(values)],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: a steadily compounding series tracks a line very closely, R-squared 0.9984.
        """
        values = [0.01, 0.012, 0.009, 0.011, 0.013, 0.008, 0.01, 0.012]
        assert_matches(apply_expr(values, stability(pl.col(COLUMN_X)).round(4)), [0.9984])


class TestStabilityProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_RETURNS))
    def test_matches_reference_for_any_input(self, case: list[float]) -> None:
        """
        Verifies that, for any in-domain return series, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, stability(pl.col(COLUMN_X))),
            [stability_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_RETURNS_MISSING, min_size=0))
    def test_matches_reference_under_missing_data(self, case: list[float | None]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        assert_matches(
            apply_expr(case, stability(pl.col(COLUMN_X))),
            [stability_reference(case)],
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_RETURNS, min_size=2))
    def test_within_unit_interval(self, case: list[float]) -> None:
        """
        Verifies that a defined result is a coefficient of determination in ``[0, 1]``.
        """
        result = apply_expr(case, stability(pl.col(COLUMN_X)))[0]
        if result is not None and not math.isnan(result):
            assert -ABSOLUTE_TOLERANCE_REFERENCE <= result <= 1.0 + ABSOLUTE_TOLERANCE_REFERENCE
