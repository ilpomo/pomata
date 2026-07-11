"""
Tests for ``pomata.pnl.cost_fixed`` — the flat charge per trade (currency).

``cost_fixed`` is single-input (``quantity``) plus a scalar ``fee``; it charges ``fee`` on every bar where the position
changes (``turnover(quantity) > 0``) and ``0`` where it is held, inheriting the flat start and turnover's null / NaN
rule. Tests use the shared ``apply_expr`` helper; ``assert_matches`` and the naive ``cost_fixed_reference`` oracle are
shared. The cost is bounded by ``fee`` and depends only on WHETHER the quantity changed, not by how much, so it is
scale-INVARIANT in the quantity (it carries a scale-invariance property in place of the homogeneity / large-magnitude
tiers).

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` per-group independence), edge
(negative-fee guard / single-row / null / NaN / flat-start / consecutive-infinities), correctness (vs the closed-form
reference and a frozen golden master), and properties (reference agreement incl. missing data, scale-invariance).
Categories are split into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.pnl.oracles import cost_fixed_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    apply_expr,
    assert_matches,
    finite_floats,
    missing_data_floats,
    subnormal_safe_floats,
)

from pomata.pnl import cost_fixed

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- cost_fixed reads the turnover only to detect a trade (W = 0, flat start, M = 0); a case is just the
# quantity series plus a scalar fee. It is scale-INVARIANT in the quantity (scaling by a positive constant leaves the
# set of trade-bars unchanged), so it carries a scale-invariance property instead of the homogeneity / large-magnitude
# tiers. That invariance is the one tier sensitive to subnormal underflow: cost_fixed is DISCONTINUOUS (fee vs 0), so
# rescaling a subnormal-magnitude quantity DOWN to exactly 0.0 silently erases a trade and flips the fee -- a pure
# floating-point artifact, not a bug. The scale tier therefore draws from subnormal_safe_floats (magnitude floored at
# SUBNORMAL_FLOOR) so the lossless power-of-two rescaling can never underflow a nonzero quantity to zero; the continuous
# siblings (cost_per_share / cost_notional) absorb the same underflow within abs-tol and need no floor. The other tiers
# keep the full finite_floats / missing_data_floats domain (exact 0.0 included). Repetitions N are the shared CI profile
# (tests/conftest.py).
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
FEE = 1.0  # the deterministic-test flat fee per trade

_FEES = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)


@st.composite
def _cases[T](draw: st.DrawFn, quantities: st.SearchStrategy[T], min_size: int = 1) -> list[T]:
    """
    A quantity series; windowless with a flat start, so every row is a defined output.
    """
    return draw(st.lists(quantities, min_size=min_size, max_size=SERIES_MAX))


class TestCostFixedContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the turnover resets per group (each group gets its own flat start).
        """
        frame = pl.DataFrame({GROUP_KEY: ["a"] * 3 + ["b"] * 3, COLUMN_X: [10.0, 10.0, -5.0, 2.0, 2.0, 2.0]})
        expr = cost_fixed(pl.col(COLUMN_X), fee=FEE).over(GROUP_KEY)
        grouped = frame.select(expr.alias("y"))["y"].to_list()
        group_a = apply_expr([10.0, 10.0, -5.0], cost_fixed(pl.col(COLUMN_X), fee=FEE))
        group_b = apply_expr([2.0, 2.0, 2.0], cost_fixed(pl.col(COLUMN_X), fee=FEE))
        assert_matches(grouped, group_a + group_b)


class TestCostFixedEdge:
    """
    Boundaries, the flat start, null / NaN handling, and the fee guard.
    """

    def test_invalid_fee_raises(self) -> None:
        """
        Verifies that a fee that is not a finite number ``>= 0`` (negative, ``NaN``, or ``±inf``) raises
        ``ValueError`` -- a charge is a finite non-negative number, so a non-finite value fails fast at the call site
        rather than silently poisoning the output with ``NaN`` / ``inf``.
        """
        for invalid in (-1.0, math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="fee must be a finite number >= 0"):
                cost_fixed(pl.col(COLUMN_X), fee=invalid)

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series charges the fee (the entry trade), not null.
        """
        assert_matches(apply_expr([10.0], cost_fixed(pl.col(COLUMN_X), fee=FEE)), [1.0])

    def test_null_propagates(self) -> None:
        """
        Verifies that a null voids its own row and the next (via turnover), matching the naive reference.
        """
        values = [10.0, None, -5.0, 20.0]
        assert_matches(apply_expr(values, cost_fixed(pl.col(COLUMN_X), fee=FEE)), cost_fixed_reference(values, FEE))

    def test_null_takes_precedence_over_nan(self) -> None:
        """
        Verifies that the traded row where a ``NaN`` quantity meets the previous row's ``null`` yields ``null``
        (``null`` takes precedence over ``NaN``), while the next trade off the ``NaN`` is ``NaN``.
        """
        assert_matches(
            apply_expr([10.0, None, math.nan, 20.0], cost_fixed(pl.col(COLUMN_X), fee=FEE)),
            [1.0, None, None, math.nan],
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a NaN propagates to its own row and the next (matching the naive reference).
        """
        values = [10.0, math.nan, -5.0, 20.0]
        assert_matches(apply_expr(values, cost_fixed(pl.col(COLUMN_X), fee=FEE)), cost_fixed_reference(values, FEE))

    def test_flat_start_first_row(self) -> None:
        """
        Verifies the first row charges the fee (the entry trade from a flat start), then the held bar is zero.
        """
        assert_matches(apply_expr([10.0, 10.0, -5.0], cost_fixed(pl.col(COLUMN_X), fee=FEE)), [1.0, 0.0, 1.0])

    def test_consecutive_infinities_make_nan(self) -> None:
        """
        Verifies the threshold masking against the reference on infinite quantities: a finite-to-``inf`` move is a trade
        (charges the fee), while two consecutive equal-sign infinities make the turnover ``inf - inf = NaN`` and the
        masked fee ``NaN``. The property tiers cannot reach this (their strategies set ``allow_infinity=False``).
        """
        values = [math.inf, math.inf, 1.0, -math.inf]
        assert_matches(apply_expr(values, cost_fixed(pl.col(COLUMN_X), fee=FEE)), cost_fixed_reference(values, FEE))


class TestCostFixedCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative quantity series.
        """
        values = [10.0, 10.0, -5.0, -5.0, 20.0, 15.0, -8.0, 12.0]
        assert_matches(
            apply_expr(values, cost_fixed(pl.col(COLUMN_X), fee=FEE)),
            cost_fixed_reference(values, FEE),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference over a five-bar quantity series at a flat fee of one.
        """
        result = apply_expr([10.0, 10.0, -5.0, -5.0, 20.0], cost_fixed(pl.col(COLUMN_X), fee=FEE).round(4))
        assert_matches(result, [1.0, 0.0, 1.0, 0.0, 1.0])


class TestCostFixedProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(finite_floats(), min_size=0), fee=_FEES)
    def test_matches_reference_for_any_input(
        self,
        case: list[float],
        fee: float,
    ) -> None:
        """
        Verifies that, for any quantity series and non-negative fee, the implementation matches the naive reference.
        """
        values = case
        assert_matches(
            apply_expr(values, cost_fixed(pl.col(COLUMN_X), fee=fee)),
            cost_fixed_reference(values, fee),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(missing_data_floats(), min_size=0), fee=_FEES)
    def test_matches_reference_under_missing_data(
        self,
        case: list[float | None],
        fee: float,
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values = case
        assert_matches(
            apply_expr(values, cost_fixed(pl.col(COLUMN_X), fee=fee)),
            cost_fixed_reference(values, fee),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(subnormal_safe_floats()), exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]))
    def test_scale_invariance(
        self,
        case: list[float],
        exponent: int,
    ) -> None:
        """
        Verifies that ``cost_fixed`` is scale-invariant: scaling every input value by a constant ``k`` leaves the
        output unchanged -- ``cost_fixed(k * x) == cost_fixed(x)``. ``k`` is a power of two, so the rescale is exact
        and adds no floating-point error.
        """
        k = 2.0**exponent
        values = case
        result_base = apply_expr(values, cost_fixed(pl.col(COLUMN_X), fee=FEE))
        result_scaled = apply_expr([value * k for value in values], cost_fixed(pl.col(COLUMN_X), fee=FEE))
        assert_matches(
            result_scaled, result_base, rel_tol=RELATIVE_TOLERANCE_PROPERTY, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE
        )
