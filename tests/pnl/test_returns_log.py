"""
Tests for ``pomata.pnl.returns_log`` — Logarithmic (continuously-compounded) Returns.

``returns_log`` is single-input and a fixed one-bar-lag transform (a warm-up of one row, no recursion), so tests use the
shared ``apply_expr`` helper to materialize the factory over a one-column ``Float64`` frame; ``assert_matches`` and the
naive ``returns_log_reference`` oracle are shared across the suite. The return is a log-ratio, so it is scale-INVARIANT
(``returns_log(k * P) == returns_log(P)``): it carries a scale-invariance property in place of the scale-homogeneity /
large-magnitude tests, which are vacuous when the input scale cancels. It also carries the across-time aggregation
metamorphic — the sum of the log returns equals the total log return — the property that distinguishes it from the
arithmetic sibling.

The ladder is the canonical one: contract (type / shape / lazy-eager / ``.over`` per-group independence), edge
(warm-up / single-row / null / NaN / domain boundaries), correctness (vs the closed-form reference and a frozen golden
master), and properties (reference agreement incl. missing data, scale-invariance, across-time aggregation). Categories
are split into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.pnl.oracles import returns_log_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    apply_expr,
    assert_matches,
    positive_missing_data,
)

from pomata.pnl import returns_log

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# primitive's. To add a primitive, set its facts here; the property tier below is then the same shape as every other's.
#   1. warmup  W = 1   (the one-bar lag ``expr.shift(1)`` is undefined on row 0, so the first row is always null)
#   2. memory  the oracle is a per-row two-endpoint transform like pomata, so the property holds from the first defined
#              row (M = 0); a case is simply a series of prices -- every row past the first is a defined output
#   3. domain  positive prices in ``[1.0, PRICE_MAX]`` -- log returns are defined on a strictly positive series; the
#              zero / negative relative and the zero previous price are pinned deterministically in the edge tier
# returns_log has no window parameter, so ``_cases`` draws only the series (no window to couple). The return is a
# log-ratio (scale-invariant, O(1) around zero), so there is no scale-homogeneity or large-magnitude VALUE test (the
# scale cancels); a scale-invariance metamorphic stands in its place. Repetitions N are the shared CI profile
# (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50
PRICE_MAX = 1e4

# The shared positive-price element strategy for the property tiers (log returns are defined on positive prices).
_POSITIVE_PRICES = st.floats(min_value=1.0, max_value=PRICE_MAX, allow_nan=False, allow_infinity=False)


@st.composite
def _cases[T](draw: st.DrawFn, prices: st.SearchStrategy[T], min_size: int = 2) -> list[T]:
    """
    A price series sized from the facts above. returns_log is windowless with a one-row warm-up, so -- unlike the
    windowed indicators' ``(series, window)`` pair -- a case is just the series; ``min_size`` defaults to two so at
    least one defined return is produced.
    """
    return draw(st.lists(prices, min_size=min_size, max_size=SERIES_MAX))


class TestReturnsLogContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(returns_log(pl.col(COLUMN_X)), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [100.0, 105.0, 102.0, 108.0, 110.0], dtype=pl.Float64)})
        result = frame.select(returns_log(pl.col(COLUMN_X)).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [100.0, 105.0, 102.0, 108.0, 110.0], dtype=pl.Float64)})
        expr = returns_log(pl.col(COLUMN_X)).alias("y")
        result_eager = frame.select(expr)
        result_lazy = frame.lazy().select(expr).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the one-bar lag resets per group and never reaches across group boundaries (so the
        first row of each group is null).
        """
        frame = pl.DataFrame(
            {GROUP_KEY: ["a"] * 4 + ["b"] * 4, COLUMN_X: [100.0, 105.0, 102.0, 108.0, 50.0, 52.0, 51.0, 55.0]}
        )
        expr = returns_log(pl.col(COLUMN_X)).over(GROUP_KEY)
        grouped = frame.select(expr.alias("y"))["y"].to_list()
        group_a = apply_expr([100.0, 105.0, 102.0, 108.0], returns_log(pl.col(COLUMN_X)))
        group_b = apply_expr([50.0, 52.0, 51.0, 55.0], returns_log(pl.col(COLUMN_X)))
        assert_matches(grouped, group_a + group_b)


class TestReturnsLogEdge:
    """
    Boundaries, warm-up, null / NaN handling, and the positive-price domain.
    """

    def test_warmup_null_count(self) -> None:
        """
        Verifies the warm-up is exactly one row: the first return is null, the second is defined.
        """
        result = apply_expr([100.0, 105.0, 102.0], returns_log(pl.col(COLUMN_X)))
        assert result[0] is None
        assert result[1] is not None

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series is all warm-up (no previous price to take the log-ratio against).
        """
        assert_matches(apply_expr([42.0], returns_log(pl.col(COLUMN_X))), [None])

    def test_empty(self) -> None:
        """
        Verifies that an empty series yields an empty result.
        """
        assert_matches(apply_expr([], returns_log(pl.col(COLUMN_X))), [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series stays null (every lag references a null).
        """
        assert_matches(apply_expr([None, None, None], returns_log(pl.col(COLUMN_X))), [None, None, None])

    def test_null_propagates(self) -> None:
        """
        Verifies that a null at the current or previous row yields null there (matching the naive reference).
        """
        values = [100.0, 105.0, None, 108.0, 110.0]
        assert_matches(
            apply_expr(values, returns_log(pl.col(COLUMN_X))),
            returns_log_reference(values),
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a NaN propagates to the positions that reference it (matching the naive reference).
        """
        values = [100.0, 105.0, math.nan, 108.0, 110.0]
        assert_matches(
            apply_expr(values, returns_log(pl.col(COLUMN_X))),
            returns_log_reference(values),
        )

    def test_domain_boundaries(self) -> None:
        """
        Verifies the IEEE-754 logarithm boundaries reproduced from Polars: a negative relative (prices straddle zero) is
        ``NaN``, a zero relative (a zero price over a positive one) is ``-inf``, and a zero previous price (a positive
        price over zero) makes the relative ``+inf`` so the log is ``+inf``.
        """
        assert_matches(
            apply_expr([10.0, -5.0, 0.0, 5.0], returns_log(pl.col(COLUMN_X))),
            [None, math.nan, -math.inf, math.inf],
        )

    def test_negative_zero_previous_price(self) -> None:
        """
        Verifies the signed-zero sign branch: over a ``-0.0`` previous price the price relative flips sign, so a
        positive price gives a negative relative (``NaN`` after the log) and a negative price a positive relative
        (``+inf``) -- the mirror of a ``+0.0`` previous price. The property tiers draw prices from ``[1.0, PRICE_MAX]``
        and cannot reach a zero, so this deterministic pin -- asserting both the implementation AND the oracle against
        the literal -- is what protects the oracle's ``copysign`` sign factor.
        """
        cases: list[tuple[list[float], list[float | None]]] = [
            ([-0.0, 5.0], [None, math.nan]),
            ([-0.0, -5.0], [None, math.inf]),
        ]
        for values, expected in cases:
            assert_matches(apply_expr(values, returns_log(pl.col(COLUMN_X))), expected)
            assert_matches(returns_log_reference(values), expected)


class TestReturnsLogCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative positive-price series.
        """
        values = [100.0, 102.0, 101.5, 105.0, 103.0, 108.0, 110.0, 109.0]
        assert_matches(
            apply_expr(values, returns_log(pl.col(COLUMN_X))),
            returns_log_reference(values),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference over a five-bar price series.
        """
        result = apply_expr([100.0, 105.0, 102.0, 108.0, 110.0], returns_log(pl.col(COLUMN_X)).round(4))
        assert_matches(result, [None, 0.0488, -0.029, 0.0572, 0.0183])


class TestReturnsLogProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_POSITIVE_PRICES, min_size=0))
    def test_matches_reference_for_any_input(
        self,
        case: list[float],
    ) -> None:
        """
        Verifies that, for any positive-price series, the implementation matches the naive reference.
        """
        values = case
        assert_matches(
            apply_expr(values, returns_log(pl.col(COLUMN_X))),
            returns_log_reference(values),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(positive_missing_data(PRICE_MAX), min_size=0))
    def test_matches_reference_under_missing_data(
        self,
        case: list[float | None],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite positive prices, the implementation matches the
        naive reference.
        """
        values = case
        assert_matches(
            apply_expr(values, returns_log(pl.col(COLUMN_X))),
            returns_log_reference(values),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(
        case=_cases(_POSITIVE_PRICES),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_invariant(
        self,
        case: list[float],
        exponent: int,
    ) -> None:
        """
        Verifies that the return is scale-invariant: rescaling the whole price series by a constant leaves every return
        unchanged (``returns_log(k * P) == returns_log(P)``), because the scale cancels in the log-ratio. ``k`` is a
        power of two so the rescaling is lossless and cannot introduce a floating-point artifact.
        """
        k = 2.0**exponent
        values = case
        result_base = apply_expr(values, returns_log(pl.col(COLUMN_X)))
        result_scaled = apply_expr([value * k for value in values], returns_log(pl.col(COLUMN_X)))
        assert_matches(
            result_scaled, result_base, rel_tol=RELATIVE_TOLERANCE_PROPERTY, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE
        )

    @given(case=_cases(st.floats(min_value=1.0, max_value=PRICE_MAX, allow_nan=False, allow_infinity=False)))
    def test_aggregates_across_time(
        self,
        case: list[float],
    ) -> None:
        """
        Verifies the defining property of log returns -- they aggregate across time by addition: the sum of the
        single-period log returns equals the total log return ``ln(P_last / P_first)``, the Meucci across-time identity.
        """
        values = case
        log_returns = apply_expr(values, returns_log(pl.col(COLUMN_X)))
        assert log_returns[0] is None
        assert all(value is not None for value in log_returns[1:])
        total = math.fsum(value for value in log_returns[1:] if value is not None)
        expected = math.log(values[-1] / values[0])
        assert math.isclose(total, expected, rel_tol=RELATIVE_TOLERANCE_PROPERTY, abs_tol=ABSOLUTE_TOLERANCE_REFERENCE)
