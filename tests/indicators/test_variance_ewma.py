"""
Tests for ``pomata.indicators.variance_ewma`` — exponentially-weighted variance over a window.

``variance_ewma`` is single-input, so tests use the shared ``apply_expr`` helper to materialize the factory over a
one-column ``Float64`` frame; ``assert_matches`` and the naive ``variance_ewma_reference`` oracle (a direct two-pass
weighted variance over explicit per-observation weights) are shared across the suite. It is homogeneous of degree 2, so
it carries a degree-2 scale-homogeneity property and a large-magnitude property with magnitude-relative tolerances.

The ladder is the canonical one: contract, edge (window floor / warm-up / null-bridge / nan-latch), correctness (vs the
closed-form reference across ``adjust`` / ``bias`` and a frozen golden master), and properties (reference agreement
incl. missing data, scale-homogeneity, large magnitude). Categories are split into classes; cross-cutting categories use
markers (see ``tests/README.md``).
"""

import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import variance_ewma_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
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

from pomata.indicators import variance_ewma

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (the window must hold ``window`` non-null values before a result is emitted)
#   2. memory  the oracle shares pomata's seeding, so the property holds from the first defined row (M = 0); each
#              example carries D in [window, 2 * window] defined values -- one window of output, never all warm-up
#   3. domain  subnormal_safe_floats(bound): finite values floored away from the subnormal-square underflow (the squared
#              deviation must stay representable); ``bound`` is the safe magnitude, widened per test below
# Windows span ``window_min`` .. WINDOW_MAX; an EWM dispersion is degenerate for one observation, so ``window_min`` is
# 2.
# Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its parameter space is larger.
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


class TestVarianceEwmaContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the recursion resets per group and never spans group boundaries.
        """
        frame = pl.DataFrame(
            {GROUP_KEY: ["a"] * 4 + ["b"] * 4, COLUMN_X: [10.0, 11.0, 13.0, 12.0, 20.0, 22.0, 26.0, 24.0]}
        )
        expr = variance_ewma(pl.col(COLUMN_X), 3).over(GROUP_KEY)
        grouped = frame.select(expr.alias("y"))["y"].to_list()
        group_a = apply_expr([10.0, 11.0, 13.0, 12.0], variance_ewma(pl.col(COLUMN_X), 3))
        group_b = apply_expr([20.0, 22.0, 26.0, 24.0], variance_ewma(pl.col(COLUMN_X), 3))
        assert_matches(grouped, group_a + group_b)


class TestVarianceEwmaEdge:
    """
    Boundaries, warm-up, and null / NaN handling.
    """

    def test_window_below_two_raises(self) -> None:
        """
        Verifies that ``window < 2`` raises ``ValueError`` (an EWM dispersion is degenerate for one observation).
        """
        with pytest.raises(ValueError, match="window must be >= 2"):
            variance_ewma(pl.col(COLUMN_X), 1)

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series is all warm-up: a window of more than one observation yields null.
        """
        assert_matches(apply_expr([42.0], variance_ewma(pl.col(COLUMN_X), 2)), [None])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series stays null (no observation ever seeds the recursion).
        """
        assert_matches(apply_expr([None, None, None], variance_ewma(pl.col(COLUMN_X), 2)), [None, None, None])

    def test_null_bridges_and_nan_latches(self) -> None:
        """
        Verifies that an interior ``null`` yields ``null`` there but the recursion recovers afterwards, while a ``NaN``
        poisons the recursion and latches for every subsequent row.
        """
        values = [10.0, 11.0, 13.0, None, 14.0, math.nan, 16.0, 17.0]
        result = apply_expr(values, variance_ewma(pl.col(COLUMN_X), 3))
        assert_matches(result, variance_ewma_reference(values, 3))
        recovered = result[4]
        assert recovered is not None
        assert not math.isnan(recovered)
        latched = result[6]
        assert latched is not None
        assert math.isnan(latched)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are null (warm-up) and the first full window is defined.
        """
        result = apply_expr([1.0, 2.0, 3.0, 4.0, 5.0], variance_ewma(pl.col(COLUMN_X), 3))
        assert result[:2] == [None, None]
        assert result[2] is not None

    def test_window_exceeds_length(self) -> None:
        """
        Verifies the whole output is null when ``window`` exceeds the series length (warm-up never completes).
        """
        assert_matches(apply_expr([1.0, 2.0, 3.0], variance_ewma(pl.col(COLUMN_X), 5)), [None, None, None])


class TestVarianceEwmaCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive reference across several windows and both ``adjust`` / ``bias`` settings.
        """
        values = [10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0, 15.0]
        for window in (2, 3, 5):
            for adjust in (False, True):
                for bias in (True, False):
                    assert_matches(
                        apply_expr(values, variance_ewma(pl.col(COLUMN_X), window, adjust=adjust, bias=bias)),
                        variance_ewma_reference(values, window, adjust=adjust, bias=bias),
                    )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: variance_ewma(window=3) over the sample series.
        """
        result = apply_expr([10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0], variance_ewma(pl.col(COLUMN_X), 3).round(4))
        assert_matches(result, [None, None, 1.6875, 0.8594, 1.5586, 0.7803, 1.4216])

    def test_golden_master_interior_null(self) -> None:
        """
        Verifies a hand-computed frozen reference across an interior ``null`` (the gap re-weights the observations).

        For ``[10, None, 11, 13, 12]`` with ``window = 3`` (``alpha = 1/2``, ``adjust=False``, ``bias=True``) the
        ``None`` ages the lag of ``10`` while contributing no term, so the explicit weights are a forward product of
        blend fractions over the non-null observations. At the last defined row ``[10, ., 11, 13, 12]`` they reduce to
        the exact ratio ``w(10) : w(11) : w(13) : w(12) = 1 : 2 : 3 : 6`` (sum ``12``), giving weighted mean
        ``(10 + 22 + 39 + 72) / 12 = 143/12`` and variance ``(1*(10 - 143/12)^2 + 2*(11 - 143/12)^2 + 3*(13 - 143/12)^2
        + 6*(12 - 143/12)^2) / 12 = 107/144 = 0.7430555556``; the previous row is ``53/36 = 1.4722222222``. These pin
        the ``ignore_nulls=False`` aging that an equal-weight or null-collapsing form would miss.
        """
        result = apply_expr([10.0, None, 11.0, 13.0, 12.0], variance_ewma(pl.col(COLUMN_X), 3))
        assert_matches(
            result,
            [None, None, None, 1.4722222222222223, 0.7430555555555556],
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )


class TestVarianceEwmaProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(
        case=_cases(subnormal_safe_floats(bound=1e6), window_min=2),
        adjust=st.booleans(),
        bias=st.booleans(),
    )
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], int],
        adjust: bool,
        bias: bool,
    ) -> None:
        """
        Verifies that, for any series, window, and ``adjust`` / ``bias``, the implementation matches the reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, variance_ewma(pl.col(COLUMN_X), window, adjust=adjust, bias=bias)),
            variance_ewma_reference(values, window, adjust=adjust, bias=bias),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(values) ** 2 * VARIANCE_TOLERANCE_FACTOR,
        )

    @given(case=_cases(missing_data_floats(min_magnitude=SUBNORMAL_FLOOR), window_min=2))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[float | None], int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_expr(values, variance_ewma(pl.col(COLUMN_X), window)),
            variance_ewma_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=input_scale(values) ** 2 * VARIANCE_TOLERANCE_FACTOR,
        )

    @given(
        case=_cases(subnormal_safe_floats(), window_min=2),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(
        self,
        case: tuple[list[float], int],
        exponent: int,
    ) -> None:
        """
        Verifies that ``variance_ewma`` is homogeneous of degree 2: scaling every input value by a constant ``k``
        scales the output by ``k`` squared -- ``variance_ewma(k * x) == k**2 * variance_ewma(x)``. ``k`` is a power
        of two, so the rescale is exact and adds no floating-point error.
        """
        k = 2.0**exponent
        values, window = case
        scaled_values = [value * k for value in values]
        result_base = apply_expr(values, variance_ewma(pl.col(COLUMN_X), window))
        result_scaled = apply_expr(scaled_values, variance_ewma(pl.col(COLUMN_X), window))
        assert_scale_homogeneous(result_scaled, result_base, k=k, degree=2)

    @given(
        case=_cases(
            st.floats(min_value=1e-3, max_value=1.0, allow_nan=False, allow_infinity=False),
            window_min=2,
        ),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: tuple[list[float], int],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees.
        """
        values, window = case
        scaled_values = [value * scale for value in values]
        assert_matches(
            apply_expr(scaled_values, variance_ewma(pl.col(COLUMN_X), window)),
            variance_ewma_reference(scaled_values, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=input_scale(scaled_values) ** 2 * VARIANCE_TOLERANCE_FACTOR,
        )
