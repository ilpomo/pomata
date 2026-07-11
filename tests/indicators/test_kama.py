"""
Tests for ``pomata.indicators.kama`` — Kaufman's Adaptive Moving Average.

``kama`` is the first recursive kernel in the library: its efficiency ratio and smoothing constant are native Polars
expressions, but the adaptive recurrence runs in a pure-Python ``map_batches`` kernel. The local ``apply_kama`` helper
materializes it over a one-column ``Float64`` frame; ``assert_matches`` and the naive ``kama_reference`` oracle (whose
efficiency ratio / smoothing constant are recomputed in a naive Python loop, a structural mirror confirming internal
consistency) are shared across the suite. It is homogeneous of degree 1.

The ladder is the canonical one: contract, edge (window floors / warm-up / seed / flat / null / NaN), correctness (vs
the closed-form reference and a frozen golden master), and properties (reference agreement incl. missing data, degree-1
scale-homogeneity, large-magnitude stability). Categories are split into classes; cross-cutting categories use markers
(see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import kama_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_EXACT,
    COLUMN_X,
    EXACT_TOLERANCE_FACTOR,
    GROUP_KEY,
    RELATIVE_TOLERANCE_REFERENCE,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    input_scale,
    positive_missing_data,
)

from pomata.indicators import kama

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (the first ``window - 1`` rows are null; row ``window - 1`` seeds the adaptive
#              recurrence at ``close`` itself, the first bar where the efficiency ratio is defined)
#   2. memory  the oracle shares pomata's seeding and recurrence, so the property holds from the first defined row
#              (M = 0); each example carries D in [window, 2 * window] defined values -- one window of output, never all
#              warm-up
#   3. domain  finite floats over the test's regime (any-input / scale / missing-data / large-magnitude), widened per
#              test below
# Windows span ``window_min`` .. WINDOW_MAX. Repetitions N are the shared CI profile (tests/conftest.py); override
# per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
WINDOW_MAX = 15


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


@st.composite
def _windows(draw: st.DrawFn) -> tuple[int, int]:
    """
    A ``(window_fast, window_slow)`` pair honoring the contract ``1 <= window_fast <= window_slow``: the slow bound is
    drawn first and the fast bound is then bounded above by it, so the now-rejected reversed regime is never sampled.
    """
    window_slow = draw(st.integers(min_value=1, max_value=40))
    window_fast = draw(st.integers(min_value=1, max_value=window_slow))
    return window_fast, window_slow


def apply_kama(
    values: Sequence[float | None],
    window: int,
    window_fast: int = 2,
    window_slow: int = 30,
) -> list[float | None]:
    """
    Materialize ``kama`` over a one-column ``Float64`` frame built from ``values``.
    """
    return apply_expr(values, kama(pl.col(COLUMN_X), window=window, window_fast=window_fast, window_slow=window_slow))


class TestKamaContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the recurrence resets per group and never spans group boundaries.
        """
        frame = pl.DataFrame(
            {GROUP_KEY: ["a"] * 5 + ["b"] * 5, COLUMN_X: [10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0, 22.0]}
        )
        result = frame.select(
            kama(pl.col(COLUMN_X), window=2, window_fast=2, window_slow=30).over(GROUP_KEY).alias("y")
        )["y"].to_list()
        group_a = apply_kama([10.0, 11.0, 12.0, 11.0, 13.0], 2)
        group_b = apply_kama([20.0, 22.0, 21.0, 23.0, 22.0], 2)
        assert_matches(result, group_a + group_b)


class TestKamaEdge:
    """
    Boundaries, warm-up, seed, flat window, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            kama(pl.col(COLUMN_X), window=0, window_fast=2, window_slow=30)

    def test_window_fast_below_one_raises(self) -> None:
        """
        Verifies that ``window_fast < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_fast must be >= 1"):
            kama(pl.col(COLUMN_X), window=10, window_fast=0, window_slow=30)

    def test_window_slow_below_one_raises(self) -> None:
        """
        Verifies that ``window_slow < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_slow must be >= 1"):
            kama(pl.col(COLUMN_X), window=10, window_fast=2, window_slow=0)

    def test_fast_above_slow_raises(self) -> None:
        """
        Verifies that ``window_fast > window_slow`` (the bounds reversed) raises ``ValueError``, while the equal-bounds
        boundary is accepted.
        """
        with pytest.raises(ValueError, match="windows must be ordered window_fast <= window_slow"):
            kama(pl.col(COLUMN_X), window=10, window_fast=30, window_slow=2)
        assert isinstance(kama(pl.col(COLUMN_X), window=10, window_fast=5, window_slow=5), pl.Expr)

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series is all warm-up for any window above one.
        """
        assert_matches(apply_kama([42.0], 3), [None])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output.
        """
        assert_matches(apply_kama([None, None, None, None], 3), [None, None, None, None])

    def test_null_bridged(self) -> None:
        """
        Verifies that an interior ``null`` is bridged: the recursion carries its state across the gap.
        """
        values = [10.0, 11.0, 12.0, None, 13.0, 14.0, 15.0, 16.0]
        assert_matches(apply_kama(values, 2), kama_reference(values, 2))

    def test_nan_latches(self) -> None:
        """
        Verifies that a NaN latches (matching the naive reference).
        """
        values = [10.0, 11.0, 12.0, 12.5, 13.0, math.nan, 15.0, 16.0]
        assert_matches(apply_kama(values, 2), kama_reference(values, 2))

    def test_warmup_and_seed(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are null and row ``window - 1`` seeds at ``close`` itself.
        """
        result = apply_kama([10.0, 11.0, 12.0, 11.0, 13.0], 3)
        assert result[:2] == [None, None]
        assert result[2] == 12.0

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a window exceeding the series length yields an all-null output.
        """
        assert_matches(apply_kama([1.0, 2.0, 3.0], 5), [None, None, None])

    def test_flat_window_stays_constant(self) -> None:
        """
        Verifies that a flat series gives efficiency ratio ``0`` (not ``0 / 0``), so KAMA stays on the constant value.
        """
        result = apply_kama([5.0, 5.0, 5.0, 5.0], 2)
        assert_matches(result, [None, 5.0, 5.0, 5.0])

    def test_interior_null_bridged(self) -> None:
        """
        Verifies the warm-up gate and gap-bridging for an interior ``null`` (``[2, 4, None, 8, 10, 12]``, window 2).
        """
        assert_matches(
            apply_kama([2.0, 4.0, None, 8.0, 10.0, 12.0], 2),
            [None, 4.0, None, None, None, 7.555555555555554],
        )


class TestKamaCorrectness:
    """
    Against the reference oracle (internal-consistency for this recurrence) and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive reference across several parameter sets.
        """
        values = [10.0, 11.0, 12.0, 11.0, 13.0, 12.5, 14.0, 13.0, 15.0, 14.5]
        for window, window_fast, window_slow in ((1, 2, 30), (2, 2, 30), (3, 2, 30), (4, 3, 20)):
            assert_matches(
                apply_kama(values, window, window_fast, window_slow),
                kama_reference(values, window, window_fast, window_slow),
                rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                abs_tol=ABSOLUTE_TOLERANCE_EXACT,
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: kama(window=2) over the sample series.
        """
        result = apply_kama([10.0, 11.0, 12.0, 11.0, 13.0, 12.5], 2)
        assert_matches(
            [None if v is None else round(v, 4) for v in result],
            [None, 11.0, 11.4444, 11.4426, 11.5522, 11.724],
        )


class TestKamaProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)),
        windows=_windows(),
    )
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], int],
        windows: tuple[int, int],
    ) -> None:
        """
        Verifies that, for any series and parameters, the implementation matches the naive reference.
        """
        values, window = case
        window_fast, window_slow = windows
        assert_matches(
            apply_kama(values, window, window_fast, window_slow),
            kama_reference(values, window, window_fast, window_slow),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_EXACT,
        )

    @given(case=_cases(positive_missing_data()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[float | None], int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window = case
        assert_matches(
            apply_kama(values, window),
            kama_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_EXACT,
        )

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(
        self,
        case: tuple[list[float], int],
        exponent: int,
    ) -> None:
        """
        Verifies that ``kama`` is homogeneous of degree 1: scaling every input value by a constant ``k`` scales the
        output by the same ``k`` -- ``kama(k * x) == k * kama(x)``. ``k`` is a power of two, so the rescale is exact
        and adds no floating-point error.
        """
        k = 2.0**exponent
        values, window = case
        scaled = [value * k for value in values]
        base = apply_kama(values, window)
        scaled_result = apply_kama(scaled, window)
        assert_scale_homogeneous(scaled_result, base, k=k, degree=1)

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
        Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees.
        """
        values, window = case
        scaled = [value * scale for value in values]
        assert_matches(
            apply_kama(scaled, window),
            kama_reference(scaled, window),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=input_scale(scaled) * EXACT_TOLERANCE_FACTOR,
        )
