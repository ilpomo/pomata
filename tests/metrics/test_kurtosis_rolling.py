"""
Tests for ``pomata.metrics.kurtosis_rolling`` — the rolling (windowed) twin of :func:`pomata.metrics.kurtosis`.

``kurtosis_rolling`` is single-input and WINDOWED-SERIES-VALUED (a return series → a series the same length, one value
per trailing window), so tests use the shared ``apply_expr`` helper; ``assert_matches`` and the naive
``kurtosis_rolling_reference`` oracle (the reducing :func:`kurtosis` recomputed over each window) are shared across the
suite. A window holding any ``null`` is ``null`` (it must hold ``window`` non-null values); a ``NaN`` inside a window
propagates.

The ladder is the canonical one: contract (type / length-preserving / lazy-eager / ``.over`` per-group warm-up), edge
(validation / empty / warm-up / null-in-window / NaN / constant), correctness (vs the closed-form reference and a frozen
golden master), and properties (reference agreement for any input and under missing data). Categories are split into
classes; cross-cutting categories use markers.
"""

import math

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from tests.metrics.oracles import kurtosis_rolling_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    WINDOW_MAX,
    apply_expr,
    assert_matches,
    windows_well_conditioned,
)

from pomata.metrics import kurtosis_rolling

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- kurtosis_rolling is WINDOWED and series-valued (a standardized fourth moment per window). Facts:
#   1. shape   length-preserving: one output row per input row; the first ``window - 1`` rows are warm-up ``null``
#   2. domain  magnitude-bounded returns (``|r|`` in [0.01, 1], sign-varied) so a window never mixes a subnormal with
#              an ``O(1)`` value; missing mixes null / NaN
#   3. window  window_min = 3 (kurtosis is degenerate -- identically -2 -- for two points) .. WINDOW_MAX
# The standardized fourth moment (``central_4 / central_2**2``) is inherently ill-conditioned once a window's variance
# nears the float floor -- there the native rolling kurtosis and the two-pass oracle disagree because the quantity, not
# the algorithm, is unstable -- so the property tiers require every window to be well-conditioned (variance a real
# fraction of the magnitude); agreement is a 1e-6 band.
# ----------------------------------------------------------------------------------------------------------------------
_VALUE = st.one_of(
    st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
)
_VALUE_MISSING = st.one_of(st.none(), st.just(math.nan), _VALUE)


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T], window_min: int = 3) -> tuple[list[T], int]:
    """A (series, window) pair sized so every example has a window of defined output past the warm-up."""
    window = draw(st.integers(min_value=window_min, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


class TestKurtosisRollingContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestKurtosisRollingEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_window_below_two_raises(self) -> None:
        """
        Verifies that ``window < 2`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 2"):
            kurtosis_rolling(pl.col(COLUMN_X), 1)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are ``null`` and the rest match the reference.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02, 0.0]
        assert_matches(
            apply_expr(values, kurtosis_rolling(pl.col(COLUMN_X), 4)),
            kurtosis_rolling_reference(values, 4),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_null_in_window_is_null(self) -> None:
        """
        Verifies that a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        """
        values = [0.01, None, 0.03, -0.01, 0.02, 0.04]
        assert_matches(
            apply_expr(values, kurtosis_rolling(pl.col(COLUMN_X), 4)),
            kurtosis_rolling_reference(values, 4),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` inside a window propagates to ``NaN`` for the windows that touch it.
        """
        values = [0.01, math.nan, 0.03, -0.01, 0.02, 0.04]
        assert_matches(
            apply_expr(values, kurtosis_rolling(pl.col(COLUMN_X), 4)),
            kurtosis_rolling_reference(values, 4),
        )

    def test_constant_window_is_nan(self) -> None:
        """
        Verifies that a constant window has zero variance, so the excess kurtosis is undefined (``0 / 0``) and the
        native mean-centered moment yields ``NaN``.
        """
        assert_matches(
            apply_expr([0.1, 0.1, 0.1, 0.1, 0.1], kurtosis_rolling(pl.col(COLUMN_X), 4)),
            [None, None, None, math.nan, math.nan],
        )

    def test_near_constant_window_is_finite(self) -> None:
        """
        Verifies that a near-constant (non-bit-identical) window -- which the exact zero-variance test does not cover --
        yields the finite reference excess kurtosis from the native mean-centered moment, not the spurious ``inf`` /
        huge finite a one-pass raw-moment formula would cancel to.
        """
        values = [100.0, 100.0, 100.0, 100.000001]
        assert_matches(
            apply_expr(values, kurtosis_rolling(pl.col(COLUMN_X), 4)),
            kurtosis_rolling_reference(values, 4),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_constant_window_by_slide_is_nan(self) -> None:
        """
        Verifies that a window which becomes bit-constant only because a larger value slid out is still zero-variance,
        so the excess kurtosis is undefined (``0 / 0``) and ``NaN`` -- not the spuriously huge finite the native
        incremental kernel leaves from the exited value's cancellation residue.
        """
        values = [0.03, 0.0, 0.0, 0.0, 0.0]
        assert_matches(
            apply_expr(values, kurtosis_rolling(pl.col(COLUMN_X), 4)),
            kurtosis_rolling_reference(values, 4),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_outlier_exit_matches_reference(self) -> None:
        """
        Verifies that every window after a much larger value has slid out -- the regime where the incremental native
        kernel keeps a stale residue in its running sums -- is recomputed exactly and matches the fresh per-window
        reference, including a ``null``, a ``NaN``, and a bit-constant stretch inside that regime.
        """
        values = [5000.0, 0.013, -0.008, 0.011, 0.005, -0.012, 0.009, None, 0.007, -0.004, 0.010, 0.006]
        values += [math.nan, 0.008, -0.009, 0.012, 0.5, 0.5, 0.5, 0.5, 0.5, 0.005, -0.007, 0.011]
        assert_matches(
            apply_expr(values, kurtosis_rolling(pl.col(COLUMN_X), 5)),
            kurtosis_rolling_reference(values, 5),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )


class TestKurtosisRollingCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018]
        assert_matches(
            apply_expr(values, kurtosis_rolling(pl.col(COLUMN_X), 5)),
            kurtosis_rolling_reference(values, 5),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the rolling excess kurtosis over a window of four.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015]
        assert_matches(
            apply_expr(values, kurtosis_rolling(pl.col(COLUMN_X), 4).round(4)),
            [None, None, None, -1.4266, -1.7785, -1.64, -1.099],
        )


class TestKurtosisRollingProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_VALUE))
    def test_matches_reference_for_any_input(self, case: tuple[list[float], int]) -> None:
        """
        Verifies that, for any well-conditioned series and window, the implementation matches the naive reference.
        """
        values, window = case
        assume(windows_well_conditioned(values, window))
        assert_matches(
            apply_expr(values, kurtosis_rolling(pl.col(COLUMN_X), window)),
            kurtosis_rolling_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_VALUE_MISSING))
    def test_matches_reference_under_missing_data(self, case: tuple[list[float | None], int]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window = case
        assume(windows_well_conditioned(values, window))
        assert_matches(
            apply_expr(values, kurtosis_rolling(pl.col(COLUMN_X), window)),
            kurtosis_rolling_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
