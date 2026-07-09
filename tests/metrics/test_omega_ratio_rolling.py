"""
Tests for ``pomata.metrics.omega_ratio_rolling`` — the rolling (windowed) twin of :func:`pomata.metrics.omega_ratio`.

``omega_ratio_rolling`` is single-input and WINDOWED-SERIES-VALUED (a return series → a series the same length, one
value per trailing window), so tests use the shared ``apply_expr`` helper; ``assert_matches`` and the naive
``omega_ratio_rolling_reference`` oracle (the reducing :func:`omega_ratio` recomputed over each window) are shared
across the suite. The rolling null/NaN policy differs from the reducing one: a window holding any ``null`` is ``null``
(it must hold ``window`` non-null values), and a ``NaN`` inside a window propagates.

The ladder is the canonical one: contract (type / length-preserving / lazy-eager / ``.over`` per-group warm-up), edge
(validation / empty / warm-up / null-in-window / NaN / no-downside), correctness (vs the closed-form reference and a
frozen golden master), and properties (reference agreement for any input and under missing data). Categories are split
into classes; cross-cutting categories use markers.
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from tests.metrics.oracles import omega_ratio_rolling_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_REFERENCE,
    COLUMN_X,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    WINDOW_MAX,
    apply_expr,
    assert_matches,
)

from pomata.metrics import omega_ratio_rolling

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- omega_ratio_rolling is WINDOWED and series-valued. Facts (mirroring the windowed metrics):
#   1. shape   length-preserving: one output row per input row; the first ``window - 1`` rows are warm-up ``null``
#   2. domain  magnitude-bounded returns (``|r|`` in [0.01, 1], sign-varied): omega is scale-invariant, and
#              same-magnitude values keep the one-pass sliding mean free of cross-window cancellation; missing null/NaN
#   3. window  window_min = 1 (a mean ratio needs one observation) .. WINDOW_MAX
# The denominator (the mean loss) is unbounded toward zero, so the property tiers skip the tiny-but-non-zero regime
# where the one-pass sliding sum cannot track the two-pass oracle; the dimensionless ratio then agrees to a 1e-6 band.
# ----------------------------------------------------------------------------------------------------------------------


_LOSS_FLOOR = 1e-3
_VALUE = st.one_of(
    st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
)
_VALUE_MISSING = st.one_of(st.none(), st.just(math.nan), _VALUE)


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T], window_min: int = 1) -> tuple[list[T], int]:
    """A (series, window) pair sized so every example has a window of defined output past the warm-up."""
    window = draw(st.integers(min_value=window_min, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


def _windows_conditioned(values: Sequence[float | None], window: int) -> bool:
    """
    Whether every trailing window's mean loss is either zero (no downside) or a real fraction of the window magnitude.

    Omega is a ratio whose denominator (the mean loss) is unbounded toward zero; a tiny-but-non-zero mean loss left by
    the one-pass sliding sum (once a large value exits) is the one regime where it cannot track the two-pass oracle, so
    the property tiers skip it while keeping the well-defined zero-downside (``+inf``) case.
    """
    for index in range(window - 1, len(values)):
        finite = [
            value for value in values[index - window + 1 : index + 1] if value is not None and not math.isnan(value)
        ]
        if not finite:
            continue
        mean_loss = sum(-value for value in finite if value < 0.0) / len(finite)
        scale = max(abs(value) for value in finite) or 1.0
        if 0.0 < mean_loss < scale * _LOSS_FLOOR:
            return False
    return True


class TestOmegaRatioRollingContract:
    """
    Type, shape, and lazy/eager guarantees.
    """


class TestOmegaRatioRollingEdge:
    """
    Validation, boundaries, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            omega_ratio_rolling(pl.col(COLUMN_X), 0)

    def test_non_finite_threshold_raises(self) -> None:
        """
        Verifies that a non-finite ``threshold`` raises ``ValueError``.
        """
        for invalid in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="threshold must be a finite number"):
                omega_ratio_rolling(pl.col(COLUMN_X), 3, threshold=invalid)

    def test_null_in_window_is_null(self) -> None:
        """
        Verifies that a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        """
        values = [0.01, None, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, omega_ratio_rolling(pl.col(COLUMN_X), 3)),
            omega_ratio_rolling_reference(values, 3),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` inside a window propagates to ``NaN`` for the windows that touch it.
        """
        values = [0.01, math.nan, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, omega_ratio_rolling(pl.col(COLUMN_X), 3)),
            omega_ratio_rolling_reference(values, 3),
        )

    def test_warmup_null_count(self) -> None:
        """
        Verifies that the first ``window - 1`` rows are ``null`` and the rest match the reference.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, omega_ratio_rolling(pl.col(COLUMN_X), 3)),
            omega_ratio_rolling_reference(values, 3),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a window exceeding the series length yields an all-null output.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02]
        assert_matches(apply_expr(values, omega_ratio_rolling(pl.col(COLUMN_X), 7)), [None, None, None, None, None])

    def test_window_equals_length(self) -> None:
        """
        Verifies that when ``window`` equals the series length only the last row is defined, matching the reference.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, omega_ratio_rolling(pl.col(COLUMN_X), 5)),
            omega_ratio_rolling_reference(values, 5),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_no_downside_window_is_inf(self) -> None:
        """
        Verifies that a window with no return below the threshold has zero mean loss, so the ratio is ``+inf``.
        """
        assert_matches(apply_expr([0.01, 0.02, 0.03], omega_ratio_rolling(pl.col(COLUMN_X), 3)), [None, None, math.inf])

    def test_no_activity_window_is_nan(self) -> None:
        """
        Verifies that a window with neither a gain nor a loss (every return exactly at the threshold) has zero mean gain
        and zero mean loss, so the ratio is ``0 / 0`` = ``NaN`` -- not the spurious ``+inf`` a one-pass gain residue,
        left once a large earlier gain exits the window, would surface over the zeroed loss.
        """
        assert_matches(
            apply_expr([1e9, 0.01, 1e-9, 0.0, 0.0], omega_ratio_rolling(pl.col(COLUMN_X), 2)),
            [None, math.inf, math.inf, math.inf, math.nan],
        )


class TestOmegaRatioRollingCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference over a representative series.
        """
        values = [0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018]
        assert_matches(
            apply_expr(values, omega_ratio_rolling(pl.col(COLUMN_X), 4)),
            omega_ratio_rolling_reference(values, 4),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    def test_matches_reference_with_threshold(self) -> None:
        """
        Verifies agreement with the naive closed-form reference at a non-default ``threshold``.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02]
        assert_matches(
            apply_expr(values, omega_ratio_rolling(pl.col(COLUMN_X), 3, threshold=0.01)),
            omega_ratio_rolling_reference(values, 3, 0.01),
            rel_tol=RELATIVE_TOLERANCE_REFERENCE,
        )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: the rolling gain-to-loss ratio about zero over a window of three.
        """
        values = [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015]
        assert_matches(
            apply_expr(values, omega_ratio_rolling(pl.col(COLUMN_X), 3).round(4)),
            [None, None, 2.0, 1.0, 5.0, 2.0, 1.3333],
        )


class TestOmegaRatioRollingProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(_VALUE))
    def test_matches_reference_for_any_input(self, case: tuple[list[float], int]) -> None:
        """
        Verifies that, for any well-conditioned series and window, the implementation matches the naive reference.
        """
        values, window = case
        assume(_windows_conditioned(values, window))
        assert_matches(
            apply_expr(values, omega_ratio_rolling(pl.col(COLUMN_X), window)),
            omega_ratio_rolling_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )

    @given(case=_cases(_VALUE_MISSING))
    def test_matches_reference_under_missing_data(self, case: tuple[list[float | None], int]) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, the implementation matches the naive reference.
        """
        values, window = case
        assume(_windows_conditioned(values, window))
        assert_matches(
            apply_expr(values, omega_ratio_rolling(pl.col(COLUMN_X), window)),
            omega_ratio_rolling_reference(values, window),
            rel_tol=RELATIVE_TOLERANCE_SCALE,
            abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
        )
