"""
Tests for ``pomata.indicators.parabolic_sar`` — Wilder's Parabolic SAR (stop and reverse).

``parabolic_sar`` is the second recursive kernel in the library: the whole stop-and-reverse recurrence runs in a
pure-Python ``map_batches`` kernel. It is multi-input (high, low), so tests use a local ``apply_parabolic_sar`` helper
to materialize the factory over a two-column ``Float64`` frame; ``assert_matches`` and the naive
``parabolic_sar_reference`` oracle (a structural re-implementation of the same recurrence, anchored by hand-derived
golden masters) are shared across the suite. It is homogeneous of degree 1.

The ladder is the canonical one: contract, edge (factor floors / warm-up / reversal / null / NaN), correctness (vs the
closed-form reference and a frozen golden master), and properties (reference agreement incl. missing data, degree-1
scale-homogeneity, large-magnitude stability). Categories are split into classes; cross-cutting categories use markers
(see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal
from tests.indicators.oracles import parabolic_sar_reference
from tests.support import (
    GROUP_KEY,
    HIGH,
    LOW,
    assert_matches,
    assert_scale_homogeneous,
    coherent_hl,
    coherent_hl_with_missing,
    materialize,
    split_pairs,
)

from pomata.indicators import parabolic_sar

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = 1   (a constant, not window-dependent: row 0 is null because the trend is seeded from the first two
#              bars, and the value at row 1 is the seed stop from which the recurrence runs)
#   2. memory  the oracle is a re-implementation of the same recurrence and seeding, so the property holds from
#              the first defined row (M = 0); with a constant warm-up of one row a case is simply a series of bars --
#              every row past the seed is output
#   3. domain  coherent_hl(): coherent (high >= low) positive-finite bars -- the stop-and-reverse recurrence is defined
#              on well-formed bars; the missing-data tier draws coherent_hl_with_missing. SERIES_MAX bars span several
#              total sizes
# parabolic_sar is a recursive state machine, homogeneous of degree 1 (the stop is a price level and the crossings are
# linear in price); its scale tier is sized by EXACT_TOLERANCE_FACTOR (an exact recursion under a power-of-two
# rescaling, so the residual is essentially zero and the factor is generous slack). It has no window parameter, so
# ``_cases`` draws only the series (no window to couple); ``acceleration`` / ``maximum`` are drawn per test. Repetitions
# N are the shared CI profile (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
SERIES_MAX = 50


@st.composite
def _cases[T](draw: st.DrawFn, bars: st.SearchStrategy[T]) -> list[T]:
    """
    A series of bars sized from the facts above. parabolic_sar is windowless (constant warm-up W = 1), so -- unlike the
    windowed indicators' ``(series, window)`` pair -- a case is just the series: every row past the one-row seed is
    output.
    """
    # NOTE: windowless -- returns the bare series (no window to couple length to); the W + D coupling of the windowed
    # ``_cases`` is vacuous here because the warm-up is the constant single seed row, not a function of a window.
    return draw(st.lists(bars, min_size=1, max_size=SERIES_MAX))


@st.composite
def _factors(draw: st.DrawFn) -> tuple[float, float]:
    """
    A valid ``(acceleration, maximum)`` pair: both are fractions in ``(0, 1]`` with ``acceleration <= maximum``, so the
    factor never runs above its cap. ``maximum`` is drawn from ``acceleration`` upward to keep the ordering by
    construction (an independent draw could land ``acceleration > maximum``, which the factory rejects).
    """
    acceleration = draw(st.floats(min_value=1e-3, max_value=0.1, allow_nan=False, allow_infinity=False))
    maximum = draw(st.floats(min_value=acceleration, max_value=0.5, allow_nan=False, allow_infinity=False))
    return acceleration, maximum


def apply_parabolic_sar(
    high: Sequence[float | None],
    low: Sequence[float | None],
    acceleration: float = 0.02,
    maximum: float = 0.20,
) -> list[float | None]:
    """
    Materialize ``parabolic_sar`` over a two-column ``Float64`` frame built from the aligned high / low lists.
    """
    return materialize(
        {HIGH: high, LOW: low}, parabolic_sar(pl.col(HIGH), pl.col(LOW), acceleration=acceleration, maximum=maximum)
    )


class TestParabolicSarContract:
    """
    Type, shape, and lazy/eager guarantees.
    """

    def test_returns_expr(self) -> None:
        """
        Verifies that the factory returns a ``pl.Expr`` without touching a frame.
        """
        assert isinstance(parabolic_sar(pl.col(HIGH), pl.col(LOW)), pl.Expr)

    def test_preserves_length_and_dtype(self) -> None:
        """
        Verifies that the output has one value per input row and is ``Float64``.
        """
        frame = pl.DataFrame({HIGH: [10.0, 11.0, 12.0], LOW: [9.0, 10.0, 11.0]})
        result = frame.select(parabolic_sar(pl.col(HIGH), pl.col(LOW)).alias("y"))
        assert result.height == frame.height
        assert result.schema["y"] == pl.Float64

    def test_lazy_eager_parity(self) -> None:
        """
        Verifies that eager and lazy application produce identical materialized output.
        """
        frame = pl.DataFrame({HIGH: [10.0, 11.0, 12.0, 11.0], LOW: [9.0, 10.0, 11.0, 10.0]})
        expr = parabolic_sar(pl.col(HIGH), pl.col(LOW)).alias("y")
        result_eager = frame.select(expr)
        result_lazy = frame.lazy().select(expr).collect()
        assert_frame_equal(result_eager, result_lazy)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the recurrence resets per group and never spans group boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * 5 + ["b"] * 5,
                HIGH: [10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 22.0, 21.0, 20.0],
                LOW: [9.0, 10.0, 11.0, 12.0, 13.0, 19.0, 20.0, 21.0, 20.0, 19.0],
            }
        )
        result = frame.select(parabolic_sar(pl.col(HIGH), pl.col(LOW)).over(GROUP_KEY).alias("y"))["y"].to_list()
        group_a = apply_parabolic_sar([10.0, 11.0, 12.0, 13.0, 14.0], [9.0, 10.0, 11.0, 12.0, 13.0])
        group_b = apply_parabolic_sar([20.0, 21.0, 22.0, 21.0, 20.0], [19.0, 20.0, 21.0, 20.0, 19.0])
        assert_matches(result, group_a + group_b)


class TestParabolicSarEdge:
    """
    Boundaries, warm-up, reversal, and null / NaN handling.
    """

    def test_invalid_acceleration_raises(self) -> None:
        """
        Verifies that an ``acceleration`` that is not a finite number ``> 0`` (zero, negative, ``NaN``, or ``±inf``)
        raises ``ValueError``.
        """
        for invalid in (0.0, -0.5, math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match=r"acceleration must be in the half-open interval \(0, 1\]"):
                parabolic_sar(pl.col(HIGH), pl.col(LOW), acceleration=invalid)

    def test_invalid_maximum_raises(self) -> None:
        """
        Verifies that a ``maximum`` that is not a finite number ``> 0`` (zero, negative, ``NaN``, or ``±inf``)
        raises ``ValueError``.
        """
        for invalid in (0.0, -0.5, math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match=r"maximum must be in the half-open interval \(0, 1\]"):
                parabolic_sar(pl.col(HIGH), pl.col(LOW), maximum=invalid)

    def test_acceleration_above_one_raises(self) -> None:
        """
        Verifies that ``acceleration > 1`` raises ``ValueError`` (the factor is a fraction in ``(0, 1]``).
        """
        with pytest.raises(ValueError, match=r"acceleration must be in the half-open interval \(0, 1\]"):
            parabolic_sar(pl.col(HIGH), pl.col(LOW), acceleration=1.5, maximum=2.0)

    def test_maximum_above_one_raises(self) -> None:
        """
        Verifies that ``maximum > 1`` raises ``ValueError`` (the factor is a fraction in ``(0, 1]``).
        """
        with pytest.raises(ValueError, match=r"maximum must be in the half-open interval \(0, 1\]"):
            parabolic_sar(pl.col(HIGH), pl.col(LOW), acceleration=0.02, maximum=1.5)

    def test_acceleration_above_maximum_raises(self) -> None:
        """
        Verifies that ``acceleration > maximum`` raises ``ValueError`` so the factor can never run above its cap.
        """
        with pytest.raises(ValueError, match="acceleration must be <= maximum"):
            parabolic_sar(pl.col(HIGH), pl.col(LOW), acceleration=0.10, maximum=0.02)

    def test_warmup_and_seed(self) -> None:
        """
        Verifies that row 0 is null (the trend seeds from the first two bars) and row 1 is the seed stop.
        """
        result = apply_parabolic_sar([10.0, 11.0, 12.0, 13.0], [9.0, 10.0, 11.0, 12.0])
        assert result[0] is None
        assert result[1] == 9.0

    def test_empty(self) -> None:
        """
        Verifies that an empty input yields an empty output.
        """
        assert_matches(apply_parabolic_sar([], []), [])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output.
        """
        assert_matches(
            apply_parabolic_sar([None, None, None, None], [None, None, None, None]), [None, None, None, None]
        )

    def test_single_row(self) -> None:
        """
        Verifies that a one-bar series is all warm-up: the stop-and-reverse recurrence needs two bars to seed the trend.
        """
        assert_matches(apply_parabolic_sar([10.0], [9.0]), [None])

    def test_reversal_flips_side(self) -> None:
        """
        Verifies that the stop trails below price in an up-trend and jumps above it on a reversal.
        """
        high = [10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0, 10.0, 11.0]
        low = [9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0, 9.0, 10.0]
        result = apply_parabolic_sar(high, low)
        # up-trend: the stop sits at or below the bar's low ...
        assert result[6] is not None
        assert result[6] <= low[6]
        # ... then a reversal lifts it above the bar's high.
        assert result[7] is not None
        assert result[7] >= high[7]

    def test_null_propagates(self) -> None:
        """
        Verifies that a null propagates (matching the naive reference).
        """
        high = [10.0, 11.0, 12.0, None, 14.0, 14.0, 12.0, 11.0]
        low = [9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0]
        assert_matches(apply_parabolic_sar(high, low), parabolic_sar_reference(high, low))

    def test_nan_propagates(self) -> None:
        """
        Verifies that a NaN propagates (matching the naive reference).
        """
        high = [10.0, 11.0, 12.0, 12.0, 14.0, math.nan, 12.0, 11.0]
        low = [9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0]
        assert_matches(apply_parabolic_sar(high, low), parabolic_sar_reference(high, low))


class TestParabolicSarCorrectness:
    """
    Against the reference oracle (internal-consistency for this recurrence) and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive reference across several acceleration / maximum settings.
        """
        high = [10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0]
        low = [9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 12.0]
        for acceleration, maximum in ((0.02, 0.20), (0.01, 0.10), (0.05, 0.50), (0.03, 0.03)):
            assert_matches(
                apply_parabolic_sar(high, low, acceleration, maximum),
                parabolic_sar_reference(high, low, acceleration, maximum),
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: parabolic_sar over the sample series with the default settings.
        """
        high = [10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0, 10.0, 11.0]
        low = [9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0, 9.0, 10.0]
        result = apply_parabolic_sar(high, low)
        assert_matches(
            [None if v is None else round(v, 4) for v in result],
            [None, 9.0, 9.0, 9.12, 9.3528, 9.7246, 10.0666, 14.0, 13.92, 13.7232],
        )

    def test_golden_master_short_seed(self) -> None:
        """
        Verifies a frozen golden master on a series whose first bar pair FALLS, so the trend seeds SHORT (every other
        deterministic series here opens with a rising pair and seeds long). Hand-derived from Wilder's short-seed rules
        at the default acceleration ``0.02`` / maximum ``0.20``.

        - bar 1 seeds short: the first up-move ``high_1 - high_0 = 9 - 10 = -1`` is below the down-move
          ``low_0 - low_1 = 9 - 8 = 1``, so the stop seeds at the prior high ``high_0 = 10`` and ``EP = low_1 = 8``.
        - bar 2: ``10 + 0.02 * (8 - 10) = 9.96``, clamped UP to the prior two highs ``max(9.96, 9, 10) = 10``; the new
          low ``7`` extends ``EP`` and lifts ``AF`` to ``0.04``.
        - bar 3: ``10 + 0.04 * (7 - 10) = 9.88`` (above both prior highs ``8, 9``); ``EP = 6``, ``AF = 0.06``.
        - bar 4: ``9.88 + 0.06 * (6 - 9.88) = 9.6472`` (above both prior highs ``7, 8``); ``low 7.5`` is above ``EP``
          and ``high 8.5`` is below the stop, so the trend holds.
        """
        high = [10.0, 9.0, 8.0, 7.0, 8.5]
        low = [9.0, 8.0, 7.0, 6.0, 7.5]
        result = apply_parabolic_sar(high, low)
        assert_matches(
            [None if v is None else round(v, 4) for v in result],
            [None, 10.0, 10.0, 9.88, 9.6472],
        )

    def test_golden_master_with_null(self) -> None:
        """
        Verifies a frozen golden master pinning the documented null behavior: a ``null`` bar yields ``null`` at that
        row and is skipped, while the running trend state bridges the gap and resumes unchanged on the next complete
        bar. Hand-derived on a rising series with a single mid-series ``null`` high at the default settings.

        - bar 1 seeds long (up-move ``11 - 10 = 1`` is at least the down-move ``9 - 10 = -1``): stop ``= low_0 = 9``,
          ``EP = high_1 = 11``, ``AF = 0.02``.
        - bar 2: ``9 + 0.02 * (11 - 9) = 9.04``, clamped DOWN to the prior two lows ``min(9.04, 10, 9) = 9``; the new
          high ``12`` extends ``EP`` and lifts ``AF`` to ``0.04``.
        - bar 3 has a ``null`` high, so it emits ``null`` and the state (stop ``9``, ``EP 12``, ``AF 0.04``, prior
          extremes from bars 1-2) is untouched.
        - bar 4 resumes from that bridged state: ``9 + 0.04 * (12 - 9) = 9.12`` (above neither prior low ``10, 11``);
          ``high 13`` extends ``EP`` and lifts ``AF`` to ``0.06``.
        - bar 5: ``9.12 + 0.06 * (13 - 9.12) = 9.3528``; ``high 14`` extends ``EP`` and lifts ``AF`` to ``0.08``.
        """
        high = [10.0, 11.0, 12.0, None, 13.0, 14.0]
        low = [9.0, 10.0, 11.0, 11.5, 12.0, 13.0]
        result = apply_parabolic_sar(high, low)
        assert_matches(
            [None if v is None else round(v, 4) for v in result],
            [None, 9.0, 9.0, None, 9.12, 9.3528],
        )


class TestParabolicSarProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(
        case=_cases(coherent_hl()),
        factors=_factors(),
    )
    def test_matches_reference_for_any_input(
        self,
        case: list[tuple[float, float]],
        factors: tuple[float, float],
    ) -> None:
        """
        Verifies that, for any series and settings, the implementation matches the naive reference.
        """
        acceleration, maximum = factors
        rows = case
        high, low = split_pairs(rows)
        assert_matches(
            apply_parabolic_sar(high, low, acceleration, maximum),
            parabolic_sar_reference(high, low, acceleration, maximum),
        )

    @given(
        case=_cases(coherent_hl()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
        factors=_factors(),
    )
    def test_scale_homogeneity(
        self,
        case: list[tuple[float, float]],
        exponent: int,
        factors: tuple[float, float],
    ) -> None:
        """
        Verifies that, for positive ``k``, ``parabolic_sar`` is homogeneous of degree 1: ``sar(k * x) == k * sar(x)``.
        ``k`` is a power of two so the rescaling is lossless and cannot perturb the stop-and-reverse recurrence; the
        acceleration / maximum factors are drawn too, so the invariant is checked across the factor space, not only the
        defaults.
        """
        acceleration, maximum = factors
        k = 2.0**exponent
        rows = case
        high, low = split_pairs(rows)
        high_scaled = [value * k for value in high]
        low_scaled = [value * k for value in low]
        base = apply_parabolic_sar(high, low, acceleration, maximum)
        scaled = apply_parabolic_sar(high_scaled, low_scaled, acceleration, maximum)
        assert_scale_homogeneous(scaled, base, k=k, degree=1)

    @given(
        case=_cases(coherent_hl()),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
        factors=_factors(),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: list[tuple[float, float]],
        scale: float,
        factors: tuple[float, float],
    ) -> None:
        """
        Verifies that at extreme magnitudes the implementation stays finite where the reference is and agrees, across
        the acceleration / maximum factor space rather than only the defaults.
        """
        acceleration, maximum = factors
        rows = case
        high = [row[0] * scale for row in rows]
        low = [row[1] * scale for row in rows]
        assert_matches(
            apply_parabolic_sar(high, low, acceleration, maximum),
            parabolic_sar_reference(high, low, acceleration, maximum),
        )

    @given(
        case=_cases(coherent_hl_with_missing()),
        factors=_factors(),
    )
    def test_matches_reference_under_missing_data(
        self,
        case: list[tuple[float | None, float | None]],
        factors: tuple[float, float],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN, the implementation matches the naive reference.
        """
        acceleration, maximum = factors
        rows = case
        high, low = split_pairs(rows)
        assert_matches(
            apply_parabolic_sar(high, low, acceleration, maximum),
            parabolic_sar_reference(high, low, acceleration, maximum),
        )
