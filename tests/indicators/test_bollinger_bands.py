"""
Tests for ``pomata.indicators.bollinger_bands`` — volatility bands around a moving average.

``bollinger_bands`` is the first multi-output indicator: it returns a single struct ``pl.Expr`` with the fields
``lower`` / ``middle`` / ``upper``. The local ``apply_bollinger_bands`` helper materializes each field over a one-column
``Float64`` frame into a dict of lists, so the shared ``assert_matches`` and the naive ``bollinger_bands_reference``
oracle (which returns the matching dict) compare band by band.

The ladder is the canonical one: contract (type / struct schema / shape / lazy-eager / ``.over`` independence), edge
(warm-up / window collapse / single-row / null / NaN), correctness (vs the closed-form reference, a frozen golden
master, and ``num_std`` scaling), and properties (reference agreement incl. missing data, degree-1 scale-homogeneity,
and large-magnitude stability). Categories are split into classes; cross-cutting categories use markers (see
``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import bollinger_bands_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_EXACT,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    STREAMING_TOLERANCE_FACTOR,
    SUBNORMAL_FLOOR,
    WINDOW_MAX,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    input_scale,
    missing_data_floats,
    subnormal_safe_floats,
)

from pomata.indicators import bollinger_bands

FIELDS = ("lower", "middle", "upper")

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W(window) = window - 1   (every band is null until the window holds ``window`` non-null values, inherited
#              identically from the sma center and the rolling standard deviation)
#   2. memory  the oracle shares pomata's seeding, so the property holds from the first defined row (M = 0); each
#              example carries D in [window, 2 * window] defined values -- one window of output, never all warm-up
#   3. domain  subnormal_safe_floats(bound): finite values floored away from the subnormal-square underflow (the band
#              half-width squares the deviation, so the squared term must stay representable); ``bound`` is the safe
#              magnitude, widened per test below
# The bands compose the rolling standard deviation, so the band half-width is a degree-1 streaming statistic whose sqrt
# amplifies the degenerate residual -- hence the STREAMING_TOLERANCE_FACTOR, sized to the data as ``input_scale *
# factor``. Windows span 1 .. WINDOW_MAX. Repetitions N are the shared CI profile (tests/conftest.py); override per-test
# only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T]) -> tuple[list[T], int]:
    """
    A (series, window) pair sized from the facts above: ``window`` over its regimes, length = warm-up + a window of
    defined values, so every example has output to check (never an all-warm-up series, the waste a ``window`` decoupled
    from the length would cause).
    """
    window = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window, max_value=2 * window))
    length = (window - 1) + defined
    return draw(st.lists(values, min_size=length, max_size=length)), window


def apply_bollinger_bands(
    values: Sequence[float | None],
    window: int,
    num_std: float = 2.0,
) -> dict[str, list[float | None]]:
    """
    Materialize each band of ``bollinger_bands`` over a one-column frame, as a dict mirroring the oracle's output.
    """
    return {
        field: apply_expr(values, bollinger_bands(pl.col(COLUMN_X), window, num_std=num_std).struct.field(field))
        for field in FIELDS
    }


class TestBollingerBandsContract:
    """
    Type, struct schema, shape, and lazy/eager guarantees.
    """

    def test_output_is_struct_with_named_fields(self) -> None:
        """
        Verifies that the output is a ``Float64`` struct with exactly the fields ``lower`` / ``middle`` / ``upper``.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 2.0, 3.0, 4.0, 5.0])})
        dtype = frame.select(bollinger_bands(pl.col(COLUMN_X), 3).alias("bb")).schema["bb"]
        assert isinstance(dtype, pl.Struct)
        assert [field.name for field in dtype.fields] == ["lower", "middle", "upper"]
        assert all(field.dtype == pl.Float64 for field in dtype.fields)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the window resets per group and never spans group boundaries.
        """
        frame = pl.DataFrame(
            {GROUP_KEY: ["a", "a", "a", "b", "b", "b"], COLUMN_X: [10.0, 11.0, 12.0, 20.0, 22.0, 21.0]}
        )
        middle = bollinger_bands(pl.col(COLUMN_X), 2).over(GROUP_KEY).struct.field("middle")
        result = frame.select(middle.alias("y"))["y"].to_list()
        assert_matches(result, [None, 10.5, 11.5, None, 21.0, 21.5])


class TestBollingerBandsEdge:
    """
    Boundaries, warm-up, window collapse, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that ``window < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window must be >= 1"):
            bollinger_bands(pl.col(COLUMN_X), 0)

    def test_invalid_num_std_raises(self) -> None:
        """
        Verifies that a ``num_std`` that is not a finite number ``> 0`` (zero, negative, ``NaN``, or ``±inf``)
        raises ``ValueError`` (a non-positive width would collapse or invert the bands).
        """
        for invalid in (0.0, -1.0, math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="num_std must be a finite number > 0"):
                bollinger_bands(pl.col(COLUMN_X), 3, num_std=invalid)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that every band is null for the first ``window - 1`` rows and defined from the first full window.
        """
        bands = apply_bollinger_bands([1.0, 2.0, 3.0, 4.0, 5.0], 3)
        for field in FIELDS:
            assert bands[field][:2] == [None, None]
            assert bands[field][2] is not None

    def test_window_one_collapses_to_close(self) -> None:
        """
        Verifies that ``window == 1`` has zero deviation, so all three bands collapse onto ``close`` itself.
        """
        bands = apply_bollinger_bands([5.0, 6.0, 7.0], 1)
        for field in FIELDS:
            assert_matches(bands[field], [5.0, 6.0, 7.0])

    def test_single_row(self) -> None:
        """
        Verifies a one-element series: ``window == 1`` collapses onto the value, a larger window is all warm-up.
        """
        for field in FIELDS:
            assert_matches(apply_bollinger_bands([42.0], 1)[field], [42.0])
            assert_matches(apply_bollinger_bands([42.0], 3)[field], [None])

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a window longer than the series yields an all-null result on every band (warm-up never completes).
        """
        bands = apply_bollinger_bands([1.0, 2.0, 3.0], 5)
        for field in FIELDS:
            assert_matches(bands[field], [None, None, None])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null input yields an all-null output on every band (the window never holds a full set).
        """
        bands = apply_bollinger_bands([None, None, None], 2)
        for field in FIELDS:
            assert_matches(bands[field], [None, None, None])

    def test_null_in_window_is_null(self) -> None:
        """
        Verifies that a ``null`` in the window yields ``null`` on every band, recovering once the window clears.
        """
        bands = apply_bollinger_bands([1.0, None, 3.0, 4.0], 2)
        assert_matches(bands["middle"], [None, None, None, 3.5])
        for field in FIELDS:
            assert bands[field][:3] == [None, None, None]
            assert bands[field][3] is not None

    def test_nan_propagates(self) -> None:
        """
        Verifies that a ``NaN`` in the window yields ``NaN`` on every band (``null`` still takes precedence).
        """
        values = [1.0, math.nan, 3.0, 4.0]
        bands = apply_bollinger_bands(values, 2)
        assert_matches(bands["middle"], [None, math.nan, math.nan, 3.5])
        reference = bollinger_bands_reference(values, 2)
        for field in ("lower", "upper"):
            assert_matches(bands[field], reference[field])


class TestBollingerBandsCorrectness:
    """
    Against the naive reference oracle, frozen golden-master values, and the ``num_std`` contract.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies that every band agrees with the naive closed-form reference across several windows.
        """
        values = [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0]
        for window in (2, 3, 4, 5):
            bands = apply_bollinger_bands(values, window)
            reference = bollinger_bands_reference(values, window)
            for field in FIELDS:
                assert_matches(bands[field], reference[field])

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: bands(window=2, num_std=2) over [2, 4, 4, 8].
        """
        bands = apply_bollinger_bands([2.0, 4.0, 4.0, 8.0], 2)
        assert_matches(bands["lower"], [None, 1.0, 4.0, 2.0])
        assert_matches(bands["middle"], [None, 3.0, 4.0, 6.0])
        assert_matches(bands["upper"], [None, 5.0, 4.0, 10.0])

    def test_num_std_scales_width(self) -> None:
        """
        Verifies that the band half-width is linear in ``num_std``: the gap to the center doubles from ``1`` to ``2``.
        """
        values = [10.0, 11.0, 12.0, 11.0, 13.0]
        narrow = apply_bollinger_bands(values, 3, num_std=1.0)
        wide = apply_bollinger_bands(values, 3, num_std=2.0)
        for index in range(len(values)):
            center = narrow["middle"][index]
            narrow_upper = narrow["upper"][index]
            wide_upper = wide["upper"][index]
            if center is None:
                assert wide_upper is None
                continue
            assert narrow_upper is not None
            assert wide_upper is not None
            narrow_gap = narrow_upper - center
            wide_gap = wide_upper - center
            assert math.isclose(
                wide_gap, 2.0 * narrow_gap, rel_tol=RELATIVE_TOLERANCE_REFERENCE, abs_tol=ABSOLUTE_TOLERANCE_EXACT
            )


class TestBollingerBandsProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(subnormal_safe_floats(bound=1e6)))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], int],
    ) -> None:
        """
        Verifies that, for any series and window, every band matches the naive reference.
        """
        values, window = case
        bands = apply_bollinger_bands(values, window)
        reference = bollinger_bands_reference(values, window)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=input_scale(values) * STREAMING_TOLERANCE_FACTOR,
            )

    @given(
        case=_cases(subnormal_safe_floats()),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(
        self,
        case: tuple[list[float], int],
        exponent: int,
    ) -> None:
        """
        Verifies that ``bollinger_bands`` is homogeneous of degree 1: scaling every input value by a constant ``k``
        scales the output by the same ``k`` -- ``bollinger_bands(k * x) == k * bollinger_bands(x)``. ``k`` is a
        power of two, so the rescale is exact and adds no floating-point error.
        """
        k = 2.0**exponent
        values, window = case
        scaled_values = [value * k for value in values]
        base = apply_bollinger_bands(values, window)
        scaled = apply_bollinger_bands(scaled_values, window)
        for field in FIELDS:
            assert_scale_homogeneous(scaled[field], base[field], k=k, degree=1)

    @given(case=_cases(missing_data_floats(min_magnitude=SUBNORMAL_FLOOR)))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[float | None], int],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, every band matches the naive reference.
        """
        values, window = case
        bands = apply_bollinger_bands(values, window)
        reference = bollinger_bands_reference(values, window)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=input_scale(values) * STREAMING_TOLERANCE_FACTOR,
            )

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
        Verifies that at extreme positive magnitudes every band stays finite where the reference is and agrees.
        """
        values, window = case
        scaled_values = [value * scale for value in values]
        bands = apply_bollinger_bands(scaled_values, window)
        reference = bollinger_bands_reference(scaled_values, window)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_SCALE,
                abs_tol=input_scale(scaled_values) * STREAMING_TOLERANCE_FACTOR,
            )
