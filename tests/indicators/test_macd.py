"""
Tests for ``pomata.indicators.macd`` — the Moving Average Convergence/Divergence oscillator.

``macd`` is a multi-output indicator: it returns a single struct ``pl.Expr`` with the fields ``macd`` / ``signal`` /
``histogram``. The local ``apply_macd`` helper materializes each field over a one-column ``Float64`` frame into a dict
of lists, so the shared ``assert_matches`` and the naive ``macd_reference`` oracle (which returns the matching dict)
compare field by field. Tests use small spans (fast 2, slow 3, signal 2) to keep the warm-up short.

The ladder is the canonical one: contract (type / struct schema / shape / lazy-eager / ``.over`` independence), edge
(warm-up / fast-equals-slow / null / NaN), correctness (vs the closed-form reference, a frozen golden master, and the
histogram identity), and properties (reference agreement incl. missing data, degree-1 scale-homogeneity, and
large-magnitude stability — ``macd`` is scale-dependent). Categories are split into classes; cross-cutting categories
use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import macd_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_EXACT,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_REFERENCE,
    RELATIVE_TOLERANCE_SCALE,
    STREAMING_TOLERANCE_FACTOR,
    apply_expr,
    assert_matches,
    assert_scale_homogeneous,
    input_scale,
    missing_data_floats,
)

from pomata.indicators import macd

FIELDS = ("macd", "signal", "histogram")

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  W = (max(window_fast, window_slow) - 1) + (window_signal - 1)   (the MACD line warms up over
#              ``max(window_fast, window_slow) - 1`` rows; the signal / histogram carry the extra ``window_signal - 1``
#              rows on top, so the deepest field is null for W rows). The property tier holds the spans fixed at
#              WINDOW_FAST / WINDOW_SLOW / WINDOW_SIGNAL, so W is the constant below.
#   2. memory  the oracle shares pomata's recursive EMA seeding, so the property holds from the first defined row
#              (M = 0); each example carries D in [1, SERIES_MAX] defined rows on top of W, so the deepest field always
#              has output to check -- never an all-warm-up series
#   3. domain  finite values, the safe magnitude widened per test below
# Every field is homogeneous of degree 1 in the price (the EMAs and their differences all scale with it), so macd
# carries a degree-1 scale-homogeneity property and magnitude-relative tolerances: the streaming EMA difference and its
# two-pass oracle diverge by about ``input_scale * machine_eps`` on degenerate inputs, sized by
# STREAMING_TOLERANCE_FACTOR. Repetitions N are the shared CI profile (tests/conftest.py); override per-test only if its
# parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
WINDOW_FAST = 2
WINDOW_SLOW = 3
WINDOW_SIGNAL = 2
WARMUP = (max(WINDOW_FAST, WINDOW_SLOW) - 1) + (WINDOW_SIGNAL - 1)
SERIES_MAX = 50


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T]) -> list[T]:
    """
    A series sized from the facts above: the property tier holds the spans fixed, so -- unlike the windowed indicators'
    ``(series, window)`` pair -- a case is just the series, floored at ``WARMUP + 1`` rows so the deepest field always
    has at least one defined output (never an all-warm-up series).
    """
    defined = draw(st.integers(min_value=1, max_value=SERIES_MAX))
    length = WARMUP + defined
    return draw(st.lists(values, min_size=length, max_size=length))


def apply_macd(
    values: Sequence[float | None],
    window_fast: int = WINDOW_FAST,
    window_slow: int = WINDOW_SLOW,
    window_signal: int = WINDOW_SIGNAL,
) -> dict[str, list[float | None]]:
    """
    Materialize each field of ``macd`` over a one-column frame, as a dict mirroring the oracle's output.
    """
    expr = macd(pl.col(COLUMN_X), window_fast=window_fast, window_slow=window_slow, window_signal=window_signal)
    return {field: apply_expr(values, expr.struct.field(field)) for field in FIELDS}


class TestMacdContract:
    """
    Type, struct schema, shape, and lazy/eager guarantees.
    """

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the EMAs reset per group: the partitioned MACD line equals the per-group calls.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * 5 + ["b"] * 5,
                COLUMN_X: [10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 19.0, 21.0, 22.0, 20.0],
            }
        )
        line = macd(pl.col(COLUMN_X), window_fast=2, window_slow=3, window_signal=2).struct.field("macd")
        grouped = frame.select(line.over(GROUP_KEY).alias("y"))["y"].to_list()
        group_a = apply_macd([10.0, 11.0, 12.0, 11.0, 13.0])["macd"]
        group_b = apply_macd([20.0, 19.0, 21.0, 22.0, 20.0])["macd"]
        assert_matches(grouped, group_a + group_b)

    def test_output_is_struct_with_named_fields(self) -> None:
        """
        Verifies that the output is a ``Float64`` struct with exactly the fields ``macd`` / ``signal`` / ``histogram``.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 2.0, 3.0, 4.0, 5.0])})
        dtype = frame.select(macd(pl.col(COLUMN_X), window_fast=2, window_slow=3, window_signal=2).alias("m")).schema[
            "m"
        ]
        assert isinstance(dtype, pl.Struct)
        assert [field.name for field in dtype.fields] == ["macd", "signal", "histogram"]
        assert all(field.dtype == pl.Float64 for field in dtype.fields)


class TestMacdEdge:
    """
    Boundaries, warm-up, the degenerate fast-equals-slow case, and null / NaN handling.
    """

    def test_window_below_one_raises(self) -> None:
        """
        Verifies that any window ``< 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_fast must be >= 1"):
            macd(pl.col(COLUMN_X), window_fast=0, window_slow=26, window_signal=9)
        with pytest.raises(ValueError, match="window_slow must be >= 1"):
            macd(pl.col(COLUMN_X), window_fast=12, window_slow=0, window_signal=9)
        with pytest.raises(ValueError, match="window_signal must be >= 1"):
            macd(pl.col(COLUMN_X), window_fast=12, window_slow=26, window_signal=0)

    def test_fast_above_slow_raises(self) -> None:
        """
        Verifies that ``window_fast > window_slow`` raises ``ValueError`` (the fast leg must be the shorter one), while
        the equal-window case is accepted.
        """
        with pytest.raises(ValueError, match="windows must be ordered window_fast <= window_slow"):
            macd(pl.col(COLUMN_X), window_fast=26, window_slow=12, window_signal=9)
        assert isinstance(macd(pl.col(COLUMN_X), window_fast=3, window_slow=3, window_signal=2), pl.Expr)

    def test_single_row(self) -> None:
        """
        Verifies behavior on a one-element series: the slow EMA never warms up, so every field is all warm-up.
        """
        bands = apply_macd([42.0])
        for field in FIELDS:
            assert_matches(bands[field], [None])

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields all null on every field.
        """
        bands = apply_macd([None, None, None, None])
        for field in FIELDS:
            assert_matches(bands[field], [None, None, None, None])

    def test_null_bridged(self) -> None:
        """
        Verifies that a ``null`` contaminates the recursive EMAs, yielding ``null`` on every field.
        """
        values = [10.0, 11.0, 12.0, None, 14.0, 15.0, 16.0, 17.0]
        bands = apply_macd(values)
        reference = macd_reference(values, 2, 3, 2)
        for field in FIELDS:
            assert_matches(bands[field], reference[field])

    def test_nan_latches(self) -> None:
        """
        Verifies that a ``NaN`` propagates through the EMAs on every field.
        """
        values = [10.0, 11.0, 12.0, math.nan, 14.0, 15.0, 16.0, 17.0]
        bands = apply_macd(values)
        reference = macd_reference(values, 2, 3, 2)
        for field in FIELDS:
            assert_matches(bands[field], reference[field])

    def test_warmup_null_count(self) -> None:
        """
        Verifies the MACD line warms up over ``window_slow - 1`` rows and the signal / histogram carry an extra
        ``window_signal - 1`` rows on top.
        """
        bands = apply_macd([10.0, 11.0, 12.0, 11.0, 13.0, 14.0])
        assert bands["macd"][:2] == [None, None]
        assert bands["macd"][2] is not None
        for field in ("signal", "histogram"):
            assert bands[field][:3] == [None, None, None]
            assert bands[field][3] is not None

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that when the longest window exceeds the series length every field is null (no slow-EMA value).
        """
        bands = apply_macd([1.0, 2.0])
        for field in FIELDS:
            assert_matches(bands[field], [None, None])

    def test_fast_equals_slow_is_zero(self) -> None:
        """
        Verifies that when ``window_fast == window_slow`` the MACD line (and so the signal and histogram) are zero.
        """
        bands = apply_macd([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], window_fast=3, window_slow=3, window_signal=2)
        for value in bands["macd"]:
            if value is not None:
                assert value == 0.0
        for value in bands["histogram"]:
            if value is not None:
                assert value == 0.0


class TestMacdCorrectness:
    """
    Against the naive reference oracle, frozen golden-master values, and the histogram identity.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies that every field agrees with the naive closed-form reference across several span combinations.
        """
        values = [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0, 14.0, 16.0, 15.0, 17.0]
        for window_fast, window_slow, window_signal in ((2, 3, 2), (3, 5, 4), (1, 4, 2)):
            bands = apply_macd(values, window_fast, window_slow, window_signal)
            reference = macd_reference(values, window_fast, window_slow, window_signal)
            for field in FIELDS:
                assert_matches(
                    bands[field],
                    reference[field],
                    rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                    abs_tol=ABSOLUTE_TOLERANCE_EXACT,
                )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: macd(fast=2, slow=3, signal=2) over the sample series.
        """
        bands = apply_macd([10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0])
        assert_matches(
            [None if v is None else round(v, 4) for v in bands["macd"]],
            [None, None, 0.5, 0.1667, 0.3889, 0.463, 0.1543, 0.3848],
        )
        assert_matches(
            [None if v is None else round(v, 4) for v in bands["signal"]],
            [None, None, None, 0.3333, 0.3704, 0.4321, 0.2469, 0.3388],
        )
        assert_matches(
            [None if v is None else round(v, 4) for v in bands["histogram"]],
            [None, None, None, -0.1667, 0.0185, 0.0309, -0.0926, 0.046],
        )

    def test_histogram_is_macd_minus_signal(self) -> None:
        """
        Verifies the defining identity ``histogram == macd - signal`` wherever all three are defined.
        """
        bands = apply_macd([10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0, 14.0, 16.0])
        for line, signal, histogram in zip(bands["macd"], bands["signal"], bands["histogram"], strict=True):
            if line is not None and signal is not None and histogram is not None:
                assert math.isclose(
                    histogram, line - signal, rel_tol=RELATIVE_TOLERANCE_REFERENCE, abs_tol=ABSOLUTE_TOLERANCE_EXACT
                )


class TestMacdProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(
        case=_cases(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)),
    )
    def test_matches_reference_for_any_input(
        self,
        case: list[float],
    ) -> None:
        """
        Verifies that, for any series, every field matches the naive reference.
        """
        values = case
        bands = apply_macd(values)
        reference = macd_reference(values, WINDOW_FAST, WINDOW_SLOW, WINDOW_SIGNAL)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=input_scale(values) * STREAMING_TOLERANCE_FACTOR,
            )

    @given(
        case=_cases(missing_data_floats()),
    )
    def test_matches_reference_under_missing_data(
        self,
        case: list[float | None],
    ) -> None:
        """
        Verifies that, for inputs freely mixing null / NaN / finite, every field matches the naive reference.
        """
        values = case
        bands = apply_macd(values)
        reference = macd_reference(values, WINDOW_FAST, WINDOW_SLOW, WINDOW_SIGNAL)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=input_scale(values) * STREAMING_TOLERANCE_FACTOR,
            )

    @given(
        case=_cases(st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)),
        exponent=st.sampled_from([-4, -3, -2, -1, 1, 2, 3, 4]),
    )
    def test_scale_homogeneity(
        self,
        case: list[float],
        exponent: int,
    ) -> None:
        """
        Verifies that ``macd`` is homogeneous of degree 1: scaling every input value by a constant ``k`` scales the
        output by the same ``k`` -- ``macd(k * x) == k * macd(x)``. ``k`` is a power of two, so the rescale is exact
        and adds no floating-point error.
        """
        k = 2.0**exponent
        values = case
        scaled_values = [value * k for value in values]
        base = apply_macd(values)
        scaled = apply_macd(scaled_values)
        for field in FIELDS:
            assert_scale_homogeneous(scaled[field], base[field], k=k, degree=1)

    @given(
        case=_cases(st.floats(min_value=1e-3, max_value=1.0, allow_nan=False, allow_infinity=False)),
        scale=st.sampled_from([1e-6, 1e6, 1e9]),
    )
    def test_matches_reference_at_large_magnitude(
        self,
        case: list[float],
        scale: float,
    ) -> None:
        """
        Verifies that at extreme magnitudes every field stays finite where the reference is and agrees.
        """
        values = case
        scaled_values = [value * scale for value in values]
        bands = apply_macd(scaled_values)
        reference = macd_reference(scaled_values, WINDOW_FAST, WINDOW_SLOW, WINDOW_SIGNAL)
        for field in FIELDS:
            assert_matches(
                bands[field],
                reference[field],
                rel_tol=RELATIVE_TOLERANCE_SCALE,
                abs_tol=input_scale(scaled_values) * STREAMING_TOLERANCE_FACTOR,
            )
