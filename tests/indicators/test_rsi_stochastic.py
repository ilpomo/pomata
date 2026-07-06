"""
Tests for ``pomata.indicators.rsi_stochastic`` — the Stochastic Relative Strength Index (Stochastic RSI).

``rsi_stochastic`` is single-input but multi-output: it returns a single struct ``pl.Expr`` with the fields ``k`` /
``d``. The local ``apply_rsi_stochastic`` helper materializes each field over a one-column ``Float64`` frame into a dict
of lists, so the shared ``assert_matches`` and the naive ``rsi_stochastic_reference`` oracle (which returns the matching
dict) compare line by line. Both lines are scale-invariant (the underlying RSI is) and lie in ``[0, 100]``.

The ladder is the canonical one: contract (type / struct schema / shape / lazy-eager / ``.over`` independence), edge
(window floors / warm-up / flat RSI / null / NaN), correctness (vs the closed-form reference and a frozen golden
master), and properties (reference agreement incl. missing data, scale-invariance, boundedness). Categories are split
into classes; cross-cutting categories use markers (see ``tests/README.md``).
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.indicators.oracles import rsi_reference, rsi_stochastic_reference
from tests.support import (
    ABSOLUTE_TOLERANCE_PROPERTY,
    ABSOLUTE_TOLERANCE_SCALE,
    BOUND_MARGIN,
    COLUMN_X,
    GROUP_KEY,
    RELATIVE_TOLERANCE_PROPERTY,
    RELATIVE_TOLERANCE_SCALE,
    apply_expr,
    assert_matches,
    positive_missing_data,
)

from pomata.indicators import rsi_stochastic

# ----------------------------------------------------------------------------------------------------------------------
# Test sizing -- derived, not chosen; the rationale is the shared method in CORRECTNESS.md, the numbers are this
# indicator's. To add an indicator, set its three facts here; the property tier below is then the same shape as every
# other indicator's.
#   1. warmup  the %K line warms up over ``window_rsi + window_k - 1`` rows (the stacked rsi warm-up and the %K range
#              look-back), and %D over a further ``window_d - 1``, so the full warm-up is
#              W = window_rsi + window_k + window_d - 2 rows
#   2. memory  the oracle shares pomata's recursive Wilder seeding, so the property holds from the first defined row
#              (M = 0); each example carries D in [window_k, 2 * window_k] defined values past the full warm-up -- a
#              range look-back of output, never all warm-up
#   3. domain  strictly positive finite floats (the underlying rsi is well-behaved on monotone-safe prices); windows
#              span 1 .. WINDOW_MAX
# Both lines are bounded scale-INVARIANT ratios in ``[0, 100]`` (the underlying rsi already is): the value is O(1)
# whatever the input magnitude, so its tolerance is ABSOLUTE (never input_scale-sized), and it carries a
# scale-INVARIANCE property in place of the homogeneity / large-magnitude tests of a scale-dependent indicator -- a
# large-magnitude test would be vacuous because the common scale cancels in the ratio. Repetitions N are the shared CI
# profile (tests/conftest.py); override per-test only if its parameter space is larger.
# ----------------------------------------------------------------------------------------------------------------------
WINDOW_MAX = 10


@st.composite
def _cases[T](draw: st.DrawFn, values: st.SearchStrategy[T]) -> tuple[list[T], int, int, int]:
    """
    A (series, window_rsi, window_k, window_d) tuple sized from the facts above: each window over its regimes, length =
    warm-up + a range look-back of defined values, so every example has output on both lines to check (never an
    all-warm-up series, the waste windows decoupled from the length would cause).
    """
    window_rsi = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    window_k = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    window_d = draw(st.integers(min_value=1, max_value=WINDOW_MAX))
    defined = draw(st.integers(min_value=window_k, max_value=2 * window_k))
    length = (window_rsi + window_k + window_d - 2) + defined
    series = draw(st.lists(values, min_size=length, max_size=length))
    return series, window_rsi, window_k, window_d


FIELDS = ("k", "d")


def apply_rsi_stochastic(
    values: Sequence[float | None],
    window_rsi: int,
    window_k: int,
    window_d: int,
) -> dict[str, list[float | None]]:
    """
    Materialize each line of ``rsi_stochastic`` over a one-column frame, as a dict mirroring the oracle's output.
    """
    return {
        field: apply_expr(
            values,
            rsi_stochastic(pl.col(COLUMN_X), window_rsi=window_rsi, window_k=window_k, window_d=window_d).struct.field(
                field
            ),
        )
        for field in FIELDS
    }


def _is_nan(value: float | None) -> bool:
    """
    True only for a ``float('nan')`` (not for ``None``).
    """
    return isinstance(value, float) and math.isnan(value)


# The %K line is 100 * (rsi - rsi_min) / (rsi_max - rsi_min). When the RSI barely moves over the %K window the
# denominator is tiny and the quotient is ill-conditioned: the float-epsilon gap between the Polars RSI and the
# independent Python RSI is amplified without bound, so the implementation and the oracle can disagree there while both
# are correct. A window whose RSI range clears this many units is comfortably well-conditioned.
MIN_WELL_CONDITIONED_RSI_RANGE = 1.0


def _flat_rsi_positions(values: Sequence[float | None], window_rsi: int, window_k: int) -> set[int]:
    """
    The %K positions whose trailing ``window_k`` RSI window is too flat for the quotient to be well-conditioned.
    """
    rsi = rsi_reference(values, window_rsi)
    flat: set[int] = set()
    for index in range(len(rsi)):
        window = [
            value for value in rsi[max(0, index - window_k + 1) : index + 1] if value is not None and not _is_nan(value)
        ]
        if len(window) < window_k or (max(window) - min(window)) < MIN_WELL_CONDITIONED_RSI_RANGE:
            flat.add(index)
    return flat


def assert_lines_match(
    applied: dict[str, list[float | None]],
    reference: dict[str, list[float | None]],
    values: Sequence[float | None],
    window_rsi: int,
    window_k: int,
    window_d: int,
    *,
    rel_tol: float = 1e-7,
    abs_tol: float = 1e-7,
) -> None:
    """
    Compare both lines against the reference, dropping only the positions where %K is ill-conditioned.

    The %K denominator ``rsi_max - rsi_min`` is exact on both sides, so at every well-conditioned position the
    ``NaN`` / finite kind is enforced strictly (an implementation that spuriously emitted ``NaN``, or dropped a ``NaN``
    the maths demands, would fail here): the conditioning arguments are mandatory and the ``NaN``-mismatch skip fires
    only inside the flat-RSI set. There the RSI range is zero or near-zero: at an exactly-flat window %K is ``0 / 0``
    and a float-epsilon RSI difference flips it between ``NaN`` and a ``[0, 100]`` boundary (one side ``NaN``, the
    other not); at a merely tiny range the quotient is finite on both sides but the tiny denominator amplifies the RSI
    epsilon past any tolerance. ``_flat_rsi_positions`` flags both, so both the ``NaN``-kind check and the value
    comparison are dropped there (``%D`` averages ``window_d`` ``%K`` values, so a ``%D`` position goes when any ``%K``
    it averages is ill-conditioned). Input-``NaN`` propagation (a ``NaN`` driven by missing data, not a flat range) is
    compared everywhere else.
    """
    flat = _flat_rsi_positions(values, window_rsi, window_k)
    for field in FIELDS:
        actual: list[float | None] = []
        expected: list[float | None] = []
        for index, (value_actual, value_expected) in enumerate(zip(applied[field], reference[field], strict=True)):
            if field == "k" and index in flat:
                continue
            if field == "d" and any(j in flat for j in range(max(0, index - window_d + 1), index + 1)):
                continue
            actual.append(value_actual)
            expected.append(value_expected)
        assert_matches(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)


class TestRsiStochasticContract:
    """
    Type, struct schema, shape, and lazy/eager guarantees.
    """

    def test_output_is_struct_with_named_fields(self) -> None:
        """
        Verifies that the output is a ``Float64`` struct with exactly the fields ``k`` / ``d``.
        """
        frame = pl.DataFrame({COLUMN_X: pl.Series(COLUMN_X, [1.0, 2.0, 3.0, 4.0, 5.0])})
        dtype = frame.select(rsi_stochastic(pl.col(COLUMN_X), window_rsi=2, window_k=2, window_d=2).alias("s")).schema[
            "s"
        ]
        assert isinstance(dtype, pl.Struct)
        assert [field.name for field in dtype.fields] == ["k", "d"]
        assert all(field.dtype == pl.Float64 for field in dtype.fields)

    def test_over_partitions_independently(self) -> None:
        """
        Verifies that under ``.over`` the RSI recursion and windows reset per group and never span boundaries.
        """
        frame = pl.DataFrame(
            {
                GROUP_KEY: ["a"] * 6 + ["b"] * 6,
                COLUMN_X: [50.0, 51.0, 50.5, 52.0, 51.5, 53.0, 20.0, 21.0, 20.5, 22.0, 21.5, 23.0],
            }
        )
        grouped = frame.select(
            rsi_stochastic(pl.col(COLUMN_X), window_rsi=3, window_k=2, window_d=2)
            .over(GROUP_KEY)
            .struct.field("k")
            .alias("k")
        )["k"].to_list()
        group_a = apply_rsi_stochastic([50.0, 51.0, 50.5, 52.0, 51.5, 53.0], window_rsi=3, window_k=2, window_d=2)
        group_b = apply_rsi_stochastic([20.0, 21.0, 20.5, 22.0, 21.5, 23.0], window_rsi=3, window_k=2, window_d=2)
        assert_matches(grouped, group_a["k"] + group_b["k"])


class TestRsiStochasticEdge:
    """
    Boundaries, warm-up, flat RSI, and null / NaN handling.
    """

    def test_window_rsi_below_one_raises(self) -> None:
        """
        Verifies that ``window_rsi < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_rsi must be >= 1"):
            rsi_stochastic(pl.col(COLUMN_X), window_rsi=0, window_k=3, window_d=3)

    def test_window_k_below_one_raises(self) -> None:
        """
        Verifies that ``window_k < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_k must be >= 1"):
            rsi_stochastic(pl.col(COLUMN_X), window_rsi=3, window_k=0, window_d=3)

    def test_window_d_below_one_raises(self) -> None:
        """
        Verifies that ``window_d < 1`` raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="window_d must be >= 1"):
            rsi_stochastic(pl.col(COLUMN_X), window_rsi=3, window_k=3, window_d=0)

    def test_warmup_null_count(self) -> None:
        """
        Verifies that ``k`` warms up over ``window_rsi + window_k - 1`` rows and ``d`` over a further ``window_d - 1``.
        """
        values = [50.0, 51.0, 50.5, 52.0, 51.5, 53.0, 52.0, 54.0, 53.5, 55.0]
        result = apply_rsi_stochastic(values, window_rsi=3, window_k=3, window_d=2)
        assert result["k"][:5] == [None, None, None, None, None]
        assert result["k"][5] is not None
        assert result["d"][:6] == [None, None, None, None, None, None]
        assert result["d"][6] is not None

    def test_all_null(self) -> None:
        """
        Verifies that an all-null series yields an all-null output on both lines.
        """
        values = [None] * 10
        result = apply_rsi_stochastic(values, window_rsi=3, window_k=3, window_d=2)
        for field in FIELDS:
            assert_matches(result[field], [None] * 10)

    def test_single_row(self) -> None:
        """
        Verifies that a one-element series is all warm-up on both lines.
        """
        result = apply_rsi_stochastic([42.0], window_rsi=3, window_k=3, window_d=2)
        for field in FIELDS:
            assert_matches(result[field], [None])

    def test_window_exceeds_length(self) -> None:
        """
        Verifies that a series shorter than the longest window yields an all-null output on both lines.
        """
        values = [1.0, 2.0, 3.0]
        result = apply_rsi_stochastic(values, window_rsi=3, window_k=3, window_d=2)
        for field in FIELDS:
            assert_matches(result[field], [None, None, None])

    def test_flat_window_is_nan(self) -> None:
        """
        Verifies that a flat RSI (a sustained trend pinning it, so highest equals lowest) yields ``NaN`` on ``k``.
        """
        result = apply_rsi_stochastic([10.0, 11.0, 12.0, 13.0, 14.0], window_rsi=2, window_k=2, window_d=1)
        defined = [value for value in result["k"] if value is not None]
        assert defined
        assert all(math.isnan(value) for value in defined)

    def test_null_bridged(self) -> None:
        """
        Verifies that an interior ``null`` is bridged: the recursion carries its state across the gap.
        """
        values = [50.0, 51.0, 50.5, None, 52.0, 52.5, 53.0, 52.0, 54.0]
        applied = apply_rsi_stochastic(values, window_rsi=2, window_k=2, window_d=2)
        reference = rsi_stochastic_reference(values, 2, 2, 2)
        assert_lines_match(applied, reference, values, 2, 2, 2)

    def test_nan_latches(self) -> None:
        """
        Verifies that a NaN propagates (matching the naive reference).
        """
        values = [50.0, 51.0, 50.5, 51.0, 52.0, math.nan, 53.0, 52.0, 54.0]
        applied = apply_rsi_stochastic(values, window_rsi=2, window_k=2, window_d=2)
        reference = rsi_stochastic_reference(values, 2, 2, 2)
        assert_lines_match(applied, reference, values, 2, 2, 2)


class TestRsiStochasticCorrectness:
    """
    Against the naive reference oracle and frozen golden-master values.
    """

    def test_matches_reference(self) -> None:
        """
        Verifies agreement with the naive closed-form reference across several window triples.
        """
        values = [50.0, 51.0, 50.5, 52.0, 51.5, 53.0, 52.0, 54.0, 53.5, 55.0, 54.0, 56.0]
        for window_rsi, window_k, window_d in ((1, 1, 1), (3, 3, 2), (4, 3, 3), (2, 5, 1)):
            applied = apply_rsi_stochastic(values, window_rsi=window_rsi, window_k=window_k, window_d=window_d)
            reference = rsi_stochastic_reference(values, window_rsi, window_k, window_d)
            assert_lines_match(
                applied,
                reference,
                values,
                window_rsi,
                window_k,
                window_d,
                rel_tol=RELATIVE_TOLERANCE_PROPERTY,
                abs_tol=ABSOLUTE_TOLERANCE_PROPERTY,
            )

    def test_golden_master(self) -> None:
        """
        Verifies the frozen reference: rsi_stochastic(window_rsi=3, window_k=3, window_d=2) over the sample series.
        """
        values = [50.0, 51.0, 50.5, 52.0, 51.5, 53.0, 52.0, 54.0, 53.5, 55.0]
        result = apply_rsi_stochastic(values, window_rsi=3, window_k=3, window_d=2)
        assert_matches(
            [None if v is None else round(v, 4) for v in result["k"]],
            [None, None, None, None, None, 94.7368, 0.0, 81.5861, 44.2237, 100.0],
        )
        assert_matches(
            [None if v is None else round(v, 4) for v in result["d"]],
            [None, None, None, None, None, None, 47.3684, 40.793, 62.9049, 72.1118],
        )


class TestRsiStochasticProperties:
    """
    Invariants that must hold for all inputs (property-based).
    """

    @given(case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)))
    def test_matches_reference_for_any_input(
        self,
        case: tuple[list[float], int, int, int],
    ) -> None:
        """
        Verifies that, for any positive series and windows, the implementation matches the naive reference.
        """
        values, window_rsi, window_k, window_d = case
        applied = apply_rsi_stochastic(values, window_rsi=window_rsi, window_k=window_k, window_d=window_d)
        reference = rsi_stochastic_reference(values, window_rsi, window_k, window_d)
        assert_lines_match(
            applied,
            reference,
            values,
            window_rsi,
            window_k,
            window_d,
            rel_tol=RELATIVE_TOLERANCE_PROPERTY,
            abs_tol=ABSOLUTE_TOLERANCE_PROPERTY,
        )

    @given(
        case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)),
        # %K is a ratio (rsi - rsi_min) / (rsi_max - rsi_min); a nearly-flat RSI gives a tiny denominator, so a
        # non-power-of-two rescaling perturbs numerator and denominator unequally and the quotient is ill-conditioned.
        # Powers of two rescale losslessly, so scale-invariance is verified exactly without that floating-point noise.
        k=st.sampled_from([2.0**exponent for exponent in (-4, -3, -2, -1, 1, 2, 3, 4)]),
    )
    def test_scale_invariance(
        self,
        case: tuple[list[float], int, int, int],
        k: float,
    ) -> None:
        """
        Verifies that ``rsi_stochastic`` is scale-invariant: scaling every input value by a constant ``k`` leaves
        the output unchanged -- ``rsi_stochastic(k * x) == rsi_stochastic(x)``. ``k`` is a power of two, so the
        rescale is exact and adds no floating-point error.
        """
        values, window_rsi, window_k, window_d = case
        base = apply_rsi_stochastic(values, window_rsi=window_rsi, window_k=window_k, window_d=window_d)
        scaled = apply_rsi_stochastic(
            [value * k for value in values], window_rsi=window_rsi, window_k=window_k, window_d=window_d
        )
        flat = _flat_rsi_positions(values, window_rsi, window_k)
        for field in FIELDS:
            for index, (value_scaled, value_base) in enumerate(zip(scaled[field], base[field], strict=True)):
                if field == "k" and index in flat:
                    continue  # ill-conditioned flat-RSI point (see assert_lines_match)
                if field == "d" and any(j in flat for j in range(max(0, index - window_d + 1), index + 1)):
                    continue
                if value_base is None:
                    assert value_scaled is None
                elif math.isnan(value_base):
                    assert value_scaled is not None
                    assert math.isnan(value_scaled)
                else:
                    assert value_scaled is not None
                    assert not math.isnan(value_scaled)
                    assert math.isclose(
                        value_scaled, value_base, rel_tol=RELATIVE_TOLERANCE_SCALE, abs_tol=ABSOLUTE_TOLERANCE_SCALE
                    )

    @given(case=_cases(st.floats(min_value=1.0, max_value=1e3, allow_nan=False, allow_infinity=False)))
    def test_bounded(
        self,
        case: tuple[list[float], int, int, int],
    ) -> None:
        """
        Verifies that every defined value lies within ``[0, 100]`` (the current RSI is always within its own range).
        """
        values, window_rsi, window_k, window_d = case
        applied = apply_rsi_stochastic(values, window_rsi=window_rsi, window_k=window_k, window_d=window_d)
        for field in FIELDS:
            for value in applied[field]:
                if value is not None and not math.isnan(value):
                    assert -BOUND_MARGIN <= value <= 100.0 + BOUND_MARGIN

    @given(case=_cases(positive_missing_data()))
    def test_matches_reference_under_missing_data(
        self,
        case: tuple[list[float | None], int, int, int],
    ) -> None:
        """
        Verifies that, for positive inputs freely mixing null / NaN, the implementation matches the naive reference.
        """
        values, window_rsi, window_k, window_d = case
        applied = apply_rsi_stochastic(values, window_rsi=window_rsi, window_k=window_k, window_d=window_d)
        reference = rsi_stochastic_reference(values, window_rsi, window_k, window_d)
        assert_lines_match(applied, reference, values, window_rsi, window_k, window_d)
