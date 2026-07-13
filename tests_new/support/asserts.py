"""
Result comparison for the test ladder: the element-wise ``null`` / ``NaN`` / finite assert and the scale-homogeneity
assert.

Both are plain functions rather than pytest fixtures so they compose cleanly with Hypothesis ``@given`` tests. The whole
suite imports them as ``from tests_new.support import assert_matches`` / ``assert_scale_homogeneous``.
"""

import math
from collections.abc import Sequence

import polars as pl
from tests_new.support.tolerances import (
    ABSOLUTE_TOLERANCE_EXACT,
    EXACT_TOLERANCE_FACTOR,
    RELATIVE_TOLERANCE_EXACT,
    RELATIVE_TOLERANCE_SCALE,
    input_scale,
)


def assert_matches(
    result_actual: Sequence[float | None],
    result_expected: Sequence[float | None],
    *,
    rel_tol: float = RELATIVE_TOLERANCE_EXACT,
    abs_tol: float = ABSOLUTE_TOLERANCE_EXACT,
) -> None:
    """
    Assert two result lists are equal element-wise, matching ``None`` and ``NaN`` exactly and floats within tolerance.

    The two lists must have the same length, and each pair is compared by the kind of the expected value so the suite's
    three-way ``null`` / ``NaN`` / finite distinction is enforced rather than blurred. A ``None`` expected value
    requires the actual value to be exactly ``None``; a ``NaN`` expected value requires the actual value to be a
    ``float`` ``NaN`` (and never ``None``); an infinite expected value requires a ``float`` infinity of the same sign;
    a finite expected value requires the actual value to be a non-``None``, non-``NaN`` ``float`` that is close within
    ``rel_tol`` / ``abs_tol``. The default tolerances are tight (``1e-12``); property-based tests that accumulate
    floating-point error pass looser values explicitly (see :mod:`tests_new.support.tolerances`).

    Args:
        result_actual: The materialized output under test (e.g. from :func:`tests_new.support.frames.apply_expr`).
        result_expected: The expected output, typically from a naive reference oracle or a frozen golden master; must be
            the same length as ``result_actual``.
        rel_tol: Relative tolerance forwarded to :func:`math.isclose` for the finite-value comparison.
        abs_tol: Absolute tolerance forwarded to :func:`math.isclose` for the finite-value comparison.

    Raises:
        AssertionError: If the lengths differ, or any element fails its kind-specific check: an expected ``None`` whose
            actual is not ``None``, an expected ``NaN`` whose actual is not a ``NaN`` ``float``, an expected infinity
            whose actual is not a ``float`` infinity of the same sign, or a finite expected value whose actual is
            ``None`` / ``NaN`` or lies outside the tolerance band.
    """
    assert len(result_actual) == len(result_expected)
    for value_actual, value_expected in zip(result_actual, result_expected, strict=True):
        if value_expected is None:
            assert value_actual is None
        elif math.isnan(value_expected):
            assert value_actual is not None
            assert math.isnan(value_actual)
        elif math.isinf(value_expected):
            assert value_actual is not None
            assert math.isinf(value_actual)
            assert (value_actual > 0.0) == (value_expected > 0.0)
        else:
            assert value_actual is not None
            assert not math.isnan(value_actual)
            assert math.isclose(value_actual, value_expected, rel_tol=rel_tol, abs_tol=abs_tol)


def assert_scale_homogeneous(
    result_scaled: Sequence[float | None],
    result_base: Sequence[float | None],
    *,
    k: float,
    degree: int,
    rel_tol: float = RELATIVE_TOLERANCE_SCALE,
) -> None:
    """
    Assert ``result_scaled`` equals ``result_base`` rescaled by ``k ** degree`` element-wise (degree-``degree``
    homogeneity; ``degree=0`` is invariance; ``degree=2`` e.g. for a variance), matching ``None`` and ``NaN`` exactly.

    The absolute floor is sized to the SCALED result magnitude -- ``input_scale(result_base) * |k| ** degree *
    EXACT_TOLERANCE_FACTOR`` -- so it tracks the rescaling: it neither swamps a small-``k`` comparison (the failure mode
    of a fixed absolute band, which would pass a wholly wrong value once ``|value_base * k ** degree|`` drops below it)
    nor demands a bit-equality the rescaling cannot deliver. Pair it with a bounded (power-of-two) scale factor so the
    floor never underflows to zero in the subnormal regime.

    Args:
        result_scaled: The output computed from the rescaled input.
        result_base: The output computed from the original input; the same length as ``result_scaled``.
        k: The scale factor applied to the input.
        degree: The homogeneity degree -- ``1`` for an output that scales linearly with the input, ``0`` for a
            scale-invariant one.
        rel_tol: Relative tolerance for the finite-value comparison (the scale tier's band by default).

    Raises:
        AssertionError: If the lengths differ, a ``None`` is unmatched, or any finite pair lies outside the band.
    """
    signed_factor = k**degree
    abs_tol = input_scale(result_base) * abs(k) ** degree * EXACT_TOLERANCE_FACTOR
    assert len(result_scaled) == len(result_base)
    for value_scaled, value_base in zip(result_scaled, result_base, strict=True):
        if value_base is None:
            assert value_scaled is None
        elif math.isnan(value_base):
            assert value_scaled is not None
            assert math.isnan(value_scaled)
        else:
            assert value_scaled is not None
            assert not math.isnan(value_scaled)
            assert math.isclose(value_scaled, value_base * signed_factor, rel_tol=rel_tol, abs_tol=abs_tol)


def assert_all_float64(dtype: pl.DataType, name: str) -> None:
    """
    Assert ``dtype`` is ``Float64`` -- or, for a multi-output struct result, that every field is ``Float64``.
    """
    if isinstance(dtype, pl.Struct):
        offenders = [field.name for field in dtype.fields if field.dtype != pl.Float64]
        assert not offenders, f"{name}: non-Float64 struct fields {offenders}"
    else:
        assert dtype == pl.Float64, f"{name}: {dtype}"
