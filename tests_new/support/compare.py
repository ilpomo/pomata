"""
Result comparison for the rungs: the element-wise ``null`` / ``NaN`` / finite assert and the scale-homogeneity assert.

Both are plain functions rather than pytest fixtures so they compose cleanly with Hypothesis ``@given`` tests. The rungs
import them from ``tests_new.support.compare``; :func:`first_mismatch` is the non-raising probe the rungs use to build a
rich failure message before re-raising.
"""

import math
from collections.abc import Sequence

from tests_new.support.tolerances import (
    TOLERANCE_ABSOLUTE_EXACT,
    TOLERANCE_FACTOR_EXACT,
    TOLERANCE_RELATIVE_EXACT,
    TOLERANCE_RELATIVE_SCALE,
    input_scale,
)


def _element_agrees(
    value_actual: float | None, value_expected: float | None, *, rel_tol: float, abs_tol: float
) -> bool:
    """Whether one actual value matches its expected value by the expected value's kind (null / NaN / ±inf / finite)."""
    if value_expected is None:
        return value_actual is None
    if math.isnan(value_expected):
        return value_actual is not None and math.isnan(value_actual)
    if math.isinf(value_expected):
        return value_actual is not None and math.isinf(value_actual) and (value_actual > 0.0) == (value_expected > 0.0)
    return (
        value_actual is not None
        and not math.isnan(value_actual)
        and math.isclose(value_actual, value_expected, rel_tol=rel_tol, abs_tol=abs_tol)
    )


def first_mismatch(
    result_actual: Sequence[float | None],
    result_expected: Sequence[float | None],
    *,
    rel_tol: float = TOLERANCE_RELATIVE_EXACT,
    abs_tol: float = TOLERANCE_ABSOLUTE_EXACT,
) -> int | None:
    """
    The index of the first element that fails its kind-specific check, or ``None`` if the two lists agree.

    The non-raising counterpart of :func:`assert_matches`: the rungs use it to locate a disagreement so
    :mod:`tests_new.support.messages` can point at the exact row before raising. A length mismatch reports the first
    index past the shorter list.

    Args:
        result_actual: The materialized output under test.
        result_expected: The expected output (oracle, golden, or pin), compared element-wise by the expected kind.
        rel_tol: Relative tolerance forwarded to :func:`math.isclose` for the finite-value comparison.
        abs_tol: Absolute tolerance forwarded to :func:`math.isclose` for the finite-value comparison.

    Returns:
        The first disagreeing index, or ``None`` when every element matches.
    """
    if len(result_actual) != len(result_expected):
        return min(len(result_actual), len(result_expected))
    for index, (value_actual, value_expected) in enumerate(zip(result_actual, result_expected, strict=True)):
        if not _element_agrees(value_actual, value_expected, rel_tol=rel_tol, abs_tol=abs_tol):
            return index
    return None


def assert_matches(
    result_actual: Sequence[float | None],
    result_expected: Sequence[float | None],
    *,
    rel_tol: float = TOLERANCE_RELATIVE_EXACT,
    abs_tol: float = TOLERANCE_ABSOLUTE_EXACT,
) -> None:
    """
    Assert two result lists are equal element-wise, matching ``None`` and ``NaN`` exactly and floats within tolerance.

    The two lists must have the same length, and each pair is compared by the kind of the expected value so the suite's
    three-way ``null`` / ``NaN`` / finite distinction is enforced rather than blurred. A ``None`` expected value needs
    the actual to be exactly ``None``; a ``NaN`` expected value requires the actual to be a ``float`` ``NaN`` (never
    ``None``); an infinite expected value requires a ``float`` infinity of the same sign; a finite expected
    value requires the actual value to be a non-``None``, non-``NaN`` ``float`` that is close within ``rel_tol`` /
    ``abs_tol``. The default tolerances are tight (``1e-12``); property-based tests that accumulate floating-point error
    pass looser values explicitly (see :mod:`tests_new.support.tolerances`).

    Args:
        result_actual: The materialized output under test.
        result_expected: The expected output, typically from a naive reference oracle or a frozen golden master; must be
            the same length as ``result_actual``.
        rel_tol: Relative tolerance forwarded to :func:`math.isclose` for the finite-value comparison.
        abs_tol: Absolute tolerance forwarded to :func:`math.isclose` for the finite-value comparison.

    Raises:
        AssertionError: If the lengths differ, or any element fails its kind-specific check.
    """
    assert len(result_actual) == len(result_expected)
    assert first_mismatch(result_actual, result_expected, rel_tol=rel_tol, abs_tol=abs_tol) is None


def assert_scale_homogeneous(
    result_scaled: Sequence[float | None],
    result_base: Sequence[float | None],
    *,
    k: float,
    degree: int,
    rel_tol: float = TOLERANCE_RELATIVE_SCALE,
) -> None:
    """
    Assert ``result_scaled`` equals ``result_base`` rescaled by ``k ** degree`` element-wise (degree-``degree``
    homogeneity; ``degree=0`` is invariance; ``degree=2`` e.g. for a variance), matching ``None`` and ``NaN`` exactly.

    The absolute floor is sized to the SCALED result magnitude -- ``input_scale(result_base) * |k| ** degree *
    TOLERANCE_FACTOR_EXACT`` -- so it tracks the rescaling: it neither swamps a small-``k`` comparison (the failure mode
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
    abs_tol = input_scale(result_base) * abs(k) ** degree * TOLERANCE_FACTOR_EXACT
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
