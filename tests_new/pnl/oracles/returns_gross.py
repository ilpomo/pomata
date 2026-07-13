"""
Naive reference oracle for ``pomata.pnl.returns_gross``.
"""

from collections.abc import Sequence


def returns_gross_reference(
    weight: Sequence[float | None],
    asset_returns: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Gross Strategy Returns over two aligned Python lists.

    The elementwise product ``weight * asset_returns``, recomputed as the oracle for
    :func:`pomata.pnl.returns_gross`. It is a pure per-row product with no lag; its one subtlety is the missing-data
    rule of plain arithmetic — ``None`` in either input propagates to ``None`` (taking precedence over ``nan``) and a
    ``nan`` propagates to ``nan`` — detailed below.

    Args:
        weight: Signed weights for each bar (may contain ``None`` and ``float('nan')``).
        asset_returns: Per-bar asset returns (may contain ``None`` and ``float('nan')``); same length as ``weight``.

    Returns:
        A list the same length as the inputs: ``weight * asset_returns`` for each row.

    Raises:
        ValueError: If ``weight`` and ``asset_returns`` do not have the same length.

    Note:
        Edge-case behavior:

        - **Null** — a ``None`` in either input makes that row ``None`` (``None`` takes precedence over ``nan``).
        - **NaN** — a ``nan`` in either input (with no ``None`` at that row) propagates to ``nan``.
    """
    if len(weight) != len(asset_returns):
        raise ValueError("weight and asset_returns must have equal length")

    results: list[float | None] = []
    for weight_value, asset_return in zip(weight, asset_returns, strict=True):
        if weight_value is None or asset_return is None:
            results.append(None)
        else:
            results.append(weight_value * asset_return)
    return results
