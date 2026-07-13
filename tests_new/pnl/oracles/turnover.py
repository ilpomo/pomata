"""
Naive reference oracle for ``pomata.pnl.turnover``.
"""

from collections.abc import Sequence


def turnover_reference(
    weight: Sequence[float | None],
) -> list[float | None]:
    """
    Naive Turnover over a Python list.

    The absolute one-bar change ``|weight[t] - weight[t-1]|`` with the pre-series weight taken as flat (``0``),
    recomputed as the oracle for :func:`pomata.pnl.turnover`. So the first row is ``|weight[0]|`` (the entry trade
    from cash), not ``None``; its subtlety is the missing-data rule of a two-endpoint difference — a ``None`` voids its
    own row and the next, a ``nan`` propagates likewise — detailed below.

    Args:
        weight: Signed weights for each bar (may contain ``None`` and ``float('nan')``).

    Returns:
        A list the same length as ``weight``: the traded fraction at each row, with the first row ``|weight[0]|``.

    Raises:
        None.

    Note:
        Edge-case behavior:

        - **Flat start** — the pre-series weight is ``0``, so the first row is ``|weight[0]|`` rather than ``None``.
        - **Null** — a ``None`` makes its own row ``None`` and the next row ``None`` (the difference references the
          previous weight); ``None`` takes precedence over ``nan``.
        - **NaN** — a ``nan`` propagates to its own row and the next, yielding ``nan`` there.
    """
    results: list[float | None] = []
    for index in range(len(weight)):
        current = weight[index]
        previous = weight[index - 1] if index > 0 else 0.0
        if current is None or previous is None:
            results.append(None)
        else:
            results.append(abs(current - previous))
    return results
