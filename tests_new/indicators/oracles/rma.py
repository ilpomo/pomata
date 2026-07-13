"""
Naive reference oracle for ``pomata.indicators.rma``.
"""

from collections.abc import Sequence

from tests_new.indicators.oracles._helpers import seeded_recursive_mean_reference


def rma_reference(
    expr: Sequence[float | None],
    window: int,
) -> list[float | None]:
    """
    Naive Wilder Moving Average (RMA / SMMA) over a Python list.

    Wilder's recursive moving average (``alpha = 1 / window``, unadjusted), recomputed as the oracle for
    :func:`pomata.indicators.rma`. The recurrence is seeded with the simple average of the first ``window``
    observations -- Wilder's canonical initialization -- through the common :func:`seeded_recursive_mean_reference`
    engine, where the null/NaN and warm-up contract is made precise.

    Args:
        expr: Input series, the observations to smooth (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window, mapped to the Wilder smoothing factor
            ``alpha = 1 / window``. Must be ``>= 1``.

    Returns:
        A list the same length as ``expr``. The first ``window - 1`` entries are ``None`` (warm-up); thereafter the
        recursively smoothed value, seeded on the ``window``-th non-null observation with the simple average of the
        first ``window`` observations.

    Raises:
        ValueError: If ``window < 1``.

    Note:
        Edge-case behavior:

        - **Null** — a leading ``None`` run is skipped and does not consume warm-up budget (the seed lands on the
          ``window``-th non-null observation); an interior ``None`` yields ``None`` at that position while the
          path-dependent recursion bridges the gap, the running average's weight decaying across it.
        - **NaN** — once a ``nan`` enters it poisons the recursion and latches ``nan`` for every subsequent value.
        - **window == 1** — the smoothing factor is ``1``, the warm-up vanishes, and the result reproduces the input.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window == 1:
        return list(expr)
    return seeded_recursive_mean_reference(expr, 1.0 / window, window)
