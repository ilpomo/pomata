"""
Naive reference oracle for ``pomata.indicators.ema``.
"""

from collections.abc import Sequence

from tests_new.indicators.oracles._helpers import seeded_recursive_mean_reference


def ema_reference(
    expr: Sequence[float | None],
    window: int,
    *,
    adjust: bool = False,
) -> list[float | None]:
    """
    Naive Exponential Moving Average over a Python list.

    The exponentially-weighted moving average (``alpha = 2 / (window + 1)``), recomputed as the oracle for
    :func:`pomata.indicators.ema`. The unadjusted form (``adjust=False``, the default) is seeded with the simple
    average of the first ``window`` observations -- the classical EMA initialization -- through the common
    :func:`seeded_recursive_mean_reference` engine. The adjusted form (``adjust=True``) is the finite-window
    unbiased weighting, exact from the first observation, with no seed to choose.

    Args:
        expr: Input series, the observations to smooth (may contain ``None`` and ``float('nan')``).
        window: Number of observations in the moving window, mapped to the smoothing factor
            ``alpha = 2 / (window + 1)``. Must be ``>= 1``.
        adjust: When ``False`` (default) use the recursive technical-analysis form; when ``True`` use the finite-window
            unbiased weighting that divides by the decaying sum of weights at each step.

    Returns:
        A list the same length as ``expr``. The first ``window - 1`` entries are ``None`` (warm-up); thereafter the
        exponentially weighted value.

    Raises:
        ValueError: If ``window < 1``.

    Note:
        Edge-case behavior:

        - **Null** — a leading ``None`` run stays ``None`` and does not consume warm-up budget; an interior ``None``
          yields ``None`` at that position without resetting the average (the weight of the last non-null observation
          decays across the gap and the next non-null value resumes a gap-aware recurrence).
        - **NaN** — a ``nan`` poisons the recursion arithmetically and yields ``nan`` for itself and every subsequent
          non-null position (still masked to ``None`` while inside the warm-up).
        - **window == 1** — the smoothing factor is ``1``, so the recursion reduces to the identity and reproduces the
          input.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window == 1:
        return list(expr)
    if not adjust:
        return seeded_recursive_mean_reference(expr, 2.0 / (window + 1.0), window)

    alpha = 2.0 / (window + 1.0)
    results: list[float | None] = []
    weighted_average: float = 0.0  # sentinel until the first non-null seeds it (gated by ``started``)
    old_weight = 1.0
    started = False
    observed_total = 0  # count of non-null observations seen (drives the warm-up mask)
    for value in expr:
        if not started:
            if value is None:
                results.append(None)
                continue
            weighted_average = value
            old_weight = 1.0
            started = True
            observed_total = 1
            results.append(None if observed_total < window else weighted_average)
            continue
        old_weight *= 1.0 - alpha
        if value is None:
            results.append(None)
            continue
        weighted_average = (old_weight * weighted_average + value) / (old_weight + 1.0)
        old_weight += 1.0
        observed_total += 1
        results.append(None if observed_total < window else weighted_average)
    return results
