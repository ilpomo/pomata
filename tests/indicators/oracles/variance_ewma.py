"""
Naive reference oracle for ``pomata.indicators.variance_ewma``.
"""

import math
from collections.abc import Sequence


def reference_variance_ewma(
    expr: Sequence[float | None],
    window: int,
    *,
    adjust: bool = False,
    bias: bool = True,
) -> list[float | None]:
    r"""
    Naive Exponentially-Weighted Variance over a Python list.

    A direct two-pass evaluation of the published weighted-variance definition, recomputed from scratch as the oracle
    for :func:`pomata.indicators.variance_ewma`. For each output row ``t`` it assembles an explicit per-observation
    weight vector over the whole prefix ``x_0 .. x_t``, takes the weighted mean in a first pass, then the weighted sum
    of squared deviations in a second pass — never the running-covariance recurrence the engine evaluates internally.
    With ``alpha = 2 / (window + 1)`` and ``lag`` counting *every* row back to the start (a ``None`` still ages the lag
    but enters no sum), the weight of the value at ``lag`` is

    - ``adjust=True``  : ``w = (1 - alpha) ** lag`` for every observation;
    - ``adjust=False`` : a forward product of per-step blend fractions. Walking the non-null observations from oldest to
      newest, the seed carries blend ``1`` and each later observation blends in with ``beta = alpha / ((1 - alpha) **
      (gap + 1) + alpha)``, where ``gap`` is the number of ``None`` rows since the previous observation; the weight
      of an observation is its own ``beta`` times the product of ``(1 - beta)`` over every newer observation. With no
      gaps this reduces to the textbook ``alpha (1 - alpha) ** lag`` (oldest ``(1 - alpha) ** lag``), and across a
      ``None`` run it reproduces the engine's ``ignore_nulls=False`` re-weighting.

    The weighted mean is ``xbar = sum_k w_k x_k / sum_k w_k`` and, with ``S1 = sum_k w_k`` and ``S2 = sum_k w_k ** 2``,

    .. math::

        \mathrm{VAR}_{\text{bias}} = \frac{\sum_k w_k (x_k - \bar{x})^2}{S_1}, \qquad
        \mathrm{VAR}_{\text{unbiased}} = \frac{S_1}{S_1^2 - S_2}\,\sum_k w_k (x_k - \bar{x})^2.

    Because the sums run over the full prefix, a ``NaN`` anywhere in scope poisons every later row (the engine's latch
    behavior) and an interior ``None`` ages the weights of everything before it without contributing a term.

    Args:
        expr: The input series (may contain ``None`` and ``float('nan')``).
        window: Span of the exponential weighting, mapped to ``alpha = 2 / (window + 1)``. Must be ``>= 2``.
        adjust: When ``False`` (default) use the recursive (blend-fraction) weighting; when ``True`` the finite-window
            bias-corrected weighting ``(1 - alpha) ** lag``.
        bias: When ``True`` (default) the population variance; when ``False`` the unbiased sample variance.

    Returns:
        A list the same length as ``expr``: the exponentially-weighted variance, ``None`` until ``window`` non-null
        observations have been seen and at each ``None`` row, ``nan`` once a ``nan`` has entered the weighted sums.

    Raises:
        ValueError: If ``window < 2``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    alpha = 2.0 / (window + 1.0)
    one_minus_alpha = 1.0 - alpha

    results: list[float | None] = []
    observed = 0
    for index in range(len(expr)):
        if expr[index] is None:
            results.append(None)
            continue
        observed += 1
        if observed < window:
            results.append(None)
            continue
        weights, values = _prefix_weights(expr, index, alpha, one_minus_alpha, adjust=adjust)
        sum_weight = math.fsum(weights)
        sum_weight_squared = math.fsum(weight * weight for weight in weights)
        mean = math.fsum(weight * value for weight, value in zip(weights, values, strict=True)) / sum_weight
        squared = math.fsum(
            weight * (value - mean) * (value - mean) for weight, value in zip(weights, values, strict=True)
        )
        if bias:
            results.append(squared / sum_weight)
        else:
            denominator = sum_weight * sum_weight - sum_weight_squared
            results.append((sum_weight / denominator) * squared if denominator > 0 else math.nan)
    return results


def _prefix_weights(
    expr: Sequence[float | None],
    index: int,
    alpha: float,
    one_minus_alpha: float,
    *,
    adjust: bool,
) -> tuple[list[float], list[float]]:
    """
    Build the explicit weight vector and matching value vector for the prefix ``x_0 .. x_index``.

    With ``adjust=True`` each non-null value at ``lag`` (counting ``None`` rows) gets ``(1 - alpha) ** lag``. With
    ``adjust=False`` the weights are a forward product of blend fractions over the non-null observations: the seed
    blends at ``1``, each later observation at ``beta = alpha / ((1 - alpha) ** (gap + 1) + alpha)`` for a preceding
    ``gap`` of ``None`` rows, and the weight is ``beta`` times the product of ``(1 - beta)`` over every newer
    observation. The two vectors share the same order (oldest first); only non-null values appear.
    """
    if adjust:
        weights: list[float] = []
        values: list[float] = []
        for lag in range(index + 1):
            value = expr[index - lag]
            if value is None:
                continue
            weights.append(one_minus_alpha**lag)
            values.append(value)
        weights.reverse()
        values.reverse()
        return weights, values

    observations: list[float] = []
    blends: list[float] = []
    gap = 0
    for position in range(index + 1):
        value = expr[position]
        if value is None:
            gap += 1
            continue
        blend = 1.0 if not observations else alpha / (one_minus_alpha ** (gap + 1) + alpha)
        observations.append(value)
        blends.append(blend)
        gap = 0
    weights = []
    for current, blend in enumerate(blends):
        weight = blend
        for newer in range(current + 1, len(blends)):
            weight *= 1.0 - blends[newer]
        weights.append(weight)
    return weights, observations
