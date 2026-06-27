"""
Shared primitives for the naive reference oracles.

Polars-semantics helpers reused across more than one oracle module -- scalar comparisons and the seeded exponential
mean shared by the EMA and Wilder (RMA) oracles. Single-use helpers stay local to their oracle; only genuinely shared
ones live here. The seeded exponential mean is computed as an explicit normalized weighted sum -- a derivation
deliberately unrelated to the production forward recurrence -- so its agreement with the shipped code is genuine
evidence of correctness, not a transcription of the same loop (see ``CORRECTNESS.md``).
"""

import math
from collections.abc import Sequence
from itertools import pairwise


def difference(minuend: float | None, subtrahend: float | None) -> float | None:
    """
    ``minuend - subtrahend``, ``None`` if either is null (``nan`` propagates).
    """
    if minuend is None or subtrahend is None:
        return None
    return minuend - subtrahend


def greater(left: float | None, right: float | None) -> bool | None:
    """
    Polars' ``left > right``: ``None`` if either is null; ``NaN`` compares greater than every finite value (and is not
    greater than itself); otherwise the usual order.
    """
    if left is None or right is None:
        return None
    left_nan = math.isnan(left)
    right_nan = math.isnan(right)
    if left_nan and right_nan:
        return False
    if left_nan:
        return True
    if right_nan:
        return False
    return left > right


def seeded_recursive_mean_reference(
    expr: Sequence[float | None],
    alpha: float,
    window: int,
) -> list[float | None]:
    """
    SMA-of-first-window-seeded exponential mean over a Python list, by an independent closed-form derivation.

    The shared engine of the unadjusted :func:`ema_reference` (``alpha = 2 / (window + 1)``) and :func:`rma_reference`
    (``alpha = 1 / window``). The shipped code carries a renormalized running state forward one bar at a time; this
    oracle instead unrolls that recurrence into an explicit weighted sum -- each output is computed directly from the
    seed and the observation history, with no running state carried between rows -- so the two are genuinely unrelated
    computations of the same published EMA rather than one loop transcribed twice; a transcription error in the forward
    carry cannot hide in the unrolled sum.

    The seed is the simple average of the first ``window`` non-null observations -- the canonical EMA and Wilder
    initialization -- treated as a single point mass at the ``window``-th non-null row. Between consecutive non-null
    observations a row gap of ``d`` (counting nulls) gives a renormalized step ``y <- lambda * y + mu * x`` with
    ``w = (1 - alpha) ** d``, ``lambda = w / (w + alpha)`` and ``mu = alpha / (w + alpha)`` (``lambda + mu == 1``, the
    gap-aware ``ignore_nulls=False`` convention). Unrolling that step over the history, the value at the ``k``-th
    observation after the seed is ``sum_i mu_i * prod_{j>i} lambda_j * x_i + prod_j lambda_j * seed`` -- a convex
    combination, so it reproduces the recurrence exactly while staying well-conditioned. Leading nulls skip the warm-up
    rather than consume it; an interior null yields null at its row while the decay continues across the gap; a ``nan``
    enters the weighted sum and latches, propagating to every later value.
    """
    decay = 1.0 - alpha
    results: list[float | None] = [None] * len(expr)
    observations = [(index, value) for index, value in enumerate(expr) if value is not None]
    if len(observations) < window:
        return results  # the seed never lands; every row stays null
    seed_value = sum(value for _, value in observations[:window]) / window
    tail = observations[window - 1 :]  # tail[0] is the seed row; tail[1:] the later non-null observations
    results[tail[0][0]] = seed_value
    # Per-step renormalization factors between consecutive observations: lambda weights the running mean, mu the new
    # value (lambda + mu == 1), sized from the row gap so a null bridge decays the weight without resetting the mean.
    lambdas: list[float] = []
    mus: list[float] = []
    for previous, current in pairwise(tail):
        weight = decay ** (current[0] - previous[0])
        total = weight + alpha
        lambdas.append(weight / total)
        mus.append(alpha / total)
    for k in range(1, len(tail)):
        # value_k = sum_i mu_i * (prod_{j>i} lambda_j) * x_i + (prod_j lambda_j) * seed -- the recurrence unrolled.
        value = 0.0
        suffix = 1.0
        for i in range(k, 0, -1):
            value += mus[i - 1] * suffix * tail[i][1]
            suffix *= lambdas[i - 1]
        value += suffix * seed_value
        results[tail[k][0]] = value
    return results
