"""
Timing helpers and the derived sizing constants for the opt-in ``benchmark`` tier (``tests/test_benchmark.py``).

The scaling guard's numbers are derived, not chosen. Cost model per function: ``t(n) = c * n**k + h``, where ``k``
is the spec's declared ``cost_degree`` and ``h`` is the function's own fixed per-call cost (expression dispatch,
column count, any ``map_batches`` round-trip), measured on a frame far below every window. The estimator is
:func:`fastest_eval` (the min of three), whose residual multiplicative noise on the serial nightly runner is
bounded at ``eps = 25%``; a logarithmic factor inflates a one-decade ratio by at most ``L = log(10n)/log(n) <= 4/3``
at the ladder's smallest base (1,000 rows).

The two inequalities the constants solve, for a base size certified at signal ``s = c * base**k / h >= 3``
(guaranteed by the ``t_base >= 4h`` stop rule, :data:`SCALING_OVERHEAD_MULTIPLE`):

- **no false alarm** — an honest degree-``k`` kernel, log factor included, measures at most
  ``10**k * L * (1 + eps) / (1 - eps) = 2.22 * 10**k``, below the bound;
- **detection** — a degree-``k + 1`` regression measures at least
  ``10**(k+1) * s / (s + 1) * (1 - eps) / (1 + eps) = 4.5 * 10**k`` at the worst certified ``s = 3``, above it.

:func:`scaling_threshold` returns ``3.0 * 10**k`` — inside the ``(2.22, 4.5) * 10**k`` window with a >= 35% margin
on both sides. The assert adds the model's own constant term (``SCALING_OVERHEAD_MULTIPLE * h``): negligible when
the base is certified, and the graceful degradation for a kernel so cheap that no ladder size clears the stop rule
(there the decade is overhead-bound and a genuine regression still explodes the absolute time). Change ``eps`` and
re-derive; no constant here is free.
"""

import time
from collections.abc import Callable

import polars as pl

# The stop rule of the certifying walk: the base size must cost at least this multiple of the function's own fixed
# per-call cost, so the scaling signal satisfies s = c * base**k / h >= SCALING_OVERHEAD_MULTIPLE - 1 = 3.
SCALING_OVERHEAD_MULTIPLE = 4.0


def scaling_threshold(cost_degree: int) -> float:
    """
    The one-decade time-ratio bound for a kernel of the declared polynomial cost degree.

    ``3.0 * 10**k`` sits between the worst honest degree-``k`` measurement (``2.22 * 10**k``, log factor and
    estimator noise included) and the weakest degree-``k + 1`` regression (``4.5 * 10**k``) — see the module
    docstring for the derivation.

    Args:
        cost_degree: The spec's declared polynomial cost degree in the row count.

    Returns:
        The bound a one-decade ratio must stay under.
    """
    return 3.0 * 10.0**cost_degree


def fastest_eval(frame: pl.DataFrame, build: Callable[[], pl.Expr]) -> float:
    """
    The fastest of three evaluations of ``build`` over ``frame``, in seconds (the min is robust to load spikes).

    Args:
        frame: The pre-built frame to evaluate against (each caller sizes its own).
        build: A no-argument factory returning the ``pl.Expr`` to time.

    Returns:
        The minimum wall-clock time, in seconds, across three ``select`` evaluations (after one warm-up pass).
    """
    expr = build().alias("y")
    frame.select(expr)  # warm any caches before timing

    def once() -> float:
        start = time.perf_counter()
        frame.select(expr)
        return time.perf_counter() - start

    return min(once() for _ in range(3))
