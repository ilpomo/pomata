"""
The single, documented home for the floating-point tolerances the test ladder compares against.

Why a tolerance ladder at all: the implementation evaluates a streaming Polars expression while the naive oracle
recomputes the same quantity a different way (fresh two-pass sums, Python loops). The two agree mathematically but
round differently, and how far they drift depends on what the test does — reproduce a closed-form reference (tightest),
agree over arbitrary fuzzed inputs (looser), or survive a rescaling (looser still). Pinning one constant per tier, with
the rationale here, keeps every indicator's tests using the SAME band for the SAME kind of check instead of re-guessing
a literal inline.

**The headline guarantee.** Every indicator reproduces its independent oracle to ten significant figures (a relative
``1e-10``) on any finite input within a sane dynamic range -- pomata's single precision promise. The bands below are how
that promise is checked per tier; the only looseners are artifact-tolerances (a rescaling that amplifies rounding, a
degenerate-window absolute floor), never a weaker value-agreement. The bound was set by measuring every indicator's
realized agreement and stress-fuzzing the suite across multiple seeds; in practice it is far tighter -- about half the
outputs reproduce the oracle to the last bit, the rest land at the float-64 noise floor.

Tiers (smallest band first):

- **Exact** — the tightest band (``1e-12``, near machine precision at unit magnitude); the :func:`assert_matches`
  default, and the absolute floor for checks of values expected to agree near-exactly, including those passing through
  zero, where the looser reference band would be trivially satisfied.
- **Reference / golden** — the implementation reproduces the closed-form oracle or a frozen golden master to near
  machine precision. ``assert_matches`` already defaults to ``1e-12``; accumulating recurrences (RMA, EWMA, the
  ``map_batches`` kernels) need the slightly looser ``REFERENCE`` pair because order-of-operation differences accrue.
- **Property** — agreement against the oracle over arbitrary Hypothesis-fuzzed inputs, where the worst-case operand
  ordering is hit; held to the SAME ``1e-10`` band as the reference tier, so the ten-significant-figure promise holds on
  fuzzed inputs too, not only on the fixed reference series.
- **Scale** — homogeneity / invariance under a common rescaling, which amplifies rounding by the scale factor.
- **Rolling-vs-oracle** — a one-pass rolling statistic against its recompute-per-window two-pass oracle: the two
  accumulate rounding differently across window slides, so the per-window agreement band is wider than the
  reference tier's; a spec declares it explicitly (``oracle_rel_tol`` / ``oracle_abs_tol``) only where its rolling
  form needs it, and the well-conditioned reductions stay on the default tight band.
- **Rolling-moment absolute floor** — the standardized rolling moments (the skewness / kurtosis pair) against their
  fresh two-pass oracle: the ``m_k / m_2^(k/2)`` quotient amplifies both paths' rounding as the window's spread
  approaches the conditioning floor. Measured worst |impl - oracle| across a 4,000-frame boundary-stressed sweep:
  ``3.7e-11`` (skewness) / ``2.3e-9`` (kurtosis), so the ``1e-7`` floor keeps ~1.5 orders of margin over the worst
  case while staying 10x inside the plain rolling band.
- **Streaming-at-magnitude** — a streaming statistic vs a two-pass oracle at extreme magnitude, where catastrophic
  cancellation forces a loose absolute term; the large-magnitude bespoke tests use :func:`input_scale` to size it to
  the data, or ``ABSOLUTE_TOLERANCE_STREAMING`` for the fixed large-magnitude case.
- **Differential** — agreement with the external C reference (TA-Lib), held to the SAME ``REFERENCE`` band as the
  internal oracle: with the canonical SMA seeding pomata matches TA-Lib bar for bar, so most indicators are compared
  over the whole series, and only a documented minority -- where TA-Lib itself deviates from the indicator's author
  over the warm-up, or carries a long implementation-specific lead-in -- on the converged tail.

The **magnitude-relative factors** size an absolute tolerance to the data as ``input_scale ** degree * factor``: a
streaming or recursive statistic and its two-pass oracle diverge by about ``magnitude ** degree * machine_eps`` on
degenerate inputs, so a fixed absolute tolerance is wrong at the extremes. The factor is set per the statistic's
conditioning -- the worst-case residual it predicts on degenerate windows, which the property tests then hold the
implementation to:

- ``STREAMING_TOLERANCE_FACTOR`` (degree 1, the square-root-amplified streaming statistics): the square root
  amplifies the relative error as the variance approaches zero, worst residual about ``1e-8``, so ``1e-6``; consumed
  through :func:`streaming_abs_tol`.
- ``EXACT_TOLERANCE_FACTOR`` (degree 1, well-conditioned kernels -- recursive (kama, the EMAs), windowed (sma, wma, the
  linear regressions), or stateless (the price transforms)): the impl-vs-oracle residual is at most a few ULP (exactly
  zero for the recursive and stateless kernels; a streaming-vs-two-pass rounding for the windowed means), far below
  ``1e-9``, which is generous slack that still rejects any real coding error.

A scale-INVARIANT output (a cycle period or phase, a 0/1 flag) is ``O(1)`` whatever the input magnitude, so its
tolerance is ABSOLUTE (``ABSOLUTE_TOLERANCE_REFERENCE``), never ``input_scale``-sized -- sizing an ``O(1)`` value to the
input magnitude is meaningless.
"""

import math
from collections.abc import Sequence

RELATIVE_TOLERANCE_EXACT = 1e-12
ABSOLUTE_TOLERANCE_EXACT = 1e-12
RELATIVE_TOLERANCE_REFERENCE = 1e-10
ABSOLUTE_TOLERANCE_REFERENCE = 1e-9
RELATIVE_TOLERANCE_PROPERTY = 1e-10
ABSOLUTE_TOLERANCE_PROPERTY = 1e-9
RELATIVE_TOLERANCE_SCALE = 1e-6
RELATIVE_TOLERANCE_ROLLING_ORACLE = 1e-6
ABSOLUTE_TOLERANCE_ROLLING_ORACLE = 1e-6
ABSOLUTE_TOLERANCE_ROLLING_MOMENT = 1e-7
ABSOLUTE_TOLERANCE_STREAMING = 1e-3
EXACT_TOLERANCE_FACTOR = 1e-9
STREAMING_TOLERANCE_FACTOR = 1e-6


def input_scale(values: Sequence[float | None]) -> float:
    """
    The largest absolute finite value in ``values`` (``1.0`` when there are none).

    Used to size magnitude-relative tolerances. Comparing a streaming Polars rolling statistic against a naive two-pass
    oracle, the two diverge by roughly ``(input magnitude) ** degree * machine_eps`` on degenerate (near-constant)
    windows — e.g. a window of equal values left behind once a large value slides out has a true variance of
    ``0`` but a small non-zero streaming residual. The absolute tolerance must scale with this rather than stay fixed;
    multiply the result by :data:`STREAMING_TOLERANCE_FACTOR`.

    Args:
        values: The raw observations to size the tolerance from; ``None`` and ``float('nan')`` entries are skipped.

    Returns:
        The maximum absolute finite value, or ``1.0`` if ``values`` holds no finite numbers.
    """
    finite = [abs(value) for value in values if isinstance(value, float) and math.isfinite(value)]
    return max(finite, default=1.0)


def streaming_abs_tol(values: Sequence[float | None], *, periods: int = 1) -> float:
    """
    The magnitude-relative absolute tolerance for a streaming statistic, sized to the data's own scale.

    Sizes the band to the data via :func:`input_scale` and :data:`STREAMING_TOLERANCE_FACTOR`. Pass ``periods`` to scale
    it by ``sqrt(periods)`` for an annualized output (volatility, downside deviation), matching that metric's own
    sqrt-of-time scaling; the default ``periods=1`` leaves it on the per-period (own) scale used by the quantile metrics
    (value-at-risk, conditional value-at-risk).

    Args:
        values: The raw observations to size the tolerance from.
        periods: The annualization factor the metric applies; the band is scaled by ``sqrt(periods)``.

    Returns:
        The magnitude-relative absolute tolerance.
    """
    return input_scale(values) * math.sqrt(periods) * STREAMING_TOLERANCE_FACTOR
