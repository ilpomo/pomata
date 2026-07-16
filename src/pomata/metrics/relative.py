r"""
Benchmark-relative metrics — performance and risk measured against a second, benchmark return series.

Every metric here is two-input: it reads a portfolio (or strategy) return series and an aligned benchmark return series,
both as fractions, and reduces the pair to a single value — except the four ``*_rolling`` twins, which emit a value
per row over a trailing window. The two series are treated as **pairwise-complete**: an
observation contributes only where BOTH legs are present, so a ``null`` in either leg drops that pair; among the
retained pairs a ``NaN`` in either leg poisons the result to ``NaN``. This is the composing layer for cross-sectional
analytics: ``beta`` is the shared regression slope reused by ``alpha`` and ``treynor_ratio``, and
``modigliani_risk_adjusted_performance`` is built on the single-input :func:`sharpe_ratio` and
:func:`volatility`, imported from their specific theme modules so the theme dependency graph stays
acyclic.
"""

import math
from typing import Final

import polars as pl

from pomata._expr import (
    float64_expr,
    per_period_rate,
    rolling_mean_exact,
    validate_finite,
    validate_periods_per_year,
    validate_window,
)
from pomata.metrics.ratio import sharpe_ratio
from pomata.metrics.risk import volatility, volatility_rolling

__all__ = (
    "alpha",
    "alpha_rolling",
    "beta",
    "beta_rolling",
    "capture_downside_ratio",
    "capture_ratio",
    "capture_upside_ratio",
    "information_ratio",
    "information_ratio_rolling",
    "modigliani_risk_adjusted_performance",
    "treynor_ratio",
    "treynor_ratio_rolling",
)

# The smallest number of complete (both-legs-present) pairs for a covariance, variance, or sample standard deviation to
# be defined; with fewer the metric is reported as ``null`` (insufficient data), taking precedence over NaN poisoning.
_MINIMUM_PAIRED_OBSERVATIONS: Final = 2


def _paired(
    returns: pl.Expr,
    benchmark: pl.Expr,
) -> tuple[pl.Expr, pl.Expr]:
    """
    Filter a return/benchmark pair to its pairwise-complete rows, returning the two filtered legs in input order.

    The shared masking step every benchmark-relative reducing metric applies before forming a covariance, capture, or
    excess: an observation is kept only where BOTH legs are present, so a ``null`` in either leg drops that pair.
    """
    both_present = returns.is_not_null() & benchmark.is_not_null()
    return returns.filter(both_present), benchmark.filter(both_present)


def _rolling_raw_beta(
    returns: pl.Expr,
    benchmark: pl.Expr,
    window: int,
) -> pl.Expr:
    """
    The population regression slope over each trailing window, from rolling raw moments.

    ``cov / var`` with ``cov = E[rb] - E[r]E[b]`` (rolling means, ``min_samples=window``) and ``var`` the native
    ``rolling_var(ddof=0)``, which is mean-centered: a near-constant benchmark yields a tiny-but-correct variance, not
    the wrong-signed residue the one-pass ``E[b^2] - E[b]^2`` leaves. The population ``ddof = 0`` cancels between the
    two. The shared core of :func:`beta_rolling`, :func:`alpha_rolling`, and :func:`treynor_ratio_rolling`. An
    incomplete (``null``-containing) window yields ``null``; an exactly-constant (zero-variance) finite benchmark
    window, detected via ``rolling_max == rolling_min``, yields ``NaN`` (an undefined slope on a flat regressor).
    """
    mean_returns = returns.rolling_mean(window, min_samples=window)
    mean_benchmark = benchmark.rolling_mean(window, min_samples=window)
    covariance = (returns * benchmark).rolling_mean(window, min_samples=window) - mean_returns * mean_benchmark
    variance = benchmark.rolling_var(window, ddof=0, min_samples=window)
    benchmark_max = benchmark.rolling_max(window, min_samples=window)
    benchmark_min = benchmark.rolling_min(window, min_samples=window)
    # ``is_finite`` keeps the guard off a NaN-poisoned window (Polars treats ``NaN == NaN`` as true), so only a genuine
    # finite flat window fires it; a NaN window falls through to ``cov / var``, which already propagates NaN.
    # ``covariance.is_not_null()`` keeps it off an incomplete (null-containing) window, which stays ``null`` under the
    # pairwise-complete contract rather than being turned into ``NaN`` by the flat-benchmark branch.
    is_flat = (benchmark_max == benchmark_min) & benchmark_max.is_finite()
    return pl.when(is_flat & covariance.is_not_null()).then(float("nan")).otherwise(covariance / variance)


def _raw_beta(
    returns: pl.Expr,
    benchmark: pl.Expr,
) -> pl.Expr:
    """
    The population regression slope ``cov(returns, benchmark) / var(benchmark)`` over complete pairs.

    The shared core of :func:`beta`, :func:`alpha`, and :func:`treynor_ratio`; the population ``ddof = 0`` in both the
    covariance and the variance cancels, so the slope is independent of the degrees-of-freedom convention. The caller
    supplies pairwise-complete (both-legs-present) inputs and applies the insufficient-data guard. A constant
    (zero-variance) benchmark is detected exactly via ``max == min`` and reported as ``NaN``, because the ``cov / var``
    floating-point cancellation cannot be relied on to surface the ``0 / 0``.
    """
    slope = pl.cov(returns, benchmark, ddof=0) / benchmark.var(ddof=0)
    # ``is_finite`` keeps the guard off a NaN-poisoned input (Polars treats ``NaN == NaN`` as true), so only a genuine
    # finite flat benchmark fires it; a NaN benchmark falls through to ``slope``, which already propagates NaN.
    benchmark_max = benchmark.max()
    is_flat = (benchmark_max == benchmark.min()) & benchmark_max.is_finite()
    return pl.when(is_flat).then(float("nan")).otherwise(slope)


def _capture(
    returns: pl.Expr,
    benchmark: pl.Expr,
    periods_per_year: int,
    *,
    upside: bool,
) -> pl.Expr:
    """
    The shared up/down capture ratio: annualized portfolio return over annualized benchmark return on selected periods.

    Selects the periods where the benchmark is positive (``upside``) or negative, then forms the geometric annualized
    return of each leg over the count of selected periods and returns their ratio (the Morningstar construction). The
    inputs are pairwise-complete; a ``NaN`` in either leg poisons to ``NaN`` (a benchmark ``NaN`` would otherwise escape
    the sign filter), with no complete pairs or no selected periods reported as ``null``.
    """
    selected = benchmark > 0.0 if upside else benchmark < 0.0
    returns_leg = returns.filter(selected)
    benchmark_leg = benchmark.filter(selected)
    count = returns_leg.len()
    portfolio_growth = (1.0 + returns_leg).product() ** (periods_per_year / count) - 1.0
    benchmark_growth = (1.0 + benchmark_leg).product() ** (periods_per_year / count) - 1.0
    poisoned = (returns.is_nan() | benchmark.is_nan()).any()
    # Domain: the geometric (compounded) growth is defined only while every selected gross return 1 + r stays
    # positive — a return at or below -1 wipes the leg out (or worse), and the fractional power of a non-positive
    # product is meaningless (parity-dependent sign, NaN on most inputs). Out of domain is a loud NaN, never a
    # plausible wrong number.
    out_of_domain = ((1.0 + returns_leg) <= 0.0).any() | ((1.0 + benchmark_leg) <= 0.0).any()
    return (
        pl.when(poisoned)
        .then(float("nan"))
        .when(out_of_domain)
        .then(float("nan"))
        .when(count < 1)
        .then(None)
        .otherwise(portfolio_growth / benchmark_growth)
    )


def alpha(
    returns: pl.Expr,
    benchmark: pl.Expr,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> pl.Expr:
    r"""
    Jensen's Alpha, the annualized excess return a portfolio earns beyond its benchmark-explained (CAPM) return.

    The per-period average of the realized return minus the return the Capital Asset Pricing Model predicts from the
    portfolio's :func:`beta`, compounded to a yearly figure:

    .. math::

        \alpha = \left(1 + \overline{(r_i - r_f) - \beta\,(b_i - r_f)}\right)^{P} - 1,

    where :math:`r_i` and :math:`b_i` are the portfolio and benchmark returns, :math:`\beta` the raw regression slope
    (:func:`beta`), :math:`P` is ``periods_per_year``, and the per-period risk-free rate is the geometric conversion
    :math:`r_f = (1 + \texttt{risk\_free\_rate})^{1/P} - 1`. A positive alpha is value added beyond market exposure. The
    annualization is geometric (alpha is a compounding return figure); :func:`treynor_ratio`, a ratio numerator,
    annualizes its excess arithmetically -- a deliberate convention difference across the relative family.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate, converted to a per-period rate geometrically (default ``0.0``).
            Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + risk_free_rate >= 0``).

    Returns:
        A single ``Float64`` value: the annualized Jensen's alpha (one value in ``select``, one per group under
        ``.over``). ``null`` when fewer than two complete pairs are present.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite or is ``< -1``.

    Note:
        **Correctness**

        The result is checked against an independent reference oracle on every input, and every edge case (missing data
        and boundaries) is given a defined behavior.

        **Edge-case behavior**

        - **Null** — an observation is used only where both legs are present; a ``null`` in either drops that pair.
        - **NaN** — a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``.
        - **Insufficient sample** — fewer than two complete pairs leaves the regression slope undefined, so the result
          is ``null``.
        - **Degenerate denominator** — a zero-variance benchmark makes :func:`beta` ``NaN`` (a ``0 / 0``), which
          propagates here.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`beta`: The regression slope this corrects the return for.
        - :func:`treynor_ratio`: The excess return per unit of the same systematic risk.
        - :func:`alpha_rolling`: The same measure over a trailing window.

    References:
        - Jensen, M. C. (1968). "The Performance of Mutual Funds in the Period 1945-1964." *The Journal of Finance*,
          23(2), 389-416.
        - https://doi.org/10.1111/j.1540-6261.1968.tb00815.x
        - https://en.wikipedia.org/wiki/Jensen%27s_alpha

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import alpha
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004],
        ...     }
        ... )
        >>> frame.select(alpha(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).round(4)).item()
        0.0233

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 6 + ["B"] * 6,
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.01, 0.025, -0.015, 0.008, -0.005, 0.012],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, 0.012, 0.02, -0.01, 0.006, -0.004, 0.01],
        ...     }
        ... )
        >>> reduced = alpha(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).over("ticker").round(4)
        >>> frame.select(alpha=reduced)["alpha"].unique().sort().to_list()
        [-0.2798, 0.0233]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [None, 0.02, 0.03, float("nan"), 0.015, 0.005],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004],
        ...     }
        ... )
        >>> frame.select(alpha(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).round(4)).item()
        nan

        **Insufficient sample** — a single complete pair yields ``null``, since the regression slope needs two
        observations:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.05],
        ...         "benchmark": [0.04],
        ...     }
        ... )
        >>> frame.select(alpha=alpha(pl.col("returns"), pl.col("benchmark"), periods_per_year=252))["alpha"].to_list()
        [None]

        **Degenerate denominator** — a constant benchmark makes the embedded beta ``NaN``, which propagates to the
        result:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.01, -0.02, 0.03],
        ...         "benchmark": [0.1, 0.1, 0.1],
        ...     }
        ... )
        >>> frame.select(alpha(pl.col("returns"), pl.col("benchmark"), periods_per_year=252)).item()
        nan
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    returns_paired, benchmark_paired = _paired(returns, benchmark)
    rf_period = per_period_rate(risk_free_rate, periods_per_year, name="risk_free_rate")
    slope = _raw_beta(returns_paired, benchmark_paired)
    excess_leg = (returns_paired - rf_period) - slope * (benchmark_paired - rf_period)
    annualized = (1.0 + excess_leg.mean()) ** periods_per_year - 1.0
    return pl.when(returns_paired.len() < _MINIMUM_PAIRED_OBSERVATIONS).then(None).otherwise(annualized).name.keep()


def alpha_rolling(
    returns: pl.Expr,
    benchmark: pl.Expr,
    window: int,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> pl.Expr:
    r"""
    Rolling Jensen's Alpha over a window — the windowed twin of :func:`alpha`.

    The annualized return beyond the CAPM-predicted return, computed over each trailing window:

    .. math::

        \alpha_t = \left(1 + (\bar{r}_t - r_f) - \beta_t\,(\bar{b}_t - r_f)\right)^{P} - 1, \qquad n = \text{window},

    where :math:`\bar{r}_t`, :math:`\bar{b}_t` are the window means, :math:`\beta_t` the rolling :func:`beta_rolling`
    slope, :math:`P` is ``periods_per_year``, and :math:`r_f = (1 + \texttt{risk\_free\_rate})^{1/P} - 1`.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        window: Number of observations in the moving window. Must be ``>= 2``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate, converted to a per-period rate geometrically (default ``0.0``).
            Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + risk_free_rate >= 0``).

    Returns:
        The rolling Jensen's alpha for each row, the same length as the input. The first ``window - 1`` rows are
        ``null`` (warm-up): the window must hold ``window`` complete pairs before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``, ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite or is ``< -1``.

    Note:
        **Correctness**

        Each window matches an independent reference oracle (the reducing :func:`alpha` over the window).

        **Edge-case behavior**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **Degenerate denominator** — a zero-variance window benchmark makes the slope ``NaN`` (a ``0 / 0``), which
          propagates here.
        - **Stability** — a near-flat (non-bit-identical) benchmark window sits at the float-conditioning limit the
          documentation's *Correctness* page documents: the one-pass rolling covariance behind the embedded slope and an
          exact two-pass recomputation can round a vanishing benchmark variance apart without bound there. The bit-flat
          window is guarded exactly (``NaN``); real market windows are far from the regime.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`alpha`: The whole-series reducing form.
        - :func:`beta_rolling`: The rolling slope this corrects the return for.
        - :func:`treynor_ratio_rolling`: The rolling excess per unit of the same systematic risk.

    References:
        - Jensen, M. C. (1968). "The Performance of Mutual Funds in the Period 1945-1964." *The Journal of Finance*,
          23(2), 389-416.
        - https://doi.org/10.1111/j.1540-6261.1968.tb00815.x
        - https://en.wikipedia.org/wiki/Jensen%27s_alpha

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import alpha_rolling
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.01, 0.02],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018],
        ...     }
        ... )
        >>> frame.select(
        ...     alpha_rolling(pl.col("returns"), pl.col("benchmark"), 4, periods_per_year=252).round(4)
        ... ).to_series().to_list()
        [None, None, None, -0.0864, -0.0096, -0.0227, 0.4932, 0.7998]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 6 + ["B"] * 6,
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.01, 0.025, -0.015, 0.008, -0.005, 0.012],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, 0.012, 0.02, -0.01, 0.006, -0.004, 0.01],
        ...     }
        ... )
        >>> expr = (
        ...     alpha_rolling(pl.col("returns"), pl.col("benchmark"), 4, periods_per_year=252).over("ticker").round(4)
        ... )
        >>> frame.with_columns(alpha_rolling=expr)["alpha_rolling"].to_list()
        [None, None, None, -0.0864, -0.0096, -0.0227, None, None, None, -0.3956, -0.1613, -0.1561]

        A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make the handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [None, float("nan"), 0.03, -0.02, 0.015, 0.005, -0.01, 0.02],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018],
        ...     }
        ... )
        >>> frame.select(
        ...     alpha_rolling(pl.col("returns"), pl.col("benchmark"), 4, periods_per_year=252).round(4)
        ... ).to_series().to_list()
        [None, None, None, None, nan, -0.0227, 0.4932, 0.7998]

        **Degenerate denominator** — a null in the returns leg wins over the constant-benchmark ``NaN`` branch on
        incomplete windows, so the result stays ``null`` until the window is complete, then ``NaN``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.02, None, 0.03, 0.01, 0.02],
        ...         "benchmark": [0.1, 0.1, 0.1, 0.1, 0.1],
        ...     }
        ... )
        >>> expr = alpha_rolling(pl.col("returns"), pl.col("benchmark"), window=3, periods_per_year=252)
        >>> frame.select(alpha_rolling=expr)["alpha_rolling"].to_list()
        [None, None, None, None, nan]
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_window(window, minimum=2)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    rf_period = per_period_rate(risk_free_rate, periods_per_year, name="risk_free_rate")
    slope = _rolling_raw_beta(returns, benchmark, window)
    mean_returns = returns.rolling_mean(window, min_samples=window)
    mean_benchmark = benchmark.rolling_mean(window, min_samples=window)
    alpha_period = (mean_returns - rf_period) - slope * (mean_benchmark - rf_period)
    return ((1.0 + alpha_period) ** periods_per_year - 1.0).name.keep()


def beta(
    returns: pl.Expr,
    benchmark: pl.Expr,
) -> pl.Expr:
    r"""
    Beta, the sensitivity of a portfolio's return to its benchmark (its systematic, non-diversifiable risk).

    The slope of the regression of the portfolio return on the benchmark return -- the population covariance over the
    benchmark variance:

    .. math::

        \beta = \frac{\operatorname{cov}(r, b)}{\operatorname{var}(b)},

    where :math:`r` is the portfolio return and :math:`b` the benchmark return. A beta of one moves with the benchmark,
    above one amplifies it, and below one dampens it. The degrees-of-freedom convention cancels between numerator and
    denominator, so the result is the same population or sample slope.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.

    Returns:
        A single ``Float64`` value: the regression slope (one value in ``select``, one per group under ``.over``).
        ``null`` when fewer than two complete pairs are present.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness**

        The result is checked against an independent reference oracle on every input, and every edge case (missing data
        and boundaries) is given a defined behavior.

        **Edge-case behavior**

        - **Null** — an observation is used only where both legs are present; a ``null`` in either drops that pair.
        - **NaN** — a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``.
        - **Insufficient sample** — fewer than two complete pairs leaves the regression slope undefined, so the result
          is ``null``.
        - **Degenerate denominator** — a zero-variance benchmark leaves the slope undefined, so the result is a
          ``0 / 0``, i.e. ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`alpha`: The benchmark-relative return that nets out beta-explained performance.
        - :func:`treynor_ratio`: The excess return per unit of this systematic risk.
        - :func:`beta_rolling`: The same slope over a trailing window.

    References:
        - Sharpe, W. F. (1964). "Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk." *The
          Journal of Finance*, 19(3), 425-442.
        - https://doi.org/10.1111/j.1540-6261.1964.tb02865.x
        - https://en.wikipedia.org/wiki/Beta_%28finance%29

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import beta
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004],
        ...     }
        ... )
        >>> frame.select(beta(pl.col("returns"), pl.col("benchmark")).round(4)).item()
        1.2726

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 6 + ["B"] * 6,
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.01, 0.025, -0.015, 0.008, -0.005, 0.012],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, 0.012, 0.02, -0.01, 0.006, -0.004, 0.01],
        ...     }
        ... )
        >>> reduced = beta(pl.col("returns"), pl.col("benchmark")).over("ticker").round(4)
        >>> frame.select(beta=reduced)["beta"].unique().sort().to_list()
        [1.2591, 1.2726]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [None, 0.02, 0.03, float("nan"), 0.015, 0.005],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004],
        ...     }
        ... )
        >>> frame.select(beta(pl.col("returns"), pl.col("benchmark")).round(4)).item()
        nan

        **Insufficient sample** — one complete pair has no regression slope, since at least two observations are needed,
        so the result is ``null``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.05],
        ...         "benchmark": [0.04],
        ...     }
        ... )
        >>> frame.select(beta=beta(pl.col("returns"), pl.col("benchmark")))["beta"].to_list()
        [None]

        **Degenerate denominator** — a constant (zero-variance) benchmark leaves the slope undefined, the ``0 / 0``
        case, so the result is ``NaN``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.01, -0.02, 0.03],
        ...         "benchmark": [0.1, 0.1, 0.1],
        ...     }
        ... )
        >>> frame.select(beta(pl.col("returns"), pl.col("benchmark"))).item()
        nan
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    returns_paired, benchmark_paired = _paired(returns, benchmark)
    return (
        pl.when(returns_paired.len() < _MINIMUM_PAIRED_OBSERVATIONS)
        .then(None)
        .otherwise(_raw_beta(returns_paired, benchmark_paired))
        .name.keep()
    )


def beta_rolling(
    returns: pl.Expr,
    benchmark: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Rolling Beta over a window — the windowed twin of :func:`beta`.

    The slope of the regression of the portfolio return on the benchmark return over each trailing window:

    .. math::

        \beta_t = \frac{\operatorname{cov}_t(r, b)}{\operatorname{var}_t(b)}, \qquad n = \text{window},

    with the covariance and variance taken over the window. The degrees-of-freedom convention cancels.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        window: Number of observations in the moving window. Must be ``>= 2``.

    Returns:
        The rolling regression slope for each row, the same length as the input. The first ``window - 1`` rows are
        ``null`` (warm-up): the window must hold ``window`` complete pairs before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``.

    Note:
        **Correctness**

        Each window matches an independent reference oracle (the reducing :func:`beta` over the window).

        **Edge-case behavior**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **Degenerate denominator** — a zero-variance window benchmark leaves the slope undefined, so the result is a
          ``0 / 0``, i.e. ``NaN``.
        - **Stability** — a near-flat (non-bit-identical) benchmark window sits at the float-conditioning limit the
          documentation's *Correctness* page documents: the one-pass rolling covariance and an exact two-pass
          recomputation can round a vanishing denominator apart without bound there. The bit-flat window is guarded
          exactly (``NaN``); real market windows are far from the regime.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`beta`: The whole-series reducing form.
        - :func:`alpha_rolling`: The benchmark-relative return built on this slope.
        - :func:`treynor_ratio_rolling`: The excess return per unit of this systematic risk.

    References:
        - Sharpe, W. F. (1964). "Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk." *The
          Journal of Finance*, 19(3), 425-442.
        - https://doi.org/10.1111/j.1540-6261.1964.tb02865.x
        - https://en.wikipedia.org/wiki/Beta_%28finance%29

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import beta_rolling
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.01, 0.02],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018],
        ...     }
        ... )
        >>> frame.select(beta_rolling(pl.col("returns"), pl.col("benchmark"), 4).round(4)).to_series().to_list()
        [None, None, None, 1.2608, 1.2628, 1.2652, 1.2592, 1.0331]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 6 + ["B"] * 6,
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.01, 0.025, -0.015, 0.008, -0.005, 0.012],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, 0.012, 0.02, -0.01, 0.006, -0.004, 0.01],
        ...     }
        ... )
        >>> expr = beta_rolling(pl.col("returns"), pl.col("benchmark"), 4).over("ticker").round(4)
        >>> frame.with_columns(beta_rolling=expr)["beta_rolling"].to_list()
        [None, None, None, 1.2608, 1.2628, 1.2652, None, None, None, 1.2851, 1.3159, 1.3466]

        A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make the handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [None, float("nan"), 0.03, -0.02, 0.015, 0.005, -0.01, 0.02],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018],
        ...     }
        ... )
        >>> frame.select(beta_rolling(pl.col("returns"), pl.col("benchmark"), 4).round(4)).to_series().to_list()
        [None, None, None, None, nan, 1.2652, 1.2592, 1.0331]

        **Degenerate denominator** — a window holding a ``null`` yields ``null`` under the pairwise-complete gate rather
        than the flat-benchmark ``NaN``, until the window clears and reports ``NaN``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.02, None, 0.03, 0.01, 0.02],
        ...         "benchmark": [0.1, 0.1, 0.1, 0.1, 0.1],
        ...     }
        ... )
        >>> expr = beta_rolling(pl.col("returns"), pl.col("benchmark"), window=3)
        >>> frame.select(beta_rolling=expr)["beta_rolling"].to_list()
        [None, None, None, None, nan]
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_window(window, minimum=2)
    return _rolling_raw_beta(returns, benchmark, window).name.keep()


def capture_downside_ratio(
    returns: pl.Expr,
    benchmark: pl.Expr,
    *,
    periods_per_year: int,
) -> pl.Expr:
    r"""
    Downside Capture Ratio, how much of the benchmark's loss a portfolio participated in during down markets.

    The portfolio's annualized return over the benchmark's annualized return, computed over only the periods where the
    benchmark fell (the Morningstar geometric construction):

    .. math::

        \mathrm{DCR} = \frac{\left(\prod_{b_i < 0}(1 + r_i)\right)^{P/n_-} - 1}
        {\left(\prod_{b_i < 0}(1 + b_i)\right)^{P/n_-} - 1},

    where the products run over the :math:`n_-` periods with a negative benchmark return and :math:`P` is
    ``periods_per_year``. A value below one means the portfolio lost less than the benchmark in down markets (good).

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.

    Returns:
        A single ``Float64`` value: the downside capture ratio (one value in ``select``, one per group under ``.over``).
        ``null`` when there are no complete pairs or no down-market periods.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``.

    Note:
        **Correctness**

        The result is checked against an independent reference oracle on every input, and every edge case (missing data
        and boundaries) is given a defined behavior.

        **Edge-case behavior**

        - **Null** — an observation is used only where both legs are present; a ``null`` in either drops that pair.
        - **NaN** — a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``.
        - **Domain** — the compounded (geometric) leg growth is defined only while every selected gross return
          ``1 + r`` stays positive; a selected return at or below ``-1`` wipes that leg out of domain, so the result is
          a loud ``NaN`` — never a plausible wrong number.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`capture_upside_ratio`: The up-market counterpart.
        - :func:`capture_ratio`: Their ratio, an overall asymmetry measure.
        - :func:`beta`: The symmetric benchmark sensitivity this asymmetric down-market measure refines.

    References:
        - Morningstar. "Upside/Downside Capture Ratio" (methodology).

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import capture_downside_ratio
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004],
        ...     }
        ... )
        >>> frame.select(
        ...     capture_downside_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).round(4)
        ... ).item()
        1.0339

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 6 + ["B"] * 6,
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.01, 0.025, -0.015, 0.008, -0.005, 0.012],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, 0.012, 0.02, -0.01, 0.006, -0.004, 0.01],
        ...     }
        ... )
        >>> reduced = (
        ...     capture_downside_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252)
        ...     .over("ticker")
        ...     .round(4)
        ... )
        >>> frame.select(capture_downside_ratio=reduced)["capture_downside_ratio"].unique().sort().to_list()
        [1.0339, 1.1095]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [None, 0.02, 0.03, float("nan"), 0.015, 0.005],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004],
        ...     }
        ... )
        >>> frame.select(
        ...     capture_downside_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).round(4)
        ... ).item()
        nan

        **Domain** — a selected portfolio return at or below ``-1`` wipes that leg out of the geometric-growth domain,
        so the result is a loud ``NaN``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.02, -1.5, 0.01],
        ...         "benchmark": [-0.01, -0.02, -0.03],
        ...     }
        ... )
        >>> frame.select(capture_downside_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252)).item()
        nan
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_periods_per_year(periods_per_year)
    returns_paired, benchmark_paired = _paired(returns, benchmark)
    return _capture(returns_paired, benchmark_paired, periods_per_year, upside=False).name.keep()


def capture_ratio(
    returns: pl.Expr,
    benchmark: pl.Expr,
    *,
    periods_per_year: int,
) -> pl.Expr:
    r"""
    Capture Ratio, the ratio of upside capture to downside capture (a single market-asymmetry score).

    The :func:`capture_upside_ratio` divided by the :func:`capture_downside_ratio` -- a value above one means the
    portfolio captures more of the benchmark's gains than of its losses:

    .. math::

        \mathrm{CR} = \frac{\mathrm{UCR}}{\mathrm{DCR}},

    where :math:`\mathrm{UCR}` and :math:`\mathrm{DCR}` are the up- and down-market capture ratios.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.

    Returns:
        A single ``Float64`` value: the capture ratio (one value in ``select``, one per group under ``.over``).
        ``null`` when either capture ratio is undefined (no complete pairs, or a missing up- or down-market regime).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``.

    Note:
        **Correctness**

        The result is checked against an independent reference oracle on every input, and every edge case (missing data
        and boundaries) is given a defined behavior.

        **Edge-case behavior**

        - **Null** — an observation is used only where both legs are present; a ``null`` in either drops that pair.
        - **NaN** — a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``.
        - **Domain** — the compounded (geometric) leg growth is defined only while every selected gross return
          ``1 + r`` stays positive; a selected return at or below ``-1`` wipes that leg out of domain, so the result is
          a loud ``NaN`` — never a plausible wrong number.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`capture_upside_ratio`: The numerator.
        - :func:`capture_downside_ratio`: The denominator.
        - :func:`beta`: The symmetric benchmark sensitivity whose up/down asymmetry this score summarizes.

    References:
        - Morningstar. "Upside/Downside Capture Ratio" (methodology).

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import capture_ratio
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004],
        ...     }
        ... )
        >>> frame.select(capture_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).round(4)).item()
        2.6612

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 6 + ["B"] * 6,
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.01, 0.025, -0.015, 0.008, -0.005, 0.012],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, 0.012, 0.02, -0.01, 0.006, -0.004, 0.01],
        ...     }
        ... )
        >>> reduced = (
        ...     capture_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).over("ticker").round(4)
        ... )
        >>> frame.select(capture_ratio=reduced)["capture_ratio"].unique().sort().to_list()
        [1.4154, 2.6612]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [None, 0.02, 0.03, float("nan"), 0.015, 0.005],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004],
        ...     }
        ... )
        >>> frame.select(capture_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).round(4)).item()
        nan

        **Domain** — a selected gross return at or below ``-1`` on the returns leg is out of the geometric-growth
        domain, so the result is a loud ``NaN``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.02, -1.5, 0.01],
        ...         "benchmark": [0.01, 0.02, -0.03],
        ...     }
        ... )
        >>> frame.select(capture_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252)).item()
        nan
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_periods_per_year(periods_per_year)
    return (
        capture_upside_ratio(returns, benchmark, periods_per_year=periods_per_year)
        / capture_downside_ratio(returns, benchmark, periods_per_year=periods_per_year)
    ).name.keep()


def capture_upside_ratio(
    returns: pl.Expr,
    benchmark: pl.Expr,
    *,
    periods_per_year: int,
) -> pl.Expr:
    r"""
    Upside Capture Ratio, how much of the benchmark's gain a portfolio participated in during up markets.

    The portfolio's annualized return over the benchmark's annualized return, computed over only the periods where the
    benchmark rose (the Morningstar geometric construction):

    .. math::

        \mathrm{UCR} = \frac{\left(\prod_{b_i > 0}(1 + r_i)\right)^{P/n_+} - 1}
        {\left(\prod_{b_i > 0}(1 + b_i)\right)^{P/n_+} - 1},

    where the products run over the :math:`n_+` periods with a positive benchmark return and :math:`P` is
    ``periods_per_year``. A value above one means the portfolio gained more than the benchmark in up markets (good).

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.

    Returns:
        A single ``Float64`` value: the upside capture ratio (one value in ``select``, one per group under ``.over``).
        ``null`` when there are no complete pairs or no up-market periods.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``.

    Note:
        **Correctness**

        The result is checked against an independent reference oracle on every input, and every edge case (missing data
        and boundaries) is given a defined behavior.

        **Edge-case behavior**

        - **Null** — an observation is used only where both legs are present; a ``null`` in either drops that pair.
        - **NaN** — a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``.
        - **Domain** — the compounded (geometric) leg growth is defined only while every selected gross return
          ``1 + r`` stays positive; a selected return at or below ``-1`` wipes that leg out of domain, so the result is
          a loud ``NaN`` — never a plausible wrong number.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`capture_downside_ratio`: The down-market counterpart.
        - :func:`capture_ratio`: Their ratio, an overall asymmetry measure.
        - :func:`beta`: The symmetric benchmark sensitivity this asymmetric up-market measure refines.

    References:
        - Morningstar. "Upside/Downside Capture Ratio" (methodology).

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import capture_upside_ratio
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004],
        ...     }
        ... )
        >>> frame.select(
        ...     capture_upside_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).round(4)
        ... ).item()
        2.7513

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 6 + ["B"] * 6,
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.01, 0.025, -0.015, 0.008, -0.005, 0.012],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, 0.012, 0.02, -0.01, 0.006, -0.004, 0.01],
        ...     }
        ... )
        >>> reduced = (
        ...     capture_upside_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252)
        ...     .over("ticker")
        ...     .round(4)
        ... )
        >>> frame.select(capture_upside_ratio=reduced)["capture_upside_ratio"].unique().sort().to_list()
        [1.5705, 2.7513]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [None, 0.02, 0.03, float("nan"), 0.015, 0.005],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004],
        ...     }
        ... )
        >>> frame.select(
        ...     capture_upside_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).round(4)
        ... ).item()
        nan

        **Domain** — a selected up-market return at or below ``-1`` is outside the geometric-growth domain, so the
        result is a loud ``NaN``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.02, -1.5, 0.01],
        ...         "benchmark": [0.01, 0.02, 0.03],
        ...     }
        ... )
        >>> frame.select(capture_upside_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252)).item()
        nan
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_periods_per_year(periods_per_year)
    returns_paired, benchmark_paired = _paired(returns, benchmark)
    return _capture(returns_paired, benchmark_paired, periods_per_year, upside=True).name.keep()


def information_ratio(
    returns: pl.Expr,
    benchmark: pl.Expr,
    *,
    periods_per_year: int,
) -> pl.Expr:
    r"""
    Information Ratio, the annualized active return per unit of tracking error.

    The mean active return (portfolio minus benchmark) divided by its sample standard deviation (the tracking error),
    annualized by the square-root-of-time rule:

    .. math::

        \mathrm{IR} = \frac{\bar{a}}{\sigma_a}\,\sqrt{P}, \qquad a_i = r_i - b_i,

    where :math:`\sigma_a` is the sample standard deviation (``ddof = 1``) of the active returns :math:`a_i` and
    :math:`P` is ``periods_per_year``. It measures the consistency of out- (or under-) performance versus the benchmark.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.

    Returns:
        A single ``Float64`` value: the annualized information ratio (one value in ``select``, one per group under
        ``.over``). ``null`` when fewer than two complete pairs are present (the tracking error is undefined).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``.

    Note:
        **Correctness**

        The result is checked against an independent reference oracle on every input, and every edge case (missing data
        and boundaries) is given a defined behavior.

        **Edge-case behavior**

        - **Null** — an observation is used only where both legs are present; a ``null`` in either drops that pair.
        - **NaN** — a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``.
        - **Insufficient sample** — fewer than two complete pairs leaves the sample tracking error undefined, so the
          result is ``null``.
        - **Degenerate denominator** — a constant active series has zero tracking error, so the result is ``+/-inf``
          (or ``NaN`` when the mean active is also zero, the ``0 / 0``) — reported, not clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`sharpe_ratio`: The total-risk analog measured against a risk-free rate, not a benchmark.
        - :func:`information_ratio_rolling`: The same measure over a trailing window.
        - :func:`alpha`: The benchmark-active return measured per unit of beta instead of tracking error.

    References:
        - Goodwin, T. H. (1998). "The Information Ratio." *Financial Analysts Journal*, 54(4), 34-43.
        - https://doi.org/10.2469/faj.v54.n4.2196
        - https://en.wikipedia.org/wiki/Information_ratio

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import information_ratio
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004],
        ...     }
        ... )
        >>> frame.select(
        ...     information_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).round(4)
        ... ).item()
        5.5663

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 6 + ["B"] * 6,
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.01, 0.025, -0.015, 0.008, -0.005, 0.012],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, 0.012, 0.02, -0.01, 0.006, -0.004, 0.01],
        ...     }
        ... )
        >>> reduced = (
        ...     information_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).over("ticker").round(4)
        ... )
        >>> frame.select(information_ratio=reduced)["information_ratio"].unique().sort().to_list()
        [0.7463, 5.5663]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [None, 0.02, 0.03, float("nan"), 0.015, 0.005],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004],
        ...     }
        ... )
        >>> frame.select(
        ...     information_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).round(4)
        ... ).item()
        nan

        **Insufficient sample** — a single complete pair yields ``null``, since the tracking error needs two
        observations:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.05],
        ...         "benchmark": [0.04],
        ...     }
        ... )
        >>> expr = information_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252)
        >>> frame.select(information_ratio=expr)["information_ratio"].to_list()
        [None]

        **Degenerate denominator** — a constant active series has zero tracking error with a positive mean, so the ratio
        is ``+inf``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.01, 0.01, 0.01],
        ...         "benchmark": [0.0, 0.0, 0.0],
        ...     }
        ... )
        >>> frame.select(information_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252)).item()
        inf

        **Degenerate denominator** — identical legs give an exactly-zero active series, the zero mean over zero tracking
        error ``0 / 0`` case, so the result is ``NaN``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.01, 0.02, 0.03],
        ...         "benchmark": [0.01, 0.02, 0.03],
        ...     }
        ... )
        >>> frame.select(information_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252)).item()
        nan
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_periods_per_year(periods_per_year)
    returns_paired, benchmark_paired = _paired(returns, benchmark)
    active = returns_paired - benchmark_paired
    # volatility at periods_per_year=1 is the per-period tracking error with the exactly-constant active series
    # pinned to exactly zero, so the documented "zero tracking error -> +/-inf" contract holds bit-exactly.
    annualized = active.mean() / volatility(active, periods_per_year=1) * math.sqrt(periods_per_year)
    return pl.when(active.len() < _MINIMUM_PAIRED_OBSERVATIONS).then(None).otherwise(annualized).name.keep()


def information_ratio_rolling(
    returns: pl.Expr,
    benchmark: pl.Expr,
    window: int,
    *,
    periods_per_year: int,
) -> pl.Expr:
    r"""
    Rolling Information Ratio over a window — the windowed twin of :func:`information_ratio`.

    The mean active return (portfolio minus benchmark) over its sample standard deviation (the tracking error),
    annualized, over each trailing window:

    .. math::

        \mathrm{IR}_t = \frac{\bar{a}_t}{\sigma_{a,t}}\,\sqrt{P}, \qquad a_i = r_i - b_i, \quad n = \text{window},

    where :math:`\sigma_{a,t}` is the sample standard deviation (``ddof = 1``) of the active returns over the window and
    :math:`P` is ``periods_per_year``.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        window: Number of observations in the moving window. Must be ``>= 2``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.

    Returns:
        The rolling information ratio for each row, the same length as the input. The first ``window - 1`` rows are
        ``null`` (warm-up): the window must hold ``window`` complete pairs before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``, or if ``periods_per_year < 1``.

    Note:
        **Correctness**

        Each window matches an independent reference oracle (the reducing :func:`information_ratio` over the window).

        **Edge-case behavior**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **Degenerate denominator** — a constant active window has zero tracking error, so the result is ``+/-inf``
          (or ``NaN`` when the mean active is also zero) — reported, not clipped.
        - **Stability** — a near-flat (non-bit-identical) active-return window sits at the float-conditioning limit the
          documentation's *Correctness* page documents: the one-pass rolling tracking error and an exact two-pass
          recomputation can round a vanishing denominator apart without bound there. The bit-flat window is pinned
          exactly (a zero tracking error, the documented ``+/-inf`` / ``NaN``); real market windows are far from the
          regime.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`information_ratio`: The whole-series reducing form.
        - :func:`sharpe_ratio_rolling`: The rolling total-risk analog measured against a risk-free rate.
        - :func:`alpha_rolling`: The rolling benchmark-active return measured per unit of beta.

    References:
        - Goodwin, T. H. (1998). "The Information Ratio." *Financial Analysts Journal*, 54(4), 34-43.
        - https://doi.org/10.2469/faj.v54.n4.2196
        - https://en.wikipedia.org/wiki/Information_ratio

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import information_ratio_rolling
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.01, 0.02],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018],
        ...     }
        ... )
        >>> frame.select(
        ...     information_ratio_rolling(pl.col("returns"), pl.col("benchmark"), 4, periods_per_year=252).round(4)
        ... ).to_series().to_list()
        [None, None, None, 2.3539, 2.3539, 5.0387, 2.8393, 22.9129]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 6 + ["B"] * 6,
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.01, 0.025, -0.015, 0.008, -0.005, 0.012],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, 0.012, 0.02, -0.01, 0.006, -0.004, 0.01],
        ...     }
        ... )
        >>> expr = (
        ...     information_ratio_rolling(pl.col("returns"), pl.col("benchmark"), 4, periods_per_year=252)
        ...     .over("ticker")
        ...     .round(4)
        ... )
        >>> frame.with_columns(information_ratio_rolling=expr)["information_ratio_rolling"].to_list()
        [None, None, None, 2.3539, 2.3539, 5.0387, None, None, None, 0.0, 0.929, -2.3932]

        A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make the handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [None, float("nan"), 0.03, -0.02, 0.015, 0.005, -0.01, 0.02],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018],
        ...     }
        ... )
        >>> frame.select(
        ...     information_ratio_rolling(pl.col("returns"), pl.col("benchmark"), 4, periods_per_year=252).round(4)
        ... ).to_series().to_list()
        [None, None, None, None, nan, 5.0387, 2.8393, 22.9129]

        **Degenerate denominator** — once the outlier slides out of the window, the window is bit-constant with zero
        tracking error, so the ratio is ``+inf``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [1000000.0, 0.1, 0.1, 0.1, 0.1],
        ...         "benchmark": [0.0, 0.0, 0.0, 0.0, 0.0],
        ...     }
        ... )
        >>> expr = information_ratio_rolling(
        ...     pl.col("returns"), pl.col("benchmark"), window=3, periods_per_year=252
        ... ).round(4)
        >>> frame.select(information_ratio_rolling=expr)["information_ratio_rolling"].to_list()
        [None, None, 9.1652, inf, inf]

        **Degenerate denominator** — a window whose active returns are all exactly zero is the ``0 / 0`` case, so the
        result is ``NaN`` even once larger active values have slid out of the window:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [-0.3233, -0.6457, 0.0, 0.4404, 0.0, 0.0, 0.0, 0.0],
        ...         "benchmark": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ...     }
        ... )
        >>> expr = information_ratio_rolling(
        ...     pl.col("returns"), pl.col("benchmark"), window=4, periods_per_year=252
        ... ).round(4)
        >>> frame.select(information_ratio_rolling=expr)["information_ratio_rolling"].to_list()
        [None, None, None, -4.5223, -1.8213, 7.9373, 7.9373, nan]
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_window(window, minimum=2)
    validate_periods_per_year(periods_per_year)
    active = returns - benchmark
    # rolling_mean_exact, not the native rolling mean: on a bit-constant active window (portfolio tracking the
    # benchmark exactly) the incremental mean's residue would ride above the exactly-zero pinned tracking error as a
    # spurious inf where the documented degeneracy is 0/0 -> NaN.
    mean_active = rolling_mean_exact(active, window)
    # The guarded rolling volatility core (the same one sharpe_ratio_rolling composes) pins a bit-constant window's
    # tracking error to exactly zero, so the documented "zero tracking error -> +/-inf" contract holds even after a
    # much larger active value has slid out of the window (the incremental rolling_std residue regime).
    tracking_error = volatility_rolling(active, window, periods_per_year=periods_per_year)
    return (mean_active * periods_per_year / tracking_error).name.keep()


def modigliani_risk_adjusted_performance(
    returns: pl.Expr,
    benchmark: pl.Expr,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> pl.Expr:
    r"""
    Modigliani Risk-Adjusted Performance (a.k.a. M-squared), the portfolio's return rescaled to the benchmark's risk.

    The return the portfolio would have earned if it had been leveraged or de-leveraged to match the benchmark's
    volatility -- the :func:`sharpe_ratio` ratio scaled back into return units by the benchmark's
    annualized volatility, plus the risk-free rate:

    .. math::

        M^2 = r_f + \mathrm{SR}\,\sigma_b,

    where :math:`\mathrm{SR}` is the annualized portfolio Sharpe ratio, :math:`\sigma_b` the benchmark's annualized
    :func:`volatility`, and :math:`r_f` is ``risk_free_rate``. Unlike a bare Sharpe ratio it is expressed
    as an annualized return, directly comparable to the benchmark's own return.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate, used both to form the Sharpe excess (geometrically per period)
            and as the additive level here (default ``0.0``). Must be finite and ``>= -1``.

    Returns:
        A single ``Float64`` value: the M-squared measure as an annualized return (one value in ``select``, one per
        group under ``.over``). ``null`` when fewer than two complete pairs are present.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite or is ``< -1``.

    Note:
        **Correctness**

        The result is checked against an independent reference oracle on every input, and every edge case (missing data
        and boundaries) is given a defined behavior.

        **Edge-case behavior**

        - **Null** — an observation is used only where both legs are present; a ``null`` in either drops that pair.
        - **NaN** — a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``.
        - **Insufficient sample** — fewer than two complete pairs leaves the embedded Sharpe ratio and benchmark
          volatility undefined, so the result is ``null``.
        - **Degenerate denominator** — a constant portfolio has zero volatility, so its :func:`sharpe_ratio` is
          infinite and the result is ``+/-inf`` — reported, not clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`sharpe_ratio`: The risk-adjusted ratio this expresses in return units.
        - :func:`volatility`: The benchmark dispersion it scales to.
        - :func:`information_ratio`: Another benchmark-relative performance measure, as a ratio.

    References:
        - Modigliani, F. & Modigliani, L. (1997). "Risk-Adjusted Performance." *The Journal of Portfolio Management*,
          23(2), 45-54.
        - https://doi.org/10.3905/jpm.23.2.45
        - https://en.wikipedia.org/wiki/Modigliani_risk-adjusted_performance

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import modigliani_risk_adjusted_performance as m_squared
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004],
        ...     }
        ... )
        >>> frame.select(m_squared(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).round(4)).item()
        1.3163

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 6 + ["B"] * 6,
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.01, 0.025, -0.015, 0.008, -0.005, 0.012],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, 0.012, 0.02, -0.01, 0.006, -0.004, 0.01],
        ...     }
        ... )
        >>> reduced = m_squared(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).over("ticker").round(4)
        >>> frame.select(modigliani_risk_adjusted_performance=reduced)[
        ...     "modigliani_risk_adjusted_performance"
        ... ].unique().sort().to_list()
        [1.1541, 1.3163]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [None, 0.02, 0.03, float("nan"), 0.015, 0.005],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004],
        ...     }
        ... )
        >>> frame.select(m_squared(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).round(4)).item()
        nan

        **Insufficient sample** — a single complete pair leaves the embedded Sharpe ratio and benchmark volatility
        undefined, so the result is ``null``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.05],
        ...         "benchmark": [0.04],
        ...     }
        ... )
        >>> expr = m_squared(pl.col("returns"), pl.col("benchmark"), periods_per_year=252)
        >>> frame.select(m_squared=expr)["m_squared"].to_list()
        [None]

        **Degenerate denominator** — a constant portfolio has zero dispersion, so the embedded Sharpe ratio is ``+inf``,
        which propagates to ``+inf``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.01, 0.01, 0.01],
        ...         "benchmark": [0.02, -0.01, 0.03],
        ...     }
        ... )
        >>> expr = m_squared(pl.col("returns"), pl.col("benchmark"), periods_per_year=252)
        >>> frame.select(expr).item()
        inf
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    returns_paired, benchmark_paired = _paired(returns, benchmark)
    base_ratio = sharpe_ratio(returns_paired, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate)
    benchmark_volatility = volatility(benchmark_paired, periods_per_year=periods_per_year)
    return (risk_free_rate + base_ratio * benchmark_volatility).name.keep()


def treynor_ratio(
    returns: pl.Expr,
    benchmark: pl.Expr,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> pl.Expr:
    r"""
    Treynor Ratio, the annualized excess return per unit of systematic (benchmark) risk.

    The portfolio's annualized arithmetic excess return divided by its :func:`beta` -- the reward-to-systematic-risk
    counterpart of the :func:`sharpe_ratio` ratio, which instead divides by total risk:

    .. math::

        \mathrm{Treynor} = \frac{\overline{(r_i - r_f)}\,P}{\beta},

    where :math:`\beta` is the raw regression slope (:func:`beta`), :math:`P` is ``periods_per_year``, and the
    per-period risk-free rate is the geometric conversion :math:`r_f = (1 + \texttt{risk\_free\_rate})^{1/P} - 1`. The
    excess is annualized arithmetically (it is a ratio numerator), where :func:`alpha` compounds geometrically.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate, converted to a per-period rate geometrically (default ``0.0``).
            Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + risk_free_rate >= 0``).

    Returns:
        A single ``Float64`` value: the annualized Treynor ratio (one value in ``select``, one per group under
        ``.over``). ``null`` when fewer than two complete pairs are present.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite or is ``< -1``.

    Note:
        **Correctness**

        The result is checked against an independent reference oracle on every input, and every edge case (missing data
        and boundaries) is given a defined behavior.

        **Edge-case behavior**

        - **Null** — an observation is used only where both legs are present; a ``null`` in either drops that pair.
        - **NaN** — a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``.
        - **Insufficient sample** — fewer than two complete pairs leaves the regression slope undefined, so the result
          is ``null``.
        - **Degenerate denominator** — a zero beta gives ``+/-inf`` (or ``NaN`` when the excess return is also zero) —
          reported, not clipped; a zero-variance benchmark instead makes :func:`beta` ``NaN``, which propagates here.
        - **Stability** — a beta bounded away from zero is the one regime the excess-over-beta quotient genuinely
          needs: as the slope vanishes the division amplifies rounding without bound, so a near-zero beta sits at the
          float-conditioning limit the documentation's *Correctness* page documents. The exact zero-beta case is guarded
          (``+/-inf``); real market betas are far from the regime.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`beta`: The denominator (systematic risk).
        - :func:`sharpe_ratio`: The total-risk analog.
        - :func:`alpha`: The benchmark-relative excess built on the same beta.
        - :func:`treynor_ratio_rolling`: The rolling (windowed) form.

    References:
        - Treynor, J. L. (1965). "How to Rate Management of Investment Funds." *Harvard Business Review*, 43(1), 63-75.
        - https://en.wikipedia.org/wiki/Treynor_ratio

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import treynor_ratio
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004],
        ...     }
        ... )
        >>> frame.select(treynor_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).round(4)).item()
        1.3201

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 6 + ["B"] * 6,
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.01, 0.025, -0.015, 0.008, -0.005, 0.012],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, 0.012, 0.02, -0.01, 0.006, -0.004, 0.01],
        ...     }
        ... )
        >>> reduced = (
        ...     treynor_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).over("ticker").round(4)
        ... )
        >>> frame.select(treynor_ratio=reduced)["treynor_ratio"].unique().sort().to_list()
        [1.1675, 1.3201]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [None, 0.02, 0.03, float("nan"), 0.015, 0.005],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004],
        ...     }
        ... )
        >>> frame.select(treynor_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).round(4)).item()
        nan

        **Insufficient sample** — a single complete pair yields ``null``, since the regression slope needs two
        observations:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.05],
        ...         "benchmark": [0.04],
        ...     }
        ... )
        >>> expr = treynor_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252)
        >>> frame.select(treynor_ratio=expr)["treynor_ratio"].to_list()
        [None]

        **Degenerate denominator** — a zero beta with a positive excess return gives ``+inf``, reported not clipped:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [3.0, 3.0, 1.0, 1.0],
        ...         "benchmark": [1.0, -1.0, 1.0, -1.0],
        ...     }
        ... )
        >>> frame.select(treynor_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252)).item()
        inf

        **Degenerate denominator** — a constant benchmark makes the embedded beta ``NaN``, so the excess-over-beta ratio
        is ``NaN``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.01, -0.02, 0.03],
        ...         "benchmark": [0.1, 0.1, 0.1],
        ...     }
        ... )
        >>> frame.select(treynor_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252)).item()
        nan
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    returns_paired, benchmark_paired = _paired(returns, benchmark)
    rf_period = per_period_rate(risk_free_rate, periods_per_year, name="risk_free_rate")
    annualized_excess = (returns_paired - rf_period).mean() * periods_per_year
    slope = _raw_beta(returns_paired, benchmark_paired)
    return (
        pl.when(returns_paired.len() < _MINIMUM_PAIRED_OBSERVATIONS)
        .then(None)
        .otherwise(annualized_excess / slope)
        .name.keep()
    )


def treynor_ratio_rolling(
    returns: pl.Expr,
    benchmark: pl.Expr,
    window: int,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> pl.Expr:
    r"""
    Rolling Treynor Ratio over a window — the windowed twin of :func:`treynor_ratio`.

    The annualized arithmetic excess return over the rolling :func:`beta_rolling`, computed over each trailing window:

    .. math::

        \mathrm{Treynor}_t = \frac{\overline{(r_i - r_f)}_t\,P}{\beta_t}, \qquad n = \text{window},

    where :math:`\beta_t` is the rolling slope, :math:`P` is ``periods_per_year``, and
    :math:`r_f = (1 + \texttt{risk\_free\_rate})^{1/P} - 1`.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        window: Number of observations in the moving window. Must be ``>= 2``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate, converted to a per-period rate geometrically (default ``0.0``).
            Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + risk_free_rate >= 0``).

    Returns:
        The rolling Treynor ratio for each row, the same length as the input. The first ``window - 1`` rows are
        ``null`` (warm-up): the window must hold ``window`` complete pairs before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``, ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite or is ``< -1``.

    Note:
        **Correctness**

        Each window matches an independent reference oracle (the reducing :func:`treynor_ratio` over the window).

        **Edge-case behavior**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **Degenerate denominator** — a window whose slope is zero gives ``+/-inf`` (or ``NaN``) — reported, not
          clipped; a zero-variance benchmark window instead makes the slope ``NaN``, which propagates here.
        - **Stability** — a near-flat (non-bit-identical) benchmark window sits at the float-conditioning limit the
          documentation's *Correctness* page documents: the one-pass rolling slope and an exact two-pass recomputation
          can round a vanishing benchmark variance — and with it the ``beta`` divisor — apart without bound there. The
          bit-flat window is guarded exactly (``NaN``); real market windows are far from the regime.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`treynor_ratio`: The whole-series reducing form.
        - :func:`beta_rolling`: The denominator (systematic risk).
        - :func:`alpha_rolling`: The rolling benchmark-relative excess built on the same slope.

    References:
        - Treynor, J. L. (1965). "How to Rate Management of Investment Funds." *Harvard Business Review*, 43(1), 63-75.
        - https://en.wikipedia.org/wiki/Treynor_ratio

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import treynor_ratio_rolling
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.01, 0.02],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018],
        ...     }
        ... )
        >>> frame.select(
        ...     treynor_ratio_rolling(pl.col("returns"), pl.col("benchmark"), 4, periods_per_year=252).round(4)
        ... ).to_series().to_list()
        [None, None, None, 0.9993, 0.7483, 1.4938, -0.5003, 1.8295]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 6 + ["B"] * 6,
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.01, 0.025, -0.015, 0.008, -0.005, 0.012],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, 0.012, 0.02, -0.01, 0.006, -0.004, 0.01],
        ...     }
        ... )
        >>> expr = (
        ...     treynor_ratio_rolling(pl.col("returns"), pl.col("benchmark"), 4, periods_per_year=252)
        ...     .over("ticker")
        ...     .round(4)
        ... )
        >>> frame.with_columns(treynor_ratio_rolling=expr)["treynor_ratio_rolling"].to_list()
        [None, None, None, 0.9993, 0.7483, 1.4938, None, None, None, 1.3726, 0.6224, 0.0]

        A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make the handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [None, float("nan"), 0.03, -0.02, 0.015, 0.005, -0.01, 0.02],
        ...         "benchmark": [0.015, -0.008, 0.025, -0.015, 0.01, 0.004, -0.012, 0.018],
        ...     }
        ... )
        >>> frame.select(
        ...     treynor_ratio_rolling(pl.col("returns"), pl.col("benchmark"), 4, periods_per_year=252).round(4)
        ... ).to_series().to_list()
        [None, None, None, None, nan, 1.4938, -0.5003, 1.8295]

        **Degenerate denominator** — a ``null`` in a window yields ``null`` under the pairwise-complete gate before the
        constant-benchmark ``NaN`` branch, so the result stays ``null`` until the window clears and then reports
        ``NaN``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [0.02, None, 0.03, 0.01, 0.02],
        ...         "benchmark": [0.1, 0.1, 0.1, 0.1, 0.1],
        ...     }
        ... )
        >>> expr = treynor_ratio_rolling(pl.col("returns"), pl.col("benchmark"), window=3, periods_per_year=252)
        >>> frame.select(treynor_ratio_rolling=expr)["treynor_ratio_rolling"].to_list()
        [None, None, None, None, nan]

        **Degenerate denominator** — a zero-beta window with a positive excess return gives ``+inf``, reported not
        clipped:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns": [3.0, 3.0, 1.0, 1.0],
        ...         "benchmark": [1.0, -1.0, 1.0, -1.0],
        ...     }
        ... )
        >>> expr = treynor_ratio_rolling(pl.col("returns"), pl.col("benchmark"), window=4, periods_per_year=252)
        >>> frame.select(treynor_ratio_rolling=expr)["treynor_ratio_rolling"].to_list()
        [None, None, None, inf]
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_window(window, minimum=2)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    rf_period = per_period_rate(risk_free_rate, periods_per_year, name="risk_free_rate")
    annualized_excess = (returns - rf_period).rolling_mean(window, min_samples=window) * periods_per_year
    return (annualized_excess / _rolling_raw_beta(returns, benchmark, window)).name.keep()
