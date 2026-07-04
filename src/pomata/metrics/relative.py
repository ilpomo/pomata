r"""
Benchmark-relative metrics — performance and risk measured against a second, benchmark return series.

Every metric here is two-input: it reads a portfolio (or strategy) return series and an aligned benchmark return series,
both as fractions, and reduces the pair to a single value. The two series are treated as **pairwise-complete**: an
observation contributes only where BOTH legs are present, so a ``null`` in either leg drops that pair; among the
retained pairs a ``NaN`` in either leg poisons the result to ``NaN``. This is the composing layer for cross-sectional
analytics: ``beta`` is the shared regression slope reused by ``alpha`` and ``treynor_ratio``, and
``modigliani_risk_adjusted_performance`` is built on the single-input :func:`sharpe_ratio` and
:func:`volatility`, imported from their specific theme modules so the theme dependency graph stays
acyclic.
"""

import math

import polars as pl

from pomata._expr import float64_expr, per_period_rate, validate_finite, validate_periods_per_year, validate_window
from pomata.metrics.ratio import sharpe_ratio
from pomata.metrics.risk import volatility

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
_MINIMUM_PAIRED_OBSERVATIONS = 2


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
    return (
        pl.when(returns.len() < 1)
        .then(None)
        .when(poisoned)
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
        returns: Per-bar net return series, as fractions (e.g. from :func:`returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate, converted to a per-period rate geometrically (default ``0.0``).
            Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + risk_free_rate > 0``).

    Returns:
        A single ``Float64`` value: the annualized Jensen's alpha (one value in ``select``, one per group under
        ``.over``). ``null`` when fewer than two complete pairs are present.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — an observation is used only where both legs are present; a ``null`` in either drops that pair.
        - **NaN** — a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``.
        - **Fewer than two pairs** — the regression slope is undefined, so the result is ``null``.
        - **Constant benchmark** — a zero-variance benchmark makes :func:`beta` ``NaN``, which propagates here.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``alpha(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).over("ticker")``.

    See Also:
        - :func:`beta`: The regression slope this corrects the return for.
        - :func:`treynor_ratio`: The excess return per unit of the same systematic risk.
        - :func:`alpha_rolling`: The same measure over a trailing window.

    References:
        - Jensen, M. C. (1968). "The Performance of Mutual Funds in the Period 1945-1964." *The Journal of Finance*,
          23(2), 389-416.
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
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
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
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    returns_paired, benchmark_paired = _paired(returns, benchmark)
    rf_period = per_period_rate(risk_free_rate, periods_per_year)
    slope = _raw_beta(returns_paired, benchmark_paired)
    excess_leg = (returns_paired - rf_period) - slope * (benchmark_paired - rf_period)
    annualized = (1.0 + excess_leg.mean()) ** periods_per_year - 1.0
    return pl.when(returns_paired.len() < _MINIMUM_PAIRED_OBSERVATIONS).then(None).otherwise(annualized)


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
        returns: Per-bar net return series, as fractions (e.g. from :func:`returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        window: Number of observations in the moving window. Must be ``>= 2``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate, converted to a per-period rate geometrically (default ``0.0``).
            Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + risk_free_rate > 0``).

    Returns:
        The rolling Jensen's alpha for each row, the same length as the input. The first ``window - 1`` rows are
        ``null`` (warm-up): the window must hold ``window`` complete pairs before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``, ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite.

    Note:
        **Correctness** -- each window matches an independent reference oracle (the reducing :func:`alpha` over the
        window).

        **Edge-case behavior:**

        - **Null** — a window with a ``null`` in either leg yields ``null`` (it must hold ``window`` complete pairs).
        - **NaN** — a ``NaN`` in either leg of the window propagates, yielding ``NaN``.
        - **Constant benchmark** — a zero-variance window benchmark makes the slope ``NaN``, which propagates here.
        - **Partitioning** — wrap the call in ``.over(...)`` so the window never spans series boundaries.

    See Also:
        - :func:`alpha`: The whole-series reducing form.
        - :func:`beta_rolling`: The rolling slope this corrects the return for.
        - :func:`treynor_ratio_rolling`: The rolling excess per unit of the same systematic risk.

    References:
        - Jensen, M. C. (1968). "The Performance of Mutual Funds in the Period 1945-1964." *The Journal of Finance*,
          23(2), 389-416.
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
        >>> frame.with_columns(expr.alias("m"))["m"].to_list()
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
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_window(window, minimum=2)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    rf_period = per_period_rate(risk_free_rate, periods_per_year)
    slope = _rolling_raw_beta(returns, benchmark, window)
    mean_returns = returns.rolling_mean(window, min_samples=window)
    mean_benchmark = benchmark.rolling_mean(window, min_samples=window)
    alpha_period = (mean_returns - rf_period) - slope * (mean_benchmark - rf_period)
    return (1.0 + alpha_period) ** periods_per_year - 1.0


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
        returns: Per-bar net return series, as fractions (e.g. from :func:`returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.

    Returns:
        A single ``Float64`` value: the regression slope (one value in ``select``, one per group under ``.over``).
        ``null`` when fewer than two complete pairs are present.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — an observation is used only where both legs are present; a ``null`` in either drops that pair.
        - **NaN** — a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``.
        - **Fewer than two pairs** — the regression slope is undefined, so the result is ``null``.
        - **Constant benchmark** — a zero-variance benchmark gives ``0 / 0``, reported as ``NaN`` rather than clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``beta(pl.col("returns"), pl.col("benchmark")).over("ticker")``.

    See Also:
        - :func:`alpha`: The benchmark-relative return that nets out beta-explained performance.
        - :func:`treynor_ratio`: The excess return per unit of this systematic risk.
        - :func:`beta_rolling`: The same slope over a trailing window.

    References:
        - Sharpe, W. F. (1964). "Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk."
          *The Journal of Finance*, 19(3), 425-442.
        - https://en.wikipedia.org/wiki/Beta_(finance)

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
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
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
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    returns_paired, benchmark_paired = _paired(returns, benchmark)
    return (
        pl.when(returns_paired.len() < _MINIMUM_PAIRED_OBSERVATIONS)
        .then(None)
        .otherwise(_raw_beta(returns_paired, benchmark_paired))
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
        returns: Per-bar net return series, as fractions (e.g. from :func:`returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        window: Number of observations in the moving window. Must be ``>= 2``.

    Returns:
        The rolling regression slope for each row, the same length as the input. The first ``window - 1`` rows are
        ``null`` (warm-up): the window must hold ``window`` complete pairs before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``.

    Note:
        **Correctness** -- each window matches an independent reference oracle (the reducing :func:`beta` over the
        window).

        **Edge-case behavior:**

        - **Null** — a window with a ``null`` in either leg yields ``null`` (it must hold ``window`` complete pairs).
        - **NaN** — a ``NaN`` in either leg of the window propagates, yielding ``NaN``.
        - **Constant benchmark** — a zero-variance window benchmark gives ``0 / 0``, reported as ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` so the window never spans series boundaries.

    See Also:
        - :func:`beta`: The whole-series reducing form.
        - :func:`alpha_rolling`: The benchmark-relative return built on this slope.
        - :func:`treynor_ratio_rolling`: The excess return per unit of this systematic risk.

    References:
        - Sharpe, W. F. (1964). "Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk."
          *The Journal of Finance*, 19(3), 425-442.
        - https://en.wikipedia.org/wiki/Beta_(finance)

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
        >>> frame.with_columns(expr.alias("m"))["m"].to_list()
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
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_window(window, minimum=2)
    return _rolling_raw_beta(returns, benchmark, window)


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
        returns: Per-bar net return series, as fractions (e.g. from :func:`returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.

    Returns:
        A single ``Float64`` value: the downside capture ratio (one value in ``select``, one per group under ``.over``).
        ``null`` when there are no complete pairs or no down-market periods.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — an observation is used only where both legs are present; a ``null`` in either drops that pair.
        - **NaN** — a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``.
        - **No down-market periods** — with no negative-benchmark period the ratio is undefined, so the result is
          ``null``.
        - **Zero benchmark loss** — a zero annualized benchmark loss gives ``+/-inf`` (or ``NaN``), reported rather than
          clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``capture_downside_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).over("ticker")``.

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
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
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
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_periods_per_year(periods_per_year)
    returns_paired, benchmark_paired = _paired(returns, benchmark)
    return _capture(returns_paired, benchmark_paired, periods_per_year, upside=False)


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
        returns: Per-bar net return series, as fractions (e.g. from :func:`returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.

    Returns:
        A single ``Float64`` value: the capture ratio (one value in ``select``, one per group under ``.over``).
        ``null`` when either capture ratio is undefined (no complete pairs, or a missing up- or down-market regime).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — an observation is used only where both legs are present; a ``null`` in either drops that pair.
        - **NaN** — a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``.
        - **Missing a market regime** — with no up-market or no down-market period a capture ratio is undefined, so the
          result is ``null``.
        - **Zero downside capture** — a zero downside capture gives ``+/-inf`` (or ``NaN``), reported rather than
          clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``capture_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).over("ticker")``.

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
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
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
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_periods_per_year(periods_per_year)
    return capture_upside_ratio(returns, benchmark, periods_per_year=periods_per_year) / capture_downside_ratio(
        returns, benchmark, periods_per_year=periods_per_year
    )


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
        returns: Per-bar net return series, as fractions (e.g. from :func:`returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.

    Returns:
        A single ``Float64`` value: the upside capture ratio (one value in ``select``, one per group under ``.over``).
        ``null`` when there are no complete pairs or no up-market periods.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — an observation is used only where both legs are present; a ``null`` in either drops that pair.
        - **NaN** — a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``.
        - **No up-market periods** — with no positive-benchmark period the ratio is undefined, so the result is
          ``null``.
        - **Zero benchmark gain** — a zero annualized benchmark gain gives ``+/-inf`` (or ``NaN``), reported rather than
          clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``capture_upside_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).over("ticker")``.

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
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
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
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_periods_per_year(periods_per_year)
    returns_paired, benchmark_paired = _paired(returns, benchmark)
    return _capture(returns_paired, benchmark_paired, periods_per_year, upside=True)


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
        returns: Per-bar net return series, as fractions (e.g. from :func:`returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.

    Returns:
        A single ``Float64`` value: the annualized information ratio (one value in ``select``, one per group under
        ``.over``). ``null`` when fewer than two complete pairs are present (the tracking error is undefined).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — an observation is used only where both legs are present; a ``null`` in either drops that pair.
        - **NaN** — a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``.
        - **Fewer than two pairs** — the sample tracking error is undefined, so the result is ``null``.
        - **Zero tracking error** — a constant active series gives ``+/-inf`` (or ``NaN`` when the mean active is also
          zero), reported rather than clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``information_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).over("ticker")``.

    See Also:
        - :func:`sharpe_ratio`: The total-risk analog measured against a risk-free rate, not a benchmark.
        - :func:`information_ratio_rolling`: The same measure over a trailing window.
        - :func:`alpha`: The benchmark-active return measured per unit of beta instead of tracking error.

    References:
        - Goodwin, T. H. (1998). "The Information Ratio." *Financial Analysts Journal*, 54(4), 34-43.
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
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
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
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_periods_per_year(periods_per_year)
    returns_paired, benchmark_paired = _paired(returns, benchmark)
    active = returns_paired - benchmark_paired
    annualized = active.mean() / active.std(ddof=1) * math.sqrt(periods_per_year)
    return pl.when(active.len() < _MINIMUM_PAIRED_OBSERVATIONS).then(None).otherwise(annualized)


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
        returns: Per-bar net return series, as fractions (e.g. from :func:`returns_net`).
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
        **Correctness** -- each window matches an independent reference oracle (the reducing :func:`information_ratio`
        over the window).

        **Edge-case behavior:**

        - **Null** — a window with a ``null`` in either leg yields ``null`` (it must hold ``window`` complete pairs).
        - **NaN** — a ``NaN`` in either leg of the window propagates, yielding ``NaN``.
        - **Zero tracking error** — a constant active window gives ``+/-inf`` (or ``NaN``), reported not clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` so the window never spans series boundaries.

    See Also:
        - :func:`information_ratio`: The whole-series reducing form.
        - :func:`sharpe_ratio_rolling`: The rolling total-risk analog measured against a risk-free rate.
        - :func:`alpha_rolling`: The rolling benchmark-active return measured per unit of beta.

    References:
        - Goodwin, T. H. (1998). "The Information Ratio." *Financial Analysts Journal*, 54(4), 34-43.
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
        >>> frame.with_columns(expr.alias("m"))["m"].to_list()
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
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_window(window, minimum=2)
    validate_periods_per_year(periods_per_year)
    active = returns - benchmark
    mean_active = active.rolling_mean(window, min_samples=window)
    tracking_error = active.rolling_std(window, ddof=1, min_samples=window)
    return mean_active / tracking_error * math.sqrt(periods_per_year)


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
        returns: Per-bar net return series, as fractions (e.g. from :func:`returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate, used both to form the Sharpe excess (geometrically per period)
            and as the additive level here (default ``0.0``). Must be finite and ``>= -1``.

    Returns:
        A single ``Float64`` value: the M-squared measure as an annualized return (one value in ``select``, one per
        group under ``.over``). ``null`` when fewer than two complete pairs are present.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — an observation is used only where both legs are present; a ``null`` in either drops that pair.
        - **NaN** — a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``.
        - **Fewer than two pairs** — the Sharpe ratio and benchmark volatility are undefined, so the result is ``null``.
        - **Zero portfolio volatility** — a constant portfolio gives an infinite Sharpe ratio, which propagates here.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``modigliani_risk_adjusted_performance(pl.col("r"), pl.col("b"), periods_per_year=252).over("ticker")``.

    See Also:
        - :func:`sharpe_ratio`: The risk-adjusted ratio this expresses in return units.
        - :func:`volatility`: The benchmark dispersion it scales to.
        - :func:`information_ratio`: Another benchmark-relative performance measure, as a ratio.

    References:
        - Modigliani, F. & Modigliani, L. (1997). "Risk-Adjusted Performance." *The Journal of Portfolio Management*,
          23(2), 45-54.
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
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
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
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    returns_paired, benchmark_paired = _paired(returns, benchmark)
    base_ratio = sharpe_ratio(returns_paired, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate)
    benchmark_volatility = volatility(benchmark_paired, periods_per_year=periods_per_year)
    return risk_free_rate + base_ratio * benchmark_volatility


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
        returns: Per-bar net return series, as fractions (e.g. from :func:`returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate, converted to a per-period rate geometrically (default ``0.0``).
            Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + risk_free_rate > 0``).

    Returns:
        A single ``Float64`` value: the annualized Treynor ratio (one value in ``select``, one per group under
        ``.over``). ``null`` when fewer than two complete pairs are present.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — an observation is used only where both legs are present; a ``null`` in either drops that pair.
        - **NaN** — a ``NaN`` in either leg of a retained pair propagates, yielding ``NaN``.
        - **Fewer than two pairs** — the regression slope is undefined, so the result is ``null``.
        - **Zero beta** — a zero systematic risk gives ``+/-inf`` (or ``NaN`` when the excess return is also zero),
          reported rather than clipped.
        - **Constant benchmark** — a zero-variance benchmark makes :func:`beta` ``NaN``, which propagates here.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``treynor_ratio(pl.col("returns"), pl.col("benchmark"), periods_per_year=252).over("ticker")``.

    See Also:
        - :func:`beta`: The denominator (systematic risk).
        - :func:`sharpe_ratio`: The total-risk analog.
        - :func:`alpha`: The benchmark-relative excess built on the same beta.

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
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
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
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    returns_paired, benchmark_paired = _paired(returns, benchmark)
    rf_period = per_period_rate(risk_free_rate, periods_per_year)
    annualized_excess = (returns_paired - rf_period).mean() * periods_per_year
    slope = _raw_beta(returns_paired, benchmark_paired)
    return pl.when(returns_paired.len() < _MINIMUM_PAIRED_OBSERVATIONS).then(None).otherwise(annualized_excess / slope)


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
        returns: Per-bar net return series, as fractions (e.g. from :func:`returns_net`).
        benchmark: Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.
        window: Number of observations in the moving window. Must be ``>= 2``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate, converted to a per-period rate geometrically (default ``0.0``).
            Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + risk_free_rate > 0``).

    Returns:
        The rolling Treynor ratio for each row, the same length as the input. The first ``window - 1`` rows are
        ``null`` (warm-up): the window must hold ``window`` complete pairs before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``, ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite.

    Note:
        **Correctness** -- each window matches an independent reference oracle (the reducing :func:`treynor_ratio` over
        the window).

        **Edge-case behavior:**

        - **Null** — a window with a ``null`` in either leg yields ``null`` (it must hold ``window`` complete pairs).
        - **NaN** — a ``NaN`` in either leg of the window propagates, yielding ``NaN``.
        - **Zero beta** — a window whose slope is zero gives ``+/-inf`` (or ``NaN``), reported rather than clipped.
        - **Constant benchmark** — a zero-variance window benchmark makes the slope ``NaN``, which propagates here.
        - **Partitioning** — wrap the call in ``.over(...)`` so the window never spans series boundaries.

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
        >>> frame.with_columns(expr.alias("m"))["m"].to_list()
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
    """
    returns = float64_expr(returns)
    benchmark = float64_expr(benchmark)
    validate_window(window, minimum=2)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    rf_period = per_period_rate(risk_free_rate, periods_per_year)
    annualized_excess = (returns - rf_period).rolling_mean(window, min_samples=window) * periods_per_year
    return annualized_excess / _rolling_raw_beta(returns, benchmark, window)
