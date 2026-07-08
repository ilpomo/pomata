"""
Return metrics — the total and annualized (CAGR) compounded return of an equity curve, and the trend stability of a
return series.
"""

import polars as pl

from pomata._expr import float64_expr, validate_periods_per_year, validate_window

__all__ = ("cagr", "cagr_rolling", "stability", "total_return", "total_return_rolling")

# The smallest number of observations a two-point regression (the stability fit) can be computed over.
_MINIMUM_REGRESSION_POINTS = 2


def cagr(
    equity_curve: pl.Expr,
    *,
    periods_per_year: int,
) -> pl.Expr:
    r"""
    Compound Annual Growth Rate (CAGR), the constant per-year rate that reproduces the curve's total growth.

    The geometric annualized return: the per-year rate which, compounded over the series, takes a unit of capital to its
    final value. With :math:`E` the equity curve (a growth factor from a unit start), :math:`N` its number of
    observations, and :math:`P` the periods per year:

    .. math::

        \mathrm{CAGR} = E_N^{\,P / N} - 1,

    since :math:`E_N` is the total growth multiple over :math:`N` periods, i.e. :math:`N / P` years. It is the geometric
    counterpart of an arithmetic average return and the numerator of :func:`calmar_ratio`.

    Args:
        equity_curve: Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive; its ``N``
            values are ``N`` period growth factors, and its final value is the total growth multiple.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.

    Returns:
        A single ``Float64`` value: the compound annual growth rate (one value in ``select``, one per group under
        ``.over``). ``null`` when there are no observations.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Domain** — the geometric growth rate is defined only on a positive terminal equity: a curve whose last
        defined value is ``<= 0`` has no fractional-power growth (the raw power would be parity-dependent garbage),
        so the result is a loud ``NaN``.

        **Edge-case behavior:**

        - **Null** — ``null`` equities are skipped; the rate uses the last defined equity and the count of defined
          observations. An all-null series yields ``null``.
        - **NaN** — a ``NaN`` anywhere yields ``NaN``.
        - **Few observations** — annualizing a handful of periods extrapolates aggressively (e.g. one period at
          ``periods_per_year = 252`` raises the growth to the 252nd power); this is the defined geometric behavior, not
          an error.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the rate is computed per
          series, e.g. ``cagr(pl.col("equity"), periods_per_year=252).over("ticker")``.

    See Also:
        - :func:`total_return`: The un-annualized total growth this is the per-year rate of.
        - :func:`cagr_rolling`: The windowed twin, computed over each trailing window.
        - :func:`calmar_ratio`: The CAGR-over-drawdown ratio built on this.

    References:
        - https://en.wikipedia.org/wiki/Compound_annual_growth_rate

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import cagr
        >>>
        >>> frame = pl.DataFrame({"equity": [1.1, 1.21]})
        >>> frame.select(cagr(pl.col("equity"), periods_per_year=1).round(4)).item()
        0.1

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 2 + ["B"] * 2,
        ...         "equity_curve": [1.1, 1.21, 1.05, 1.1025],
        ...     }
        ... )
        >>> reduced = cagr(pl.col("equity_curve"), periods_per_year=1).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [0.05, 0.1]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"equity_curve": [1.0, 1.1, None, 1.21, float("nan"), 1.3]})
        >>> frame.select(cagr(pl.col("equity_curve"), periods_per_year=1).round(4)).item()
        nan
    """
    equity_curve = float64_expr(equity_curve)
    validate_periods_per_year(periods_per_year)
    defined = equity_curve.drop_nulls()
    growth = defined.last() ** (periods_per_year / defined.count()) - 1
    # Domain: the fractional power is defined only on a positive terminal equity. A wiped-out (or sign-flipped)
    # curve has no geometric growth rate — the raw power would be parity-dependent garbage (a plausible finite for
    # integer-valued exponents, NaN otherwise) — so out of domain is a loud NaN, never a plausible wrong number.
    return (
        pl.when(equity_curve.is_nan().any())
        .then(pl.lit(float("nan")))
        .when(defined.last() <= 0.0)
        .then(pl.lit(float("nan")))
        .otherwise(growth)
        .name.keep()
    )


def cagr_rolling(
    equity_curve: pl.Expr,
    window: int,
    *,
    periods_per_year: int,
) -> pl.Expr:
    r"""
    Rolling Compound Annual Growth Rate over a window — the windowed twin of :func:`cagr`.

    The geometric annualized return over each trailing window, from the window's two endpoints:

    .. math::

        \mathrm{CAGR}_t = \left(\frac{E_t}{E_{t-n+1}}\right)^{P/(n-1)} - 1, \qquad n = \text{window},

    where :math:`E` is the equity curve and :math:`P` is ``periods_per_year``. The window's two endpoints span ``n - 1``
    periods, so the growth ratio is annualized over ``n - 1``; as an endpoint quantity it depends only on the first and
    last equity of the window, not the path between them.

    Args:
        equity_curve: Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.
        window: Number of observations in the moving window. Must be ``>= 2``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.

    Returns:
        The rolling compound annual growth rate for each row, the same length as the input. The first ``window - 1``
        rows are ``null`` (warm-up): the window must reach back ``window`` rows before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``, or if ``periods_per_year < 1``.

    Note:
        **Correctness** -- each window matches an independent reference oracle (the endpoint ratio annualized).

        **Domain** — the geometric growth rate is defined only on a positive endpoint ratio: a window whose equity
        crossed or touched zero has no fractional-power growth, so that window is a loud ``NaN``.

        **Edge-case behavior:**

        - **Null** — a ``null`` at either window endpoint yields ``null``; being an endpoint quantity, an interior
          ``null`` does not affect the result.
        - **NaN** — a ``NaN`` at either endpoint propagates, yielding ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` so the window never spans series boundaries.

    See Also:
        - :func:`cagr`: The whole-series reducing form.
        - :func:`total_return_rolling`: The non-annualized windowed return.
        - :func:`total_return`: The whole-series, non-annualized total growth.

    References:
        - https://en.wikipedia.org/wiki/Compound_annual_growth_rate

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import cagr_rolling
        >>>
        >>> frame = pl.DataFrame({"equity": [1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25]})
        >>> frame.select(cagr_rolling(pl.col("equity"), 3, periods_per_year=4).round(4))["equity"].to_list()
        [None, None, 0.1025, 0.1901, 0.1995, 0.1736, 0.1815]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "equity": [1.0, 1.1, 1.05, 1.2, 1.15, 1.0, 1.02, 1.08, 1.05, 1.12],
        ...     }
        ... )
        >>> rolled = cagr_rolling(pl.col("equity"), 3, periods_per_year=4).over("ticker").round(4)
        >>> frame.select(rolled.alias("m"))["m"].to_list()
        [None, None, 0.1025, 0.1901, 0.1995, None, None, 0.1664, 0.0597, 0.0754]

        A ``null`` or ``NaN`` at a window endpoint propagates, while a ``NaN`` interior to a window is ignored:

        >>> frame = pl.DataFrame({"equity": [None, 1.1, 1.05, 1.2, float("nan"), 1.3, 1.25]})
        >>> frame.select(cagr_rolling(pl.col("equity"), 3, periods_per_year=4).round(4))["equity"].to_list()
        [None, None, None, 0.1901, nan, 0.1736, nan]
    """
    equity_curve = float64_expr(equity_curve)
    validate_window(window, minimum=2)
    validate_periods_per_year(periods_per_year)
    ratio = equity_curve / equity_curve.shift(window - 1)
    # Domain: the fractional power is defined only on a positive endpoint ratio (see cagr) — a window whose equity
    # crossed or touched zero has no geometric growth rate, so that window is a loud NaN.
    return (
        pl.when(ratio <= 0.0)
        .then(pl.lit(float("nan")))
        .otherwise(ratio ** (periods_per_year / (window - 1)) - 1.0)
        .name.keep()
    )


def stability(
    returns: pl.Expr,
) -> pl.Expr:
    r"""
    Stability, the goodness-of-fit of the cumulative log-return path to a straight line.

    The coefficient of determination (:math:`R^2`) of an ordinary-least-squares regression of the cumulative log returns
    on time -- how close to a steady exponential the equity path grows. With :math:`c_t = \sum_{i \le t} \ln(1 + r_i)`
    the cumulative log return and :math:`t` the observation index:

    .. math::

        \mathrm{stability} = R^2\big(\,t,\; c_t\,\big) = \operatorname{corr}(t, c_t)^2 \in [0, 1].

    A value near one means the strategy compounds at a near-constant rate; a low value means an erratic path.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`); each must exceed
            ``-1`` (a return of ``-100%`` or worse makes the cumulative log undefined).

    Returns:
        A single ``Float64`` value in ``[0, 1]``: the trend stability (one value in ``select``, one per group under
        ``.over``). ``null`` when fewer than two returns are present (the regression is undefined).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped, and the time index is taken over the retained observations (so an
          interior gap does not leave a hole in the regression).
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **Out of domain** — a return at or below ``-1`` makes the cumulative log undefined, yielding ``NaN``.
        - **Flat path** — an all-zero (or otherwise perfectly flat) cumulative log has no variance to explain, so the
          result is ``NaN``; fewer than two observations yields ``null``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``stability(pl.col("returns")).over("ticker")``.

    See Also:
        - :func:`cagr`: The growth rate whose steadiness this measures.
        - :func:`~pomata.indicators.linear_regression`: The least-squares trend line whose goodness-of-fit this scores.
        - :func:`~pomata.indicators.linear_regression_slope`: The slope of that same least-squares trend.

    References:
        - https://en.wikipedia.org/wiki/Coefficient_of_determination

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import stability
        >>>
        >>> frame = pl.DataFrame({"returns": [0.01, 0.012, 0.009, 0.011, 0.013, 0.008, 0.01, 0.012]})
        >>> frame.select(stability(pl.col("returns")).round(4)).item()
        0.9984

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 8 + ["B"] * 8,
        ...         "returns": [
        ...             0.01,
        ...             0.012,
        ...             0.009,
        ...             0.011,
        ...             0.013,
        ...             0.008,
        ...             0.01,
        ...             0.012,
        ...             0.02,
        ...             -0.01,
        ...             0.03,
        ...             -0.02,
        ...             0.025,
        ...             -0.015,
        ...             0.018,
        ...             -0.012,
        ...         ],
        ...     }
        ... )
        >>> reduced = stability(pl.col("returns")).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [0.3855, 0.9984]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.01, 0.012, None, 0.009, float("nan"), 0.011]})
        >>> frame.select(stability(pl.col("returns")).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    defined = returns.drop_nulls()
    cumulative_log = defined.log1p().cum_sum()
    index = pl.int_range(0, defined.len(), dtype=pl.Int64)
    # Clamp to the documented [0, 1] upper bound: a perfect linear fit gives a correlation that can round to 1 + eps,
    # whose square would otherwise escape the bound by a floating-point ulp.
    r_squared = (pl.corr(index, cumulative_log) ** 2).clip(upper_bound=1.0)
    return (
        pl.when(defined.len() < _MINIMUM_REGRESSION_POINTS)
        .then(None)
        .when(returns.is_nan().any())
        .then(pl.lit(float("nan")))
        .when(cumulative_log.n_unique() <= 1)
        .then(pl.lit(float("nan")))
        .otherwise(r_squared)
        .name.keep()
    )


def total_return(
    equity_curve: pl.Expr,
) -> pl.Expr:
    r"""
    Total Return, the overall compounded return of an equity curve.

    The final value of the equity curve relative to its unit start — the cumulative gain (or loss) over the whole
    series:

    .. math::

        R = E_N - 1,

    where :math:`E_N` is the final growth factor (e.g. :math:`1.25` for a 25% total gain). Because the curve compounds a
    unit of capital, the final value already is the total growth multiple.

    Args:
        equity_curve: Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive; its ``N``
            values are ``N`` period growth factors, and its final value is the total growth multiple.

    Returns:
        A single ``Float64`` value: the total compounded return (one value in ``select``, one per group under
        ``.over``). ``null`` when there are no observations.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — ``null`` equities are skipped; the result uses the last defined equity. An all-null series yields
          ``null``.
        - **NaN** — a ``NaN`` anywhere yields ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the result is computed per
          series, e.g. ``total_return(pl.col("equity")).over("ticker")``.

    See Also:
        - :func:`cagr`: The annualized (per-year) form of this total growth.
        - :func:`total_return_rolling`: The windowed twin, over each trailing window.
        - :func:`~pomata.pnl.equity_curve`: The pnl builder that produces the input curve.

    References:
        - https://en.wikipedia.org/wiki/Total_return

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import total_return
        >>>
        >>> frame = pl.DataFrame({"equity": [1.1, 1.045, 1.254, 1.3794]})
        >>> frame.select(total_return(pl.col("equity")).round(4)).item()
        0.3794

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 4 + ["B"] * 4,
        ...         "equity_curve": [1.1, 1.045, 1.254, 1.3794, 1.02, 1.05, 0.98, 1.12],
        ...     }
        ... )
        >>> reduced = total_return(pl.col("equity_curve")).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [0.12, 0.3794]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"equity_curve": [1.1, 1.045, None, 1.254, float("nan"), 1.3794]})
        >>> frame.select(total_return(pl.col("equity_curve")).round(4)).item()
        nan
    """
    equity_curve = float64_expr(equity_curve)
    growth = equity_curve.drop_nulls().last() - 1
    return pl.when(equity_curve.is_nan().any()).then(pl.lit(float("nan"))).otherwise(growth).name.keep()


def total_return_rolling(
    equity_curve: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Rolling Total Return over a window — the windowed twin of :func:`total_return`.

    The growth over each trailing window, normalized by the window's own starting equity:

    .. math::

        R_t = \frac{E_t}{E_{t-n+1}} - 1, \qquad n = \text{window},

    where :math:`E` is the equity curve. Unlike :func:`total_return` (which assumes a unit start over the whole series),
    the rolling form measures the return of the last ``window`` bars and so divides by the window's first equity. As an
    endpoint quantity it depends only on the first and last equity of the window, not the path between them.

    Args:
        equity_curve: Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.
        window: Number of observations in the moving window. Must be ``>= 2``.

    Returns:
        The rolling total return for each row, the same length as the input. The first ``window - 1`` rows are ``null``
        (warm-up): the window must reach back ``window`` rows before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``.

    Note:
        **Correctness** -- each window matches an independent reference oracle (the endpoint ratio less one).

        **Edge-case behavior:**

        - **Null** — a ``null`` at either window endpoint yields ``null``; being an endpoint quantity, an interior
          ``null`` does not affect the result.
        - **NaN** — a ``NaN`` at either endpoint propagates, yielding ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` so the window never spans series boundaries.

    See Also:
        - :func:`total_return`: The whole-series reducing form.
        - :func:`cagr_rolling`: The annualized (per-year) windowed counterpart.
        - :func:`cagr`: The whole-series, annualized growth rate.

    References:
        - https://en.wikipedia.org/wiki/Total_return

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import total_return_rolling
        >>>
        >>> frame = pl.DataFrame({"equity": [1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25]})
        >>> frame.select(total_return_rolling(pl.col("equity"), 3).round(4))["equity"].to_list()
        [None, None, 0.05, 0.0909, 0.0952, 0.0833, 0.087]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "equity": [1.0, 1.1, 1.05, 1.2, 1.15, 1.0, 1.02, 1.08, 1.05, 1.12],
        ...     }
        ... )
        >>> rolled = total_return_rolling(pl.col("equity"), 3).over("ticker").round(4)
        >>> frame.select(rolled.alias("m"))["m"].to_list()
        [None, None, 0.05, 0.0909, 0.0952, None, None, 0.08, 0.0294, 0.037]

        A ``null`` or ``NaN`` at a window endpoint propagates, while a ``NaN`` interior to a window is ignored:

        >>> frame = pl.DataFrame({"equity": [None, 1.1, 1.05, 1.2, float("nan"), 1.3, 1.25]})
        >>> frame.select(total_return_rolling(pl.col("equity"), 3).round(4))["equity"].to_list()
        [None, None, None, 0.0909, nan, 0.0833, nan]
    """
    equity_curve = float64_expr(equity_curve)
    validate_window(window, minimum=2)
    return equity_curve / equity_curve.shift(window - 1) - 1.0
