"""
Returns — the price-to-return transforms that begin the PnL pipeline.
"""

import polars as pl

from pomata._expr import float64_expr

__all__ = ("returns_log", "returns_simple")


def returns_log(
    expr: pl.Expr,
) -> pl.Expr:
    r"""
    Logarithmic Returns, also known as continuously-compounded or log returns.

    The natural logarithm of the gross return — the price relative to the previous observation — measuring the
    instantaneous growth rate that, compounded continuously, takes the series from one bar to the next:

    .. math::

        r_t = \ln\!\left(\frac{P_t}{P_{t-1}}\right) = \ln P_t - \ln P_{t-1}.

    Log returns are the representation that **aggregates across time**: the multi-period log return is the plain sum of
    the single-period log returns, :math:`\sum_t r_t = \ln(P_T / P_0)`, since the logarithm turns the product of gross
    returns into a sum. They are defined on a strictly positive price series and are the natural input to time-series
    models and any horizon-aggregation by addition; for combining holdings into a portfolio at one point in time use
    :func:`returns_simple`, which aggregates across assets instead — the two are not interchangeable.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).

    Returns:
        The log return for each row, the same length as ``expr``. The first value is ``null`` (warm-up) -- the lagged
        term ``expr.shift(1)`` is undefined for the first row, so no return can be measured there.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness**

        The result is checked against an independent reference oracle on every input, and every edge case (missing data,
        boundaries, and warm-up where applicable) is given a defined behavior.

        **Edge-case behavior**

        - **Null** — a ``null`` price makes that row ``null`` (``null`` takes precedence over ``NaN``) — reading two
          endpoints, a ``null`` at the current or the previous row voids the output that references it.
        - **NaN** — a ``NaN`` price yields ``NaN`` for that row — a fixed-lag transform of two endpoints, not a
          recurrence, so a ``NaN`` (like a ``null``) contaminates only the rows that reference it and never latches
          onto the rest of the series.
        - **Domain** — a negative price relative (the prices straddle zero) is outside the logarithm's
          strictly-positive domain, so the result is a loud ``NaN`` — except with both prices negative, where the ratio
          is positive and the log is silently finite, an economically meaningless number the caller must screen for.
        - **Degenerate denominator** — both the price and the previous price zero give a ``0 / 0``, i.e. ``NaN`` (the
          logarithm then carries the ``NaN`` through); a zero price relative logs to ``-inf`` and a positive price over
          a zero previous price logs to ``+inf`` — reported, not clipped (a negative-zero ``-0.0`` previous price swaps
          which zero case applies but does not arise from real price data).
        - **Non-finite input** — an ``inf`` price follows IEEE-754 through the ratio and its logarithm, where two
          consecutive same-sign infinite prices divide to ``inf / inf = NaN`` (the sign, and that indeterminate
          ``inf / inf``, included).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`returns_simple`: The arithmetic sibling, which aggregates across assets rather than across time.
        - :func:`cumulative_pnl`: The additive running total; log returns sum to the total log return over a horizon.
        - :func:`equity_curve`: Compounds the gross returns into the growth path of one unit of capital.

    References:
        - Meucci, A. (2010). "Quant Nugget 2: Linear vs. Compounded Returns." *GARP Risk Professional*, April 2010,
          49-51.
        - https://en.wikipedia.org/wiki/Rate_of_return#Logarithmic_or_continuously_compounded_return

    Examples:
        >>> import polars as pl
        >>> from pomata.pnl import returns_log
        >>>
        >>> frame = pl.DataFrame({"close": [100.0, 102.0, 101.0, 105.0, 104.0, 107.0, 110.0, 108.0, 112.0]})
        >>> frame.select(returns_log(pl.col("close")).round(4).alias("returns_log"))["returns_log"].to_list()
        [None, 0.0198, -0.0099, 0.0388, -0.0096, 0.0284, 0.0277, -0.0183, 0.0364]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 4 + ["B"] * 4,
        ...         "close": [100.0, 105.0, 102.0, 108.0, 50.0, 52.0, 51.0, 55.0],
        ...     }
        ... )
        >>> frame.with_columns(returns_log(pl.col("close")).over("ticker").round(4).alias("r"))["r"].to_list()
        [None, 0.0488, -0.029, 0.0572, None, 0.0392, -0.0194, 0.0755]

        A ``null`` (whose lag voids the next bar too) and a ``NaN`` (which propagates) touch only the positions that
        reference them before the series recovers, making the missing-data handling visible:

        >>> frame = pl.DataFrame({"close": [100.0, 105.0, None, 108.0, 110.0, float("nan"), 113.0, 115.0]})
        >>> frame.select(returns_log(pl.col("close")).round(4).alias("returns_log"))["returns_log"].to_list()
        [None, 0.0488, None, None, 0.0183, nan, nan, 0.0175]
    """
    expr = float64_expr(expr)
    # The natural log of the one-bar price relative; the first row is null (no prior) and the IEEE-754 log carries the
    # boundary values (zero relative -> -inf, negative relative -> NaN) documented in the Note.
    return (expr / expr.shift(1)).log().name.keep()


def returns_simple(
    expr: pl.Expr,
) -> pl.Expr:
    r"""
    Simple Returns, also known as arithmetic or linear returns.

    The fractional change of the series relative to its previous observation — the gross return minus one — the
    everyday "percent return" of a holding over one bar:

    .. math::

        r_t = \frac{P_t}{P_{t-1}} - 1 = \frac{P_t - P_{t-1}}{P_{t-1}}.

    Simple returns are the representation that **aggregates across assets**: a portfolio's single-period return is the
    weighted sum of its constituents' simple returns, :math:`r_p = \sum_i w_i\, r_i`, which makes them the correct input
    for combining holdings and for portfolio construction at one point in time. For aggregating one holding's return
    across many periods by addition use :func:`returns_log` instead — the two are not interchangeable.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).

    Returns:
        The simple return for each row, the same length as ``expr``. The first value is ``null`` (warm-up) -- the lagged
        term ``expr.shift(1)`` is undefined for the first row, so no return can be measured there.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness**

        The result is checked against an independent reference oracle on every input, and every edge case (missing data,
        boundaries, and warm-up where applicable) is given a defined behavior.

        **Edge-case behavior**

        - **Null** — a ``null`` price makes that row ``null`` (``null`` takes precedence over ``NaN``) — reading two
          endpoints, a ``null`` at the current or the previous row voids the output that references it.
        - **NaN** — a ``NaN`` price yields ``NaN`` for that row — a fixed-lag transform of two endpoints, not a
          recurrence, so a ``NaN`` (like a ``null``) contaminates only the rows that reference it and never latches
          onto the rest of the series.
        - **Degenerate denominator** — the previous price is ``0``, so a zero change is a ``0 / 0``, i.e. ``NaN`` —
          while a non-zero change over it is ``+/-inf`` (the sign tracks the change), reported not clipped, and a
          negative-zero ``-0.0`` previous price flips that sign but does not arise from real price data.
        - **Non-finite input** — an ``inf`` price follows IEEE-754 through the ratio and the minus one, where two
          consecutive same-sign infinite prices divide to ``inf / inf = NaN`` (the sign, and that indeterminate
          ``inf / inf``, included).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`returns_log`: The logarithmic sibling, which aggregates across time rather than across assets.
        - :func:`equity_curve`: Compounds the simple returns into the growth path of one unit of capital.
        - :func:`cumulative_pnl`: The additive running total of a per-bar P&L or return series.

    References:
        - Meucci, A. (2010). "Quant Nugget 2: Linear vs. Compounded Returns." *GARP Risk Professional*, April 2010,
          49-51.
        - https://en.wikipedia.org/wiki/Rate_of_return

    Examples:
        >>> import polars as pl
        >>> from pomata.pnl import returns_simple
        >>>
        >>> frame = pl.DataFrame({"close": [100.0, 102.0, 101.0, 105.0, 104.0, 107.0, 110.0, 108.0, 112.0]})
        >>> frame.select(returns_simple(pl.col("close")).round(4).alias("returns_simple"))["returns_simple"].to_list()
        [None, 0.02, -0.0098, 0.0396, -0.0095, 0.0288, 0.028, -0.0182, 0.037]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 4 + ["B"] * 4,
        ...         "close": [100.0, 105.0, 102.0, 108.0, 50.0, 52.0, 51.0, 55.0],
        ...     }
        ... )
        >>> frame.with_columns(returns_simple(pl.col("close")).over("ticker").round(4).alias("r"))["r"].to_list()
        [None, 0.05, -0.0286, 0.0588, None, 0.04, -0.0192, 0.0784]

        A ``null`` (whose lag voids the next bar too) and a ``NaN`` (which propagates) touch only the positions that
        reference them before the series recovers, making the missing-data handling visible:

        >>> frame = pl.DataFrame({"close": [100.0, 105.0, None, 108.0, 110.0, float("nan"), 113.0, 115.0]})
        >>> frame.select(returns_simple(pl.col("close")).round(4).alias("returns_simple"))["returns_simple"].to_list()
        [None, 0.05, None, None, 0.0185, nan, nan, 0.0177]
    """
    expr = float64_expr(expr)
    # The one-bar price relative minus one; the first row is null (no prior) and a zero previous price divides by zero
    # following IEEE-754 (0/0 -> NaN, non-zero/0 -> +/-inf), as documented in the Note.
    return (expr / expr.shift(1) - 1).name.keep()
