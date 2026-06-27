"""
Accounting — turning weights/quantities and returns/prices into per-bar strategy P&L and the capital curves.
"""

import polars as pl

from pomata._expr import float64_expr, validate_positive

__all__ = (
    "cumulative_pnl",
    "dividend",
    "equity_curve",
    "pnl_gross",
    "pnl_gross_inverse",
    "pnl_net",
    "returns_gross",
    "returns_net",
    "turnover",
)


def cumulative_pnl(
    returns: pl.Expr,
) -> pl.Expr:
    r"""
    Cumulative P&L, the additive running total of a per-bar P&L (or return) series.

    The plain cumulative sum of the per-bar values to date:

    .. math::

        \mathrm{cumPnL}_t = \sum_{i \le t} x_i.

    P&L in currency is **additive** — you sum dollars, you do not compound them — so for the cash / position flow this
    running sum is your total P&L to date (pair it with :func:`pnl_net`). For the return flow, where capital is
    reinvested, the cumulation is **compounded**: use :func:`equity_curve` instead, which is what "cumulative return"
    conventionally means. The additive name lives here, on the currency P&L, where additive is the standard; the per-bar
    inputs are unchanged either way, only the cumulation differs.

    Args:
        returns: Input per-bar values to cumulate — the strategy's net P&L (e.g. from :func:`pnl_net`) for a currency
            total, or a net-return series for an additive (fixed-notional) return total.

    Returns:
        The running sum for each row, the same length as ``returns``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every
        edge case (missing data, boundaries, and warm-up where applicable) is given a defined behavior, documented
        under **Edge-case behavior** below.

        **Edge-case behavior:**

        - **Null** — a ``null`` return contributes nothing and emits ``null`` at that row, while the running sum carries
          across it unchanged (the cumulation skips the gap rather than breaking on it).
        - **NaN** — a ``NaN`` return propagates into the running sum and every later row stays ``NaN`` (it is a real
          value that contaminates the total, unlike a ``null`` gap).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the running sum restarts per
          series and never carries across boundaries, e.g. ``cumulative_pnl(pl.col("returns")).over("ticker")``.

    See Also:
        - :func:`equity_curve`: The compounded (reinvested) return-flow cumulation, a product of one-plus-returns.
        - :func:`pnl_net`: The per-bar net P&L this typically cumulates in the cash flow.

    References:
        - https://en.wikipedia.org/wiki/Rate_of_return

    Examples:
        >>> import polars as pl
        >>> from pomata.pnl import cumulative_pnl
        >>> frame = pl.DataFrame({"returns": [0.1, -0.05, 0.2, 0.1]})
        >>> frame.select(cumulative_pnl(pl.col("returns")).round(4).alias("cumulative_pnl"))["cumulative_pnl"].to_list()
        [0.1, 0.05, 0.25, 0.35]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker accumulates independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "returns": [0.1, 0.2, -0.05, 0.0, 0.1, 0.1],
        ...     }
        ... )
        >>> frame.with_columns(cumulative_pnl(pl.col("returns")).over("ticker").round(4).alias("c"))["c"].to_list()
        [0.1, 0.3, 0.25, 0.0, 0.1, 0.2]

        A ``null`` (skipped, the total carries across it) and a ``NaN`` (which contaminates every later row) make the
        missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.1, None, 0.2, float("nan"), 0.1]})
        >>> frame.select(cumulative_pnl(pl.col("returns")).round(4).alias("cumulative_pnl"))["cumulative_pnl"].to_list()
        [0.1, None, 0.3, nan, nan]
    """
    returns = float64_expr(returns)
    # Plain cumulative sum: a null is skipped (emits null, the total carries across it), a NaN propagates -- the Polars
    # cum_sum semantics documented in the Note.
    return returns.cum_sum()


def dividend(
    quantity: pl.Expr,
    dividend_per_share: pl.Expr,
) -> pl.Expr:
    r"""
    Dividend Cashflow, the per-bar dividend income (or expense) of a held quantity.

    The quantity held times the dividend paid per share that bar — the cash a position receives when the instrument
    distributes a dividend (a long receives, a short pays):

    .. math::

        d_t = q_t \cdot \mathrm{dps}_t, \qquad q = \text{quantity},\ \mathrm{dps} = \text{dividend per share}.

    The dividend per share is a per-bar series, zero on ordinary bars and the cash amount on ex-dividend bars. This is a
    holding cashflow on the **income** side (not a cost): add it to the gross PnL (e.g.
    ``pnl_gross(...) + dividend(...)``) before subtracting costs.

    Args:
        quantity: Signed position size in units / shares / contracts held over the bar; a long (positive) receives the
            dividend, a short (negative) pays it.
        dividend_per_share: Dividend paid per share for the bar (e.g. ``pl.col("dividend")``); zero on ordinary bars.

    Returns:
        The dividend cashflow for each row, the same length as the inputs.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every
        edge case (missing data, boundaries, and warm-up where applicable) is given a defined behavior, documented
        under **Edge-case behavior** below.

        **Edge-case behavior:**

        - **Null** — a ``null`` in either input makes that row ``null`` (``null`` takes precedence over ``NaN``).
        - **NaN** — a ``NaN`` in either input (with no ``null``) propagates, yielding ``NaN`` for that row.
        - **Partitioning** — the product is elementwise (each row uses only its own pair), so ``.over(...)`` partitions
          identically and is optional here, unlike the lagged / cumulative functions.

    See Also:
        - :func:`cost_borrow`: The equity holding cashflow on the cost side (short-borrow).
        - :func:`pnl_gross`: The gross position PnL this income is added to.

    References:
        - https://en.wikipedia.org/wiki/Dividend

    Examples:
        >>> import polars as pl
        >>> from pomata.pnl import dividend
        >>> frame = pl.DataFrame(
        ...     {
        ...         "quantity": [100.0, 100.0, 100.0, 0.0, -50.0],
        ...         "dividend_per_share": [0.0, 0.0, 0.5, 0.0, 0.5],
        ...     }
        ... )
        >>> expr = dividend(pl.col("quantity"), pl.col("dividend_per_share")).round(4)
        >>> frame.select(expr.alias("dividend"))["dividend"].to_list()
        [0.0, 0.0, 50.0, 0.0, -25.0]

        A ``null`` (which propagates) and a ``NaN`` make the missing-data handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "quantity": [100.0, None, 100.0, float("nan")],
        ...         "dividend_per_share": [0.0, 0.5, float("nan"), 0.5],
        ...     }
        ... )
        >>> expr = dividend(pl.col("quantity"), pl.col("dividend_per_share")).round(4)
        >>> frame.select(expr.alias("dividend"))["dividend"].to_list()
        [0.0, None, nan, nan]
    """
    quantity = float64_expr(quantity)
    dividend_per_share = float64_expr(dividend_per_share)
    # Pure elementwise product: the held quantity times the per-share dividend; null propagates (taking precedence over
    # NaN), NaN propagates.
    return quantity * dividend_per_share


def equity_curve(
    returns: pl.Expr,
) -> pl.Expr:
    r"""
    Equity Curve, the compounded growth of one unit of capital over a return series.

    The cumulative product of the gross returns (one plus each per-bar return) — the value of one unit of capital that
    **reinvests** its P&L each bar, so every return compounds on the grown capital:

    .. math::

        \mathrm{equity}_t = \prod_{i \le t} (1 + r_i).

    This is the standard equity curve and the multiplicative twin of :func:`cumulative_pnl`: use this when the P&L is
    reinvested (the total-return convention) and the additive :func:`cumulative_pnl` when the notional is held fixed.
    It is also the natural input to a drawdown, so the metrics family consumes it directly.

    Args:
        returns: Input per-bar returns to compound, typically the strategy's gross or net returns (e.g. from
            :func:`returns_gross`).

    Returns:
        The compounded equity for each row, the same length as ``returns``, expressed as a growth factor relative to a
        starting capital of ``1`` (multiply by the starting capital for a currency curve).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every
        edge case (missing data, boundaries, and warm-up where applicable) is given a defined behavior, documented
        under **Edge-case behavior** below.

        **Edge-case behavior:**

        - **Null** — a ``null`` return emits ``null`` at that row while the running product carries across it unchanged
          (a missing bar contributes a neutral factor of one rather than breaking the curve); a leading warm-up ``null``
          (e.g. the first row of :func:`returns_simple`) therefore stays ``null`` and the curve begins at the first
          defined return.
        - **NaN** — a ``NaN`` return propagates into the running product and every later row stays ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the product restarts per series
          and never carries across boundaries, e.g. ``equity_curve(pl.col("returns")).over("ticker")``.

    See Also:
        - :func:`cumulative_pnl`: The additive (fixed-notional) twin, a cumulative sum of returns.
        - :func:`returns_gross`: The per-bar strategy returns this typically compounds.

    References:
        - https://en.wikipedia.org/wiki/Rate_of_return

    Examples:
        >>> import polars as pl
        >>> from pomata.pnl import equity_curve
        >>> frame = pl.DataFrame({"returns": [0.1, -0.05, 0.2, 0.1]})
        >>> frame.select(equity_curve(pl.col("returns")).round(4).alias("equity"))["equity"].to_list()
        [1.1, 1.045, 1.254, 1.3794]

        A leading warm-up ``null`` (as produced by :func:`returns_simple`) stays ``null``; the curve begins at the first
        defined return:

        >>> frame = pl.DataFrame({"returns": [None, 0.1, 0.2, -0.05]})
        >>> frame.select(equity_curve(pl.col("returns")).round(4).alias("equity"))["equity"].to_list()
        [None, 1.1, 1.32, 1.254]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker compounds independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "returns": [0.1, 0.2, -0.05, 0.0, 0.1, 0.1],
        ...     }
        ... )
        >>> frame.with_columns(equity_curve(pl.col("returns")).over("ticker").round(4).alias("e"))["e"].to_list()
        [1.1, 1.32, 1.254, 1.0, 1.1, 1.21]
    """
    returns = float64_expr(returns)
    # Cumulative product of one-plus-returns: a null is skipped (emits null, the product carries across it), a NaN
    # propagates -- the Polars cum_prod semantics documented in the Note.
    return (1.0 + returns).cum_prod()


def pnl_gross(
    quantity: pl.Expr,
    price: pl.Expr,
    *,
    multiplier: float = 1.0,
) -> pl.Expr:
    r"""
    Gross Position PnL, the per-bar mark-to-market profit and loss of a held quantity.

    The signed quantity held over a bar times the bar's price change times the contract multiplier — the strategy's
    gross P&L for that bar in the price's currency, before transaction costs. This is the cash / position flow's
    counterpart to :func:`returns_gross`: use it when you hold a **quantity** of an instrument at a **price** (so the
    instrument's multiplier, and later dividends / funding / FX, can be booked honestly), rather than a weight and a
    return.

    .. math::

        \mathrm{pnl}^{\mathrm{gross}}_t = q_t \cdot (P_t - P_{t-1}) \cdot m, \qquad q = \text{quantity},\ m =
        \text{multiplier}.

    Summed over time it is the **total** mark-to-market PnL (realized plus unrealized combined); pomata does not split
    realized from unrealized (that needs cost-basis lot accounting, which a vectorized ``pl.Expr`` does not carry).

    Args:
        quantity: Signed position size in units / shares / contracts held over the bar (e.g. ``100``, ``-2``).
        price: Instrument price series (e.g. ``pl.col("close")``); must share a length and alignment with ``quantity``.
        multiplier: Contract multiplier / point value (e.g. ``50`` for an E-mini S&P future); ``1.0`` for cash equity
            and spot. Must be a finite number ``> 0``.

    Returns:
        The gross PnL for each row, the same length as the inputs. The first value is ``null`` (warm-up): the previous
        price ``price.shift(1)`` is undefined for the first row, so no price change can be measured there.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``multiplier`` is not a finite number ``> 0`` (i.e. ``<= 0``, ``NaN``, or ``±inf``).

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every
        edge case (missing data, boundaries, and warm-up where applicable) is given a defined behavior, documented
        under **Edge-case behavior** below.

        **No lookahead (alignment is the caller's):** the PnL assumes ``quantity`` at row ``t`` is the position held
        over the price change into row ``t``. To stay lookahead-free, that quantity must depend only on information
        available before that price; if it is decided on the same bar's close, lag it by one bar
        (``pnl_gross(quantity.shift(1), price)``). Nothing is shifted for you, so a quantity you have already aligned is
        never double-shifted.

        **Edge-case behavior:**

        - **Null** — a ``null`` in ``quantity``, ``price``, or the previous ``price`` makes that row ``null`` (``null``
          takes precedence over ``NaN``).
        - **NaN** — a ``NaN`` in either input (with no ``null``) propagates, yielding ``NaN`` for that row.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the one-bar price change never
          reaches across series boundaries, e.g. ``pnl_gross(pl.col("quantity"), pl.col("price")).over("ticker")``.

    See Also:
        - :func:`returns_gross`: The return-flow counterpart (weight times asset return).
        - :func:`pnl_net`: Subtracts the composed cost from this gross PnL.
        - :func:`pnl_gross_inverse`: The coin-margined (inverse-contract) version, nonlinear in price.

    References:
        - https://en.wikipedia.org/wiki/Mark-to-market_accounting
        - https://en.wikipedia.org/wiki/Rate_of_return

    Examples:
        >>> import polars as pl
        >>> from pomata.pnl import pnl_gross
        >>> frame = pl.DataFrame(
        ...     {
        ...         "quantity": [10.0, 10.0, -5.0, -5.0, 20.0],
        ...         "price": [100.0, 102.0, 101.0, 104.0, 103.0],
        ...     }
        ... )
        >>> frame.select(pnl_gross(pl.col("quantity"), pl.col("price")).round(4).alias("pnl"))["pnl"].to_list()
        [None, 20.0, 5.0, -15.0, -20.0]

        A futures contract multiplier scales the PnL (here ``50`` per point):

        >>> expr = pnl_gross(pl.col("quantity"), pl.col("price"), multiplier=50.0).round(4)
        >>> frame.select(expr.alias("pnl"))["pnl"].to_list()
        [None, 1000.0, 250.0, -750.0, -1000.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "quantity": [10.0, 10.0, -5.0, 2.0, 2.0, 2.0],
        ...         "price": [100.0, 102.0, 101.0, 50.0, 51.0, 49.0],
        ...     }
        ... )
        >>> frame.with_columns(pnl_gross(pl.col("quantity"), pl.col("price")).over("ticker").round(4).alias("p"))[
        ...     "p"
        ... ].to_list()
        [None, 20.0, 5.0, None, 2.0, -4.0]

        A ``null`` (which propagates) and a ``NaN`` make the missing-data handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "quantity": [10.0, None, -5.0, 20.0],
        ...         "price": [100.0, 102.0, float("nan"), 104.0],
        ...     }
        ... )
        >>> frame.select(pnl_gross(pl.col("quantity"), pl.col("price")).round(4).alias("pnl"))["pnl"].to_list()
        [None, None, nan, nan]
    """
    quantity = float64_expr(quantity)
    price = float64_expr(price)
    validate_positive(multiplier, "multiplier")
    # Per-bar mark-to-market: quantity held over the one-bar price change, times the contract multiplier. Row 0 is null
    # (no prior price); null propagates (taking precedence over NaN), NaN propagates; no lag is applied (see the Note).
    return quantity * (price - price.shift(1)) * multiplier


def pnl_gross_inverse(
    quantity: pl.Expr,
    price: pl.Expr,
    *,
    multiplier: float = 1.0,
) -> pl.Expr:
    r"""
    Gross Inverse-Contract PnL (coin-margined), the per-bar mark-to-market profit and loss settled in the base coin.

    An inverse (coin-margined) perpetual or futures contract carries a fixed notional in the **quote** currency (e.g.
    ``1`` USD per contract) but settles its profit and loss in the **base** coin (e.g. BTC). Its value per contract is
    therefore the reciprocal of the price, so the PnL is the signed quantity times the contract notional times the
    one-bar change in that reciprocal — nonlinear in the price, the one case the linear :func:`pnl_gross` cannot
    express:

    .. math::

        \mathrm{pnl}^{\mathrm{gross}}_t = q_t \cdot m \cdot \left( \frac{1}{P_{t-1}} - \frac{1}{P_t} \right), \qquad
        q = \text{quantity},\ m = \text{multiplier}.

    A long gains as the price rises (the reciprocal falls), exactly as for a linear contract, but the coin-denominated
    payoff is concave in the price for a long (convex for a short), since the contract's coin value ``1 / P`` is convex.
    Summed over time it is the **total** mark-to-market PnL (realized plus unrealized
    combined); pomata does not split realized from unrealized (that needs cost-basis lot accounting, which a vectorized
    ``pl.Expr`` does not carry).

    Args:
        quantity: Signed position size in units / shares / contracts held over the bar (e.g. ``100``, ``-2``).
        price: Instrument price series, the quote per base unit (e.g. USD per BTC, ``pl.col("close")``); must be
            strictly positive (see the **Domain** note) and share a length and alignment with ``quantity``.
        multiplier: Contract notional in the quote currency — the quote value of one contract (e.g. ``1`` USD for an
            inverse BTC/USD perpetual, ``100`` on some venues); ``1.0`` for a one-unit contract. Must be a finite
            number ``> 0``.

    Returns:
        The gross PnL for each row, in the base coin, the same length as the inputs. The first value is ``null``
        (warm-up): the previous price ``price.shift(1)`` is undefined for the first row, so no price change can be
        measured there.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``multiplier`` is not a finite number ``> 0`` (i.e. ``<= 0``, ``NaN``, or ``±inf``).

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every
        edge case (missing data, boundaries, and warm-up where applicable) is given a defined behavior, documented
        under **Edge-case behavior** below.

        **No lookahead (alignment is the caller's):** the PnL assumes ``quantity`` at row ``t`` is the position held
        over the price change into row ``t``. To stay lookahead-free, that quantity must depend only on information
        available before that price; if it is decided on the same bar's close, lag it by one bar
        (``pnl_gross_inverse(quantity.shift(1), price)``). Nothing is shifted for you, so a quantity you have already
        aligned is never double-shifted.

        **Domain** — the payoff is defined on strictly positive prices. Following IEEE-754 division, a zero current
        price makes ``1 / P_t`` infinite, so the bar is ``-inf`` (a long) or ``+inf`` (a short); a zero previous price
        makes ``1 / P_{t-1}`` infinite, so the bar takes the opposite sign; and a negative price yields a finite but
        economically meaningless value (the reciprocal flips sign). These are the documented and intended boundary
        values rather than an error.

        **Edge-case behavior:**

        - **Null** — a ``null`` in ``quantity``, ``price``, or the previous ``price`` makes that row ``null`` (``null``
          takes precedence over ``NaN``).
        - **NaN** — a ``NaN`` in either input (with no ``null``) propagates, yielding ``NaN`` for that row.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the one-bar price change never
          reaches across series boundaries, e.g.
          ``pnl_gross_inverse(pl.col("quantity"), pl.col("price")).over("ticker")``.

    See Also:
        - :func:`pnl_gross`: The linear (quote-margined) counterpart; use it when the contract settles in the quote
          currency rather than the base coin.
        - :func:`pnl_net`: Subtracts the composed cost from this gross PnL.
        - :func:`cost_funding`: The perpetual-swap funding leg, the companion holding cost.

    References:
        - https://en.wikipedia.org/wiki/Perpetual_futures
        - https://en.wikipedia.org/wiki/Mark-to-market_accounting

    Examples:
        >>> import polars as pl
        >>> from pomata.pnl import pnl_gross_inverse
        >>> frame = pl.DataFrame(
        ...     {
        ...         "quantity": [1.0, 1.0, -2.0, -2.0, 3.0],
        ...         "price": [100.0, 110.0, 105.0, 120.0, 115.0],
        ...     }
        ... )
        >>> expr = pnl_gross_inverse(pl.col("quantity"), pl.col("price")).round(6)
        >>> frame.select(expr.alias("pnl"))["pnl"].to_list()
        [None, 0.000909, 0.000866, -0.002381, -0.001087]

        A contract notional scales the coin PnL (here ``100`` USD per contract):

        >>> expr = pnl_gross_inverse(pl.col("quantity"), pl.col("price"), multiplier=100.0).round(4)
        >>> frame.select(expr.alias("pnl"))["pnl"].to_list()
        [None, 0.0909, 0.0866, -0.2381, -0.1087]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "quantity": [1.0, 1.0, -2.0, 2.0, 2.0, 2.0],
        ...         "price": [100.0, 110.0, 105.0, 50.0, 55.0, 52.0],
        ...     }
        ... )
        >>> frame.with_columns(
        ...     pnl_gross_inverse(pl.col("quantity"), pl.col("price")).over("ticker").round(6).alias("p")
        ... )["p"].to_list()
        [None, 0.000909, 0.000866, None, 0.003636, -0.002098]

        A ``null`` (which propagates) and a ``NaN`` make the missing-data handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "quantity": [1.0, None, -2.0, 3.0],
        ...         "price": [100.0, 110.0, float("nan"), 120.0],
        ...     }
        ... )
        >>> expr = pnl_gross_inverse(pl.col("quantity"), pl.col("price")).round(6)
        >>> frame.select(expr.alias("pnl"))["pnl"].to_list()
        [None, None, nan, nan]
    """
    quantity = float64_expr(quantity)
    price = float64_expr(price)
    validate_positive(multiplier, "multiplier")
    # Per-bar mark-to-market in the base coin: the contract value is the reciprocal of the price, so the PnL is the
    # quantity times the notional times the one-bar change in 1/price. Row 0 is null (no prior price); null propagates
    # (taking precedence over NaN), NaN propagates; no lag is applied (see the Note).
    return quantity * multiplier * (1.0 / price.shift(1) - 1.0 / price)


def pnl_net(
    pnl_gross: pl.Expr,
    cost: pl.Expr,
) -> pl.Expr:
    r"""
    Net Position PnL, the gross position PnL after transaction costs.

    The gross per-bar position PnL minus the per-bar transaction cost, both in the account currency — the cash flow's
    net P&L, the counterpart of :func:`returns_net`:

    .. math::

        \mathrm{pnl}^{\mathrm{net}}_t = \mathrm{pnl}^{\mathrm{gross}}_t - c_t.

    A pure elementwise subtraction with no built-in cost model: the caller composes the cost from the cost components
    (summing several with ``+``) and passes it, e.g.
    ``pnl_net(pnl_gross(quantity, price), cost_per_share(quantity, fee) + cost_notional(quantity, price, rate))``.

    Args:
        pnl_gross: Gross per-bar position PnL, typically from :func:`pnl_gross`.
        cost: Per-bar transaction cost in the same currency, typically from :func:`cost_per_share` (sum several with
            ``+``).

    Returns:
        The net PnL for each row, the same length as the inputs.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every
        edge case (missing data, boundaries, and warm-up where applicable) is given a defined behavior, documented
        under **Edge-case behavior** below.

        **Edge-case behavior:**

        - **Null** — a ``null`` in either input makes that row ``null`` (``null`` takes precedence over ``NaN``).
        - **NaN** — a ``NaN`` in either input (with no ``null``) propagates, yielding ``NaN`` for that row.
        - **Partitioning** — the subtraction is elementwise (each row uses only its own pair), so ``.over(...)``
          partitions identically and is optional here, unlike the lagged / cumulative functions.

    See Also:
        - :func:`pnl_gross`: The gross position PnL this nets costs from.
        - :func:`cost_per_share`: A usual source of ``cost`` (sum several cost components with ``+``).
        - :func:`cumulative_pnl`: Cumulates these net PnL into a running currency total.

    References:
        - https://en.wikipedia.org/wiki/Mark-to-market_accounting
        - https://en.wikipedia.org/wiki/Rate_of_return

    Examples:
        >>> import polars as pl
        >>> from pomata.pnl import pnl_net
        >>> frame = pl.DataFrame(
        ...     {
        ...         "pnl_gross": [20.0, 5.0, -15.0, -20.0, 8.0],
        ...         "cost": [2.0, 0.0, 3.0, 0.0, 1.0],
        ...     }
        ... )
        >>> frame.select(pnl_net(pl.col("pnl_gross"), pl.col("cost")).round(4).alias("pnl_net"))["pnl_net"].to_list()
        [18.0, 5.0, -18.0, -20.0, 7.0]

        A ``null`` (which propagates) and a ``NaN`` make the missing-data handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "pnl_gross": [20.0, None, -15.0, float("nan")],
        ...         "cost": [2.0, 3.0, float("nan"), 0.0],
        ...     }
        ... )
        >>> frame.select(pnl_net(pl.col("pnl_gross"), pl.col("cost")).round(4).alias("pnl_net"))["pnl_net"].to_list()
        [18.0, None, nan, nan]
    """
    pnl_gross = float64_expr(pnl_gross)
    cost = float64_expr(cost)
    # Pure elementwise subtraction: null propagates (taking precedence over NaN), NaN propagates; no cost model is baked
    # in, so the caller composes and sums the cost components (see the Note).
    return pnl_gross - cost


def returns_gross(
    weight: pl.Expr,
    asset_returns: pl.Expr,
) -> pl.Expr:
    r"""
    Gross Strategy Returns, the per-bar return of a weight before costs.

    The signed weight times the asset's per-bar return — the strategy's gross return for that bar, before any
    transaction costs:

    .. math::

        r^{\mathrm{gross}}_t = w_t \cdot r_t, \qquad w = \text{weight}.

    Because simple returns aggregate across assets (a portfolio's return is the weighted sum of its constituents'), this
    per-leg product is the building block of a multi-asset gross return: sum it over the legs of a panel. It is a pure
    elementwise multiply with **no built-in lag**: each row pairs ``weight`` with ``asset_returns`` at the same index,
    so the caller is responsible for alignment.

    Args:
        weight: Signed weight, the fraction of capital held (e.g. ``1.0`` fully long, ``-0.5`` half short);
            ``|weight| > 1`` is leverage.
        asset_returns: Per-bar asset returns, typically from :func:`returns_simple` (e.g.
            ``returns_simple(pl.col("close"))``).

    Returns:
        The gross strategy return for each row, the same length as the inputs. There is no window and no warm-up of its
        own: every row is the product of its own ``weight`` and ``asset_returns`` (so a warm-up ``null`` is inherited
        only from the inputs, e.g. the first row of :func:`returns_simple`).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every
        edge case (missing data, boundaries, and warm-up where applicable) is given a defined behavior, documented
        under **Edge-case behavior** below.

        **No lookahead (alignment is the caller's):** the product assumes ``weight`` at row ``t`` is the weight held
        over ``asset_returns`` at row ``t``. To stay lookahead-free, that weight must depend only on information
        available **before** that return; if your weight is decided on the same bar that closes the return, lag it by
        one bar -- ``returns_gross(weight.shift(1), asset_returns)`` -- so the weight reflects only the prior close.
        Nothing is shifted for you, so a weight you have already aligned is never double-shifted.

        **Edge-case behavior:**

        - **Null** — a ``null`` in either input makes that row ``null`` (the product propagates ``null``, which takes
          precedence over ``NaN``).
        - **NaN** — a ``NaN`` in either input (with no ``null`` at that row) propagates, yielding ``NaN`` for that row.
        - **Partitioning** — the product is elementwise (each row uses only its own pair), so it is already correct on a
          multi-series panel: ``.over(...)`` partitions identically and is therefore optional here, unlike the
          lagged / cumulative functions where it is required to stop state spanning series boundaries.

    See Also:
        - :func:`returns_simple`: The usual source of ``asset_returns``.
        - :func:`turnover`: The traded fraction of the same ``weight``, the basis for transaction costs.
        - :func:`equity_curve`: Compounds these per-bar returns into a capital curve.

    References:
        - Meucci, A. (2010). Quant Nugget 2: Linear vs. Compounded Returns. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1586656
        - https://en.wikipedia.org/wiki/Rate_of_return

    Examples:
        >>> import polars as pl
        >>> from pomata.pnl import returns_gross
        >>> frame = pl.DataFrame(
        ...     {
        ...         "weight": [1.0, 0.5, -1.0, -1.0, 0.5],
        ...         "asset_returns": [0.02, -0.01, 0.03, -0.02, 0.04],
        ...     }
        ... )
        >>> expr = returns_gross(pl.col("weight"), pl.col("asset_returns")).round(4)
        >>> frame.select(expr.alias("returns_gross"))["returns_gross"].to_list()
        [0.02, -0.005, -0.03, 0.02, 0.02]

        On a multi-ticker panel the product is already per-row, so ``.over`` is optional and shown only for consistency:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "weight": [1.0, -1.0, 0.5, 0.5, 0.5, -1.0],
        ...         "asset_returns": [0.02, 0.03, -0.01, 0.04, -0.02, 0.01],
        ...     }
        ... )
        >>> expr = returns_gross(pl.col("weight"), pl.col("asset_returns")).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("g"))["g"].to_list()
        [0.02, -0.03, -0.005, 0.02, -0.01, -0.01]

        A ``null`` (which propagates through the product) and a ``NaN`` make the missing-data handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "weight": [1.0, None, -1.0, 0.5],
        ...         "asset_returns": [0.02, 0.03, float("nan"), 0.04],
        ...     }
        ... )
        >>> expr = returns_gross(pl.col("weight"), pl.col("asset_returns")).round(4)
        >>> frame.select(expr.alias("returns_gross"))["returns_gross"].to_list()
        [0.02, None, nan, 0.02]
    """
    weight = float64_expr(weight)
    asset_returns = float64_expr(asset_returns)
    # Pure elementwise product: null propagates (taking precedence over NaN), NaN propagates; no lag is applied, so the
    # caller owns alignment (see the Note).
    return weight * asset_returns


def returns_net(
    returns_gross: pl.Expr,
    cost: pl.Expr,
) -> pl.Expr:
    r"""
    Net Strategy Returns, the gross return after transaction costs.

    The gross per-bar strategy return minus the per-bar transaction cost — the strategy's net return, which is the
    series the performance and risk metrics consume:

    .. math::

        r^{\mathrm{net}}_t = r^{\mathrm{gross}}_t - c_t.

    It is a pure elementwise subtraction with no built-in cost model: the caller composes the cost from the cost
    components (summing several with ``+``) and passes it, e.g.
    ``returns_net(returns_gross(weight, asset_returns), cost_proportional(weight, rate))``.

    Args:
        returns_gross: Gross per-bar strategy returns, typically from :func:`returns_gross`.
        cost: Per-bar transaction cost as a return drag, typically from :func:`cost_proportional` (sum several with
            ``+``).

    Returns:
        The net strategy return for each row, the same length as the inputs. There is no window and no warm-up of its
        own: every row is ``returns_gross`` minus ``cost`` at that row.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every
        edge case (missing data, boundaries, and warm-up where applicable) is given a defined behavior, documented
        under **Edge-case behavior** below.

        **Edge-case behavior:**

        - **Null** — a ``null`` in either input makes that row ``null`` (the subtraction propagates ``null``, which
          takes precedence over ``NaN``).
        - **NaN** — a ``NaN`` in either input (with no ``null`` at that row) propagates, yielding ``NaN`` for that row.
        - **Partitioning** — the subtraction is elementwise (each row uses only its own pair), so it is already correct
          on a multi-series panel: ``.over(...)`` partitions identically and is therefore optional here, unlike the
          lagged / cumulative functions where it is required.

    See Also:
        - :func:`returns_gross`: The gross return this nets costs from.
        - :func:`cost_proportional`: The usual source of ``cost`` (a proportional, bps-of-notional fee).
        - :func:`cost_slippage`: A second cost component (a fixed half-spread); sum it with the others.
        - :func:`equity_curve`: Compounds these net returns into a capital curve.

    References:
        - https://en.wikipedia.org/wiki/Rate_of_return
        - https://en.wikipedia.org/wiki/Transaction_cost

    Examples:
        >>> import polars as pl
        >>> from pomata.pnl import returns_net
        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns_gross": [0.05, -0.02, 0.03, 0.01, 0.0],
        ...         "cost": [0.0005, 0.0015, 0.0005, 0.0, 0.0005],
        ...     }
        ... )
        >>> expr = returns_net(pl.col("returns_gross"), pl.col("cost")).round(4)
        >>> frame.select(expr.alias("returns_net"))["returns_net"].to_list()
        [0.0495, -0.0215, 0.0295, 0.01, -0.0005]

        A ``null`` (which propagates through the subtraction) and a ``NaN`` make the missing-data handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "returns_gross": [0.05, None, 0.03, float("nan")],
        ...         "cost": [0.0005, 0.0015, float("nan"), 0.0],
        ...     }
        ... )
        >>> expr = returns_net(pl.col("returns_gross"), pl.col("cost")).round(4)
        >>> frame.select(expr.alias("returns_net"))["returns_net"].to_list()
        [0.0495, None, nan, nan]
    """
    returns_gross = float64_expr(returns_gross)
    cost = float64_expr(cost)
    # Pure elementwise subtraction: null propagates (taking precedence over NaN), NaN propagates; no cost model is
    # baked in, so the caller composes and sums the cost components (see the Note).
    return returns_gross - cost


def turnover(
    weight: pl.Expr,
) -> pl.Expr:
    r"""
    Turnover, the traded fraction of capital between consecutive bars.

    The absolute change in the weight from one bar to the next — how much was bought or sold to move from the
    previous weight to the current one, as a fraction of capital:

    .. math::

        \mathrm{turnover}_t = \lvert w_t - w_{t-1} \rvert, \qquad w = \text{weight}.

    The pre-series weight is taken as flat (``0``), so the first bar is :math:`\lvert w_0 \rvert`: entering the
    initial weight from cash is itself a trade. Turnover is the basis for proportional transaction costs (a cost
    per unit traded), and is a dimensionless churn measure in its own right.

    Args:
        weight: Signed weight, the fraction of capital held (e.g. ``1.0`` fully long, ``-0.5`` half short);
            ``|weight| > 1`` is leverage.

    Returns:
        The traded fraction for each row, the same length as ``weight``. The first row is ``|weight_0|`` (the trade
        from a flat start), not ``null``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every
        edge case (missing data, boundaries, and warm-up where applicable) is given a defined behavior, documented
        under **Edge-case behavior** below.

        **Edge-case behavior:**

        - **Flat start** — the weight before the series is taken as ``0``, so the first row is ``|weight_0|`` rather
          than ``null``; establishing the initial weight from cash is a real trade and carries its cost.
        - **Null** — a ``null`` weight makes its own row ``null`` and also the next row ``null`` (the difference
          references the previous weight), then turnover resumes; ``null`` takes precedence over ``NaN``.
        - **NaN** — a ``NaN`` weight propagates to its own row and the next, yielding ``NaN`` there.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the one-bar difference never
          reaches across series boundaries (and each series gets its own flat start), e.g.
          ``turnover(pl.col("weight")).over("ticker")``.

    See Also:
        - :func:`returns_gross`: The gross return of the same ``weight``.

    References:
        - https://www.investopedia.com/terms/p/portfolioturnover.asp

    Examples:
        >>> import polars as pl
        >>> from pomata.pnl import turnover
        >>> frame = pl.DataFrame({"weight": [0.5, 1.0, -0.5, -0.5, 0.0]})
        >>> frame.select(turnover(pl.col("weight")).round(4).alias("turnover"))["turnover"].to_list()
        [0.5, 0.5, 1.5, 0.0, 0.5]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker starts flat and never differences across the
        boundary:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "weight": [0.5, 1.0, -0.5, 1.0, 1.0, 0.0],
        ...     }
        ... )
        >>> frame.with_columns(turnover(pl.col("weight")).over("ticker").round(4).alias("t"))["t"].to_list()
        [0.5, 0.5, 1.5, 1.0, 0.0, 1.0]

        A ``null`` (which voids its own row and the next) and a ``NaN`` make the missing-data handling visible:

        >>> frame = pl.DataFrame({"weight": [0.5, None, -0.5, float("nan"), 0.0]})
        >>> frame.select(turnover(pl.col("weight")).round(4).alias("turnover"))["turnover"].to_list()
        [0.5, None, None, nan, nan]
    """
    weight = float64_expr(weight)
    # Absolute one-bar change with the pre-series weight taken as flat (fill_value 0.0), so the first row is the
    # |weight_0| entry trade; null propagates to its own row and the next, NaN likewise (see the Note).
    return (weight - weight.shift(1, fill_value=0.0)).abs()
