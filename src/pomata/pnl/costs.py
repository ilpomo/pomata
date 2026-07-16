"""
Transaction costs — composable per-bar cost components, each summed into the total (a return drag or a currency amount).
"""

import polars as pl

from pomata._expr import float64_expr, validate_positive
from pomata.pnl.accounting import turnover

__all__ = (
    "cost_borrow",
    "cost_fixed",
    "cost_funding",
    "cost_notional",
    "cost_per_share",
    "cost_proportional",
    "cost_slippage",
)


def cost_borrow(
    quantity: pl.Expr,
    price: pl.Expr,
    *,
    rate: float,
) -> pl.Expr:
    r"""
    Short-Borrow Cost, the per-bar fee for holding a short position.

    The per-bar carrying cost of a short: the short notional (the absolute short size times the price) times a per-bar
    borrow rate. Only short positions pay it; a long or flat position has zero borrow cost:

    .. math::

        c_t = \max(-q_t,\ 0) \cdot P_t \cdot \mathrm{rate}, \qquad q = \text{quantity}.

    It is a holding cost (charged on the position held, not on a trade), so it is elementwise — no turnover, no lag.
    This is a currency / cash-flow cost component; build it per bar and subtract it (with any others) from the gross PnL
    via :func:`pnl_net`.

    Args:
        quantity: Signed position size in units / shares / contracts held over the bar; only the short part (``q < 0``)
            is charged.
        price: Instrument price series (e.g. ``pl.col("close")``); must share a length and alignment with ``quantity``.
        rate: Per-bar borrow rate, as a fraction of the short notional (e.g. an annual rate divided by the bars per
            year). Must be a finite number ``>= 0``.

    Returns:
        The per-bar borrow cost for each row, the same length as the inputs -- a non-negative cost on short bars (for a
        non-negative price; a negative price yields an economically meaningless negative value) and ``0`` on long or
        flat bars.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``rate`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).

    Note:
        **Correctness:**
        The result is checked against an independent reference oracle on every input, and every edge case (missing data,
        boundaries, and warm-up where applicable) is given a defined behavior.

        **Long / flat:**

        A non-negative quantity has zero borrow cost (only the short part is charged).

        **Edge-case behavior:**

        - **Null** — a ``null`` quantity makes that row ``null`` (``null`` takes precedence over ``NaN``).
        - **NaN** — a ``NaN`` quantity yields ``NaN`` for that row.
        - **Non-finite input** — an ``inf`` quantity follows IEEE-754 through the arithmetic, where the short-only clip
          frees an infinite long (``0``) and an infinite short notional charges an ``inf`` fee (the sign, and any
          ``inf - inf = NaN``, included).
        - **Partitioning** — already correct on a multi-series panel: ``.over(...)`` partitions identically and is
          therefore optional here, unlike the turnover-based / cumulative functions.

    See Also:
        - :func:`dividend`: The equity holding cashflow on the income side.
        - :func:`cost_funding`: The perpetual-swap holding cost.
        - :func:`pnl_net`: Subtracts the composed cost from the gross PnL.

    References:
        - D'Avolio, G. (2002). "The market for borrowing stock." *Journal of Financial Economics*, 66(2-3), 271-306.
        - https://doi.org/10.1016/S0304-405X(02)00206-4
        - https://en.wikipedia.org/wiki/Securities_lending

    Examples:
        >>> import polars as pl
        >>> from pomata.pnl import cost_borrow
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "quantity": [100.0, -50.0, -50.0, -20.0, -20.0],
        ...         "price": [10.0, 11.0, 12.0, 13.0, 14.0],
        ...     }
        ... )
        >>> expr = cost_borrow(pl.col("quantity"), pl.col("price"), rate=0.0001).round(6)
        >>> frame.select(expr.alias("cost"))["cost"].to_list()
        [0.0, 0.055, 0.06, 0.026, 0.028]

        On a multi-ticker panel, partition with ``.over`` — for this elementwise holding cost it is optional (the
        result is identical without it) and shown here only for consistency:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "quantity": [100.0, -50.0, -50.0, -20.0, -20.0, 30.0],
        ...         "price": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        ...     }
        ... )
        >>> expr = cost_borrow(pl.col("quantity"), pl.col("price"), rate=0.0001).over("ticker").round(6)
        >>> frame.with_columns(expr.alias("c"))["c"].to_list()
        [0.0, 0.055, 0.06, 0.026, 0.028, 0.0]

        A ``null`` (which propagates) and a ``NaN`` make the missing-data handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "quantity": [-50.0, None, -50.0, float("nan"), -20.0],
        ...         "price": [10.0, 11.0, float("nan"), 12.0, 13.0],
        ...     }
        ... )
        >>> expr = cost_borrow(pl.col("quantity"), pl.col("price"), rate=0.0001).round(6)
        >>> frame.select(expr.alias("cost"))["cost"].to_list()
        [0.05, None, nan, nan, 0.026]
    """
    quantity = float64_expr(quantity)
    price = float64_expr(price)
    validate_positive(rate, "rate", allow_zero=True)
    # Charge only the short part of the position: clip the negated quantity at zero (longs / flat -> 0), times the price
    # and the per-bar rate. Elementwise; null / NaN propagate through the arithmetic.
    return ((-quantity).clip(lower_bound=0.0) * price * rate).name.keep()


def cost_fixed(
    quantity: pl.Expr,
    *,
    fee: float,
) -> pl.Expr:
    r"""
    Fixed Transaction Cost, a flat charge per trade.

    A flat fee in the account currency charged on every bar where the position changes (a trade occurs), and nothing on
    bars where it is held unchanged:

    .. math::

        c_t = \begin{cases} \mathrm{fee} & \text{if } \lvert \Delta q_t \rvert > 0 \\ 0 & \text{otherwise} \end{cases},
        \qquad q = \text{quantity}.

    A trade is detected from the :func:`turnover` of the quantity (with the pre-series quantity flat, so the entry trade
    counts). This is a currency / cash-flow cost component; build it per bar and subtract it (with any others) from the
    gross PnL via :func:`pnl_net`.

    Args:
        quantity: Signed position size in units / shares / contracts held over the bar (e.g. ``100``, ``-2``).
        fee: Flat charge per trade, in the account currency. Must be a finite number ``>= 0``.

    Returns:
        The per-bar fixed cost for each row, the same length as ``quantity`` -- ``fee`` where the quantity changes (the
        first row counts as a trade from a flat start) and ``0`` where it is held.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``fee`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).

    Note:
        **Correctness:**
        The result is checked against an independent reference oracle on every input, and every edge case (missing data,
        boundaries, and warm-up where applicable) is given a defined behavior.

        **Flat start:**

        The pre-series quantity is taken as ``0`` (via :func:`turnover`), so the first row charges the ``fee`` (entering
        the initial position is a trade).

        **Edge-case behavior:**

        - **Null** — a ``null`` quantity makes that row ``null`` (``null`` takes precedence over ``NaN``).
        - **NaN** — a ``NaN`` quantity yields ``NaN`` for that row.
        - **Non-finite input** — an ``inf`` quantity follows IEEE-754 through the arithmetic of the turnover
          difference, whose ``inf`` marks a trade and charges the flat ``fee`` (the sign, and any ``inf - inf = NaN``,
          included).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history, e.g. ``cost_fixed(pl.col("quantity"), fee=1.0).over("ticker")``.

    See Also:
        - :func:`cost_per_share`: A per-unit-traded commission.
        - :func:`cost_notional`: A proportional (bps-of-notional) commission.
        - :func:`pnl_net`: Subtracts the composed cost from the gross PnL.

    References:
        - https://en.wikipedia.org/wiki/Transaction_cost

    Examples:
        >>> import polars as pl
        >>> from pomata.pnl import cost_fixed
        >>>
        >>> frame = pl.DataFrame({"quantity": [10.0, 10.0, -5.0, -5.0, 20.0]})
        >>> frame.select(cost_fixed(pl.col("quantity"), fee=1.0).round(4).alias("cost"))["cost"].to_list()
        [1.0, 0.0, 1.0, 0.0, 1.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker starts flat:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "quantity": [10.0, 10.0, -5.0, 2.0, 2.0, 2.0],
        ...     }
        ... )
        >>> frame.with_columns(cost_fixed(pl.col("quantity"), fee=1.0).over("ticker").round(4).alias("c"))[
        ...     "c"
        ... ].to_list()
        [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]

        A ``null`` (which voids its own row and the next) and a ``NaN`` make the missing-data handling visible:

        >>> frame = pl.DataFrame({"quantity": [10.0, None, -5.0, float("nan"), 20.0]})
        >>> frame.select(cost_fixed(pl.col("quantity"), fee=1.0).round(4).alias("cost"))["cost"].to_list()
        [1.0, None, None, nan, nan]
    """
    quantity = float64_expr(quantity)
    validate_positive(fee, "fee", allow_zero=True)
    # Charge the flat fee where the quantity changes, 0 where it is held; mask null / NaN through (a bare ``when`` would
    # mis-handle them, since Polars orders NaN as greater than every number and treats a null condition as false).
    traded = turnover(quantity)
    return (
        pl.when(traded.is_null() | traded.is_nan()).then(traded).when(traded > 0.0).then(fee).otherwise(0.0)
    ).name.keep()


def cost_funding(
    quantity: pl.Expr,
    price: pl.Expr,
    funding_rate: pl.Expr,
) -> pl.Expr:
    r"""
    Funding Cost, the per-bar perpetual-swap funding payment seen as a cost to the holder.

    A perpetual swap has no expiry, so a periodic funding payment tethers it to the spot price: each funding bar the
    holder pays (or receives) the position notional times the funding rate. Taken as a cost to the holder, it is the
    signed product of the position, the price, and the per-bar funding rate:

    .. math::

        c_t = q_t \cdot P_t \cdot f_t, \qquad q = \text{quantity},\ f = \text{funding\_rate}.

    The rate is **signed and per-bar**: a positive rate debits a long (a positive cost) and credits a short (a negative
    cost — a rebate), and a negative rate flips both. It is a holding cost (charged on the position held, not on a
    trade), so it is elementwise — no turnover, no lag. This is a currency / cash-flow cost component; build it per bar
    and subtract it (with any others) from the gross PnL via :func:`pnl_net`.

    Args:
        quantity: Signed position size in units / shares / contracts held over the bar (e.g. ``100``, ``-2``).
        price: Instrument price series (e.g. ``pl.col("close")``); must share a length and alignment with ``quantity``.
        funding_rate: Per-bar funding rate as a signed fraction of notional, supplied as a series so it can be ``0``
            on the bars between funding events (e.g. ``0.0001`` = 1 bp); a positive rate charges longs and rebates
            shorts.

    Returns:
        The per-bar funding cost for each row, the same length as the inputs -- positive where the holder pays and
        negative (a rebate) where the holder receives.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness:**
        The result is checked against an independent reference oracle on every input, and every edge case (missing data,
        boundaries, and warm-up where applicable) is given a defined behavior.

        **Sign:**

        The cost follows ``sign(quantity) * sign(funding_rate)``: a long pays a positive rate and is rebated by a
        negative one; a short is the mirror image.

        **Off-funding bars:**

        Pass ``funding_rate = 0`` on bars with no funding event; the cost is then ``0`` there.

        **Edge-case behavior:**

        - **Null** — a ``null`` quantity makes that row ``null`` (``null`` takes precedence over ``NaN``).
        - **NaN** — a ``NaN`` quantity yields ``NaN`` for that row.
        - **Non-finite input** — an ``inf`` quantity follows IEEE-754 through the arithmetic, the signed triple product
          ``quantity * price * funding_rate`` (the sign, and any ``inf - inf = NaN``, included).
        - **Partitioning** — already correct on a multi-series panel: ``.over(...)`` partitions identically and is
          therefore optional here, unlike the turnover-based / cumulative functions.

    See Also:
        - :func:`cost_borrow`: The short-borrow holding cost on the equity side.
        - :func:`cost_notional`: The maker/taker fee on each perpetual-swap trade.
        - :func:`pnl_net`: Subtracts the composed cost from the gross PnL.

    References:
        - Shiller, R. J. (1993). "Measuring Asset Values for Cash Settlement in Derivative Markets: Hedonic Repeated
          Measures Indices and Perpetual Futures." *The Journal of Finance*, 48(3), 911-931.
        - https://doi.org/10.1111/j.1540-6261.1993.tb04024.x
        - https://en.wikipedia.org/wiki/Perpetual_futures

    Examples:
        >>> import polars as pl
        >>> from pomata.pnl import cost_funding
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "quantity": [10.0, 10.0, -5.0, -5.0, 20.0],
        ...         "price": [100.0, 102.0, 101.0, 104.0, 103.0],
        ...         "funding_rate": [0.0001, 0.0001, 0.0001, -0.0001, 0.0001],
        ...     }
        ... )
        >>> expr = cost_funding(pl.col("quantity"), pl.col("price"), pl.col("funding_rate")).round(6)
        >>> frame.select(expr.alias("cost"))["cost"].to_list()
        [0.1, 0.102, -0.0505, 0.052, 0.206]

        On a multi-ticker panel, partition with ``.over`` — for this elementwise holding cost it is optional (the
        result is identical without it) and shown here only for consistency:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "quantity": [10.0, 10.0, -5.0, 2.0, 2.0, -3.0],
        ...         "price": [100.0, 102.0, 101.0, 50.0, 51.0, 49.0],
        ...         "funding_rate": [0.0001, 0.0001, 0.0001, 0.0001, -0.0001, 0.0001],
        ...     }
        ... )
        >>> expr = cost_funding(pl.col("quantity"), pl.col("price"), pl.col("funding_rate")).over("ticker").round(6)
        >>> frame.with_columns(expr.alias("c"))["c"].to_list()
        [0.1, 0.102, -0.0505, 0.01, -0.0102, -0.0147]

        A ``null`` (which propagates) and a ``NaN`` make the missing-data handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "quantity": [10.0, None, -5.0, float("nan"), 20.0],
        ...         "price": [100.0, 102.0, 101.0, 104.0, 103.0],
        ...         "funding_rate": [0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
        ...     }
        ... )
        >>> expr = cost_funding(pl.col("quantity"), pl.col("price"), pl.col("funding_rate")).round(6)
        >>> frame.select(expr.alias("cost"))["cost"].to_list()
        [0.1, None, -0.0505, nan, 0.206]
    """
    quantity = float64_expr(quantity)
    price = float64_expr(price)
    funding_rate = float64_expr(funding_rate)
    # The funding payment on the held notional: position times price times the signed per-bar rate. Elementwise; null /
    # NaN propagate through the arithmetic. The rate carries its own sign, so there is no parameter to validate here.
    return (quantity * price * funding_rate).name.keep()


def cost_notional(
    quantity: pl.Expr,
    price: pl.Expr,
    *,
    rate: float,
) -> pl.Expr:
    r"""
    Notional Transaction Cost, a fee as a fraction of the traded notional.

    The traded notional (the units traded times the price) times a flat rate — the bps-of-notional commission charged on
    each trade, the maker/taker fee shape of crypto and FX venues:

    .. math::

        c_t = \lvert \Delta q_t \rvert \cdot P_t \cdot \mathrm{rate}, \qquad q = \text{quantity}.

    The units traded come from the :func:`turnover` of the quantity (with the pre-series quantity flat). This is a
    currency / cash-flow cost component; build it per bar and subtract it (with any others) from the gross PnL via
    :func:`pnl_net`.

    Args:
        quantity: Signed position size in units / shares / contracts held over the bar (e.g. ``100``, ``-2``).
        price: Instrument price series (e.g. ``pl.col("close")``); must share a length and alignment with ``quantity``.
        rate: Proportional cost rate, the fee as a fraction of traded notional (e.g. ``0.001`` = 10 bps). Must be a
            finite number ``>= 0``.

    Returns:
        The per-bar notional cost for each row, the same length as the inputs. The first row charges on
        ``|quantity_0| * price_0`` (the entry trade from a flat start).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``rate`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).

    Note:
        **Correctness:**
        The result is checked against an independent reference oracle on every input, and every edge case (missing data,
        boundaries, and warm-up where applicable) is given a defined behavior.

        **Flat start:**

        The pre-series quantity is taken as ``0`` (via :func:`turnover`), so the first row charges on the entry trade.

        **Edge-case behavior:**

        - **Null** — a ``null`` quantity makes that row ``null`` (``null`` takes precedence over ``NaN``).
        - **NaN** — a ``NaN`` quantity yields ``NaN`` for that row.
        - **Non-finite input** — an ``inf`` quantity follows IEEE-754 through the arithmetic of the turnover
          difference, an infinite move charging an ``inf`` cost (the sign, and any ``inf - inf = NaN``, included).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history, e.g. ``cost_notional(pl.col("quantity"), pl.col("price"), rate=0.001).over("ticker")``.

    See Also:
        - :func:`cost_per_share`: A per-unit-traded commission.
        - :func:`cost_fixed`: A flat charge per trade.
        - :func:`pnl_net`: Subtracts the composed cost from the gross PnL.

    References:
        - https://en.wikipedia.org/wiki/Transaction_cost

    Examples:
        >>> import polars as pl
        >>> from pomata.pnl import cost_notional
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "quantity": [10.0, 10.0, -5.0, -5.0, 20.0],
        ...         "price": [100.0, 102.0, 101.0, 104.0, 103.0],
        ...     }
        ... )
        >>> expr = cost_notional(pl.col("quantity"), pl.col("price"), rate=0.001).round(4)
        >>> frame.select(expr.alias("cost"))["cost"].to_list()
        [1.0, 0.0, 1.515, 0.0, 2.575]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker starts flat:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "quantity": [10.0, 10.0, -5.0, 2.0, 2.0, 2.0],
        ...         "price": [100.0, 102.0, 101.0, 50.0, 51.0, 49.0],
        ...     }
        ... )
        >>> frame.with_columns(
        ...     cost_notional(pl.col("quantity"), pl.col("price"), rate=0.001).over("ticker").round(4).alias("c")
        ... )["c"].to_list()
        [1.0, 0.0, 1.515, 0.1, 0.0, 0.0]

        A ``null`` (which voids the rows that reference it) and a ``NaN`` make the missing-data handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "quantity": [10.0, None, -5.0, float("nan"), 20.0],
        ...         "price": [100.0, 102.0, 101.0, 104.0, float("nan")],
        ...     }
        ... )
        >>> expr = cost_notional(pl.col("quantity"), pl.col("price"), rate=0.001).round(4)
        >>> frame.select(expr.alias("cost"))["cost"].to_list()
        [1.0, None, None, nan, nan]
    """
    quantity = float64_expr(quantity)
    price = float64_expr(price)
    validate_positive(rate, "rate", allow_zero=True)
    return (turnover(quantity) * price * rate).name.keep()


def cost_per_share(
    quantity: pl.Expr,
    *,
    fee: float,
) -> pl.Expr:
    r"""
    Per-Share Transaction Cost, a commission per unit traded.

    The number of units traded times a flat per-unit fee — the per-share commission of equity and futures brokers:

    .. math::

        c_t = \lvert \Delta q_t \rvert \cdot \mathrm{fee}, \qquad q = \text{quantity}.

    The units traded come from the :func:`turnover` of the quantity (with the pre-series quantity flat). This is a
    currency / cash-flow cost component; build it per bar and subtract it (with any others) from the gross PnL via
    :func:`pnl_net`.

    Args:
        quantity: Signed position size in units / shares / contracts held over the bar (e.g. ``100``, ``-2``).
        fee: Commission per unit traded, in the account currency (e.g. ``0.01`` = one cent per share). Must be a
            finite number ``>= 0``.

    Returns:
        The per-bar per-share cost for each row, the same length as ``quantity``. The first row charges on
        ``|quantity_0|`` (the entry trade from a flat start).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``fee`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).

    Note:
        **Correctness:**
        The result is checked against an independent reference oracle on every input, and every edge case (missing data,
        boundaries, and warm-up where applicable) is given a defined behavior.

        **Flat start:**

        The pre-series quantity is taken as ``0`` (via :func:`turnover`), so the first row charges on ``|quantity_0|``
        (entering the initial position is a trade).

        **Edge-case behavior:**

        - **Null** — a ``null`` quantity makes that row ``null`` (``null`` takes precedence over ``NaN``).
        - **NaN** — a ``NaN`` quantity yields ``NaN`` for that row.
        - **Non-finite input** — an ``inf`` quantity follows IEEE-754 through the arithmetic of the turnover
          difference, an infinite move charging an ``inf`` cost (the sign, and any ``inf - inf = NaN``, included).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history, e.g. ``cost_per_share(pl.col("quantity"), fee=0.01).over("ticker")``.

    See Also:
        - :func:`cost_fixed`: A flat charge per trade.
        - :func:`cost_notional`: A proportional (bps-of-notional) commission.
        - :func:`pnl_net`: Subtracts the composed cost from the gross PnL.

    References:
        - https://en.wikipedia.org/wiki/Transaction_cost

    Examples:
        >>> import polars as pl
        >>> from pomata.pnl import cost_per_share
        >>>
        >>> frame = pl.DataFrame({"quantity": [10.0, 10.0, -5.0, -5.0, 20.0]})
        >>> frame.select(cost_per_share(pl.col("quantity"), fee=0.01).round(4).alias("cost"))["cost"].to_list()
        [0.1, 0.0, 0.15, 0.0, 0.25]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker starts flat:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "quantity": [10.0, 10.0, -5.0, 2.0, 2.0, 2.0],
        ...     }
        ... )
        >>> frame.with_columns(cost_per_share(pl.col("quantity"), fee=0.01).over("ticker").round(4).alias("c"))[
        ...     "c"
        ... ].to_list()
        [0.1, 0.0, 0.15, 0.02, 0.0, 0.0]

        A ``null`` (which voids its own row and the next) and a ``NaN`` make the missing-data handling visible:

        >>> frame = pl.DataFrame({"quantity": [10.0, None, -5.0, float("nan"), 20.0]})
        >>> frame.select(cost_per_share(pl.col("quantity"), fee=0.01).round(4).alias("cost"))["cost"].to_list()
        [0.1, None, None, nan, nan]
    """
    quantity = float64_expr(quantity)
    validate_positive(fee, "fee", allow_zero=True)
    return (turnover(quantity) * fee).name.keep()


def cost_proportional(
    weight: pl.Expr,
    *,
    rate: float,
) -> pl.Expr:
    r"""
    Proportional Transaction Cost, a fee charged as a fraction of the traded notional.

    The per-bar broker commission: the fraction of capital traded that bar (the :func:`turnover`) times a flat rate, the
    classic bps-of-notional fee:

    .. math::

        c_t = \mathrm{turnover}_t \cdot \mathrm{rate}.

    It is one orthogonal cost component: build it per bar and subtract it (with any others) from the gross return via
    :func:`returns_net`. A fixed bid-ask half-spread has the same shape against turnover; that distinct cost axis is
    :func:`cost_slippage`.

    Args:
        weight: Signed weight, the fraction of capital held (e.g. ``1.0`` fully long, ``-0.5`` half short);
            ``|weight| > 1`` is leverage.
        rate: Proportional cost rate, the fee as a fraction of traded notional (e.g. ``0.001`` = 10 bps). Must be a
            finite number ``>= 0``.

    Returns:
        The per-bar proportional cost for each row, the same length as ``weight``. The first row is
        ``|weight_0| * rate`` (the cost of the entry trade from a flat start, per :func:`turnover`).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``rate`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).

    Note:
        **Correctness:**
        The result is checked against an independent reference oracle on every input, and every edge case (missing data,
        boundaries, and warm-up where applicable) is given a defined behavior.

        **Flat start:**

        The weight before the series is taken as ``0`` (via :func:`turnover`), so the first row is ``|weight_0| *
        rate``: establishing the initial weight carries its cost.

        **Edge-case behavior:**

        - **Null** — a ``null`` weight makes that row ``null`` (``null`` takes precedence over ``NaN``).
        - **NaN** — a ``NaN`` weight yields ``NaN`` for that row.
        - **Non-finite input** — an ``inf`` weight follows IEEE-754 through the arithmetic of the turnover difference,
          an infinite move charging an ``inf`` cost (the sign, and any ``inf - inf = NaN``, included).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history, e.g. ``cost_proportional(pl.col("weight"), rate=0.001).over("ticker")``.

    See Also:
        - :func:`cost_slippage`: The fixed half-spread cost, the complementary per-trade leg; sum the two for both.
        - :func:`turnover`: The traded fraction this scales.
        - :func:`returns_net`: Subtracts the composed cost from the gross return.

    References:
        - Magill, M. J. P. & Constantinides, G. M. (1976). "Portfolio selection with transactions costs." *Journal of
          Economic Theory*, 13(2), 245-263.
        - https://doi.org/10.1016/0022-0531(76)90018-1
        - https://en.wikipedia.org/wiki/Transaction_cost

    Examples:
        >>> import polars as pl
        >>> from pomata.pnl import cost_proportional
        >>>
        >>> frame = pl.DataFrame({"weight": [0.5, 1.0, -0.5, -0.5, 0.0]})
        >>> expr = cost_proportional(pl.col("weight"), rate=0.001).round(4)
        >>> frame.select(expr.alias("cost"))["cost"].to_list()
        [0.0005, 0.0005, 0.0015, 0.0, 0.0005]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker starts flat and never reaches across the
        boundary:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "weight": [0.5, 1.0, -0.5, 1.0, 1.0, 0.0],
        ...     }
        ... )
        >>> frame.with_columns(cost_proportional(pl.col("weight"), rate=0.001).over("ticker").round(4).alias("c"))[
        ...     "c"
        ... ].to_list()
        [0.0005, 0.0005, 0.0015, 0.001, 0.0, 0.001]

        A ``null`` (which voids its own row and the next) and a ``NaN`` make the missing-data handling visible:

        >>> frame = pl.DataFrame({"weight": [0.5, None, -0.5, float("nan"), 0.0]})
        >>> frame.select(cost_proportional(pl.col("weight"), rate=0.001).round(4).alias("cost"))["cost"].to_list()
        [0.0005, None, None, nan, nan]
    """
    weight = float64_expr(weight)
    validate_positive(rate, "rate", allow_zero=True)
    return (turnover(weight) * rate).name.keep()


def cost_slippage(
    weight: pl.Expr,
    *,
    half_spread: float,
) -> pl.Expr:
    r"""
    Slippage Cost, the fixed bid-ask half-spread crossed on each trade.

    The per-bar market cost of crossing the spread: the fraction of capital traded that bar (the :func:`turnover`) times
    a fixed half-spread, the cost paid per side relative to the mid price:

    .. math::

        c_t = \mathrm{turnover}_t \cdot \mathrm{half\_spread}.

    It is one orthogonal cost component, distinct from the broker fee (:func:`cost_proportional`): build it per bar and
    subtract it (with any others) from the gross return via :func:`returns_net`. ``half_spread`` is the per-side cost,
    half the full bid-ask spread, taken directly (no hidden division).

    Args:
        weight: Signed weight, the fraction of capital held (e.g. ``1.0`` fully long, ``-0.5`` half short);
            ``|weight| > 1`` is leverage.
        half_spread: Fixed bid-ask half-spread crossed per trade, as a fraction (half the full spread; e.g. ``0.002``).
            Must be a finite number ``>= 0``.

    Returns:
        The per-bar slippage cost for each row, the same length as ``weight``. The first row is
        ``|weight_0| * half_spread`` (the cost of the entry trade from a flat start, per :func:`turnover`).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``half_spread`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``).

    Note:
        **Correctness:**
        The result is checked against an independent reference oracle on every input, and every edge case (missing data,
        boundaries, and warm-up where applicable) is given a defined behavior.

        **Flat start:**

        The weight before the series is taken as ``0`` (via :func:`turnover`), so the first row is ``|weight_0| *
        half_spread``: establishing the initial weight crosses the spread.

        **Edge-case behavior:**

        - **Null** — a ``null`` weight makes that row ``null`` (``null`` takes precedence over ``NaN``).
        - **NaN** — a ``NaN`` weight yields ``NaN`` for that row.
        - **Non-finite input** — an ``inf`` weight follows IEEE-754 through the arithmetic of the turnover difference,
          an infinite move charging an ``inf`` cost (the sign, and any ``inf - inf = NaN``, included).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history, e.g. ``cost_slippage(pl.col("weight"), half_spread=0.002).over("ticker")``.

    See Also:
        - :func:`cost_proportional`: The proportional broker fee, the complementary per-trade leg; sum the two for both.
        - :func:`turnover`: The traded fraction this scales.
        - :func:`returns_net`: Subtracts the composed cost from the gross return.

    References:
        - Demsetz, H. (1968). "The Cost of Transacting." *The Quarterly Journal of Economics*, 82(1), 33-53.
        - https://doi.org/10.2307/1882244
        - https://en.wikipedia.org/wiki/Slippage_%28finance%29

    Examples:
        >>> import polars as pl
        >>> from pomata.pnl import cost_slippage
        >>>
        >>> frame = pl.DataFrame({"weight": [0.5, 1.0, -0.5, -0.5, 0.0]})
        >>> expr = cost_slippage(pl.col("weight"), half_spread=0.002).round(4)
        >>> frame.select(expr.alias("cost"))["cost"].to_list()
        [0.001, 0.001, 0.003, 0.0, 0.001]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker starts flat and never reaches across the
        boundary:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "weight": [0.5, 1.0, -0.5, 1.0, 1.0, 0.0],
        ...     }
        ... )
        >>> frame.with_columns(cost_slippage(pl.col("weight"), half_spread=0.002).over("ticker").round(4).alias("c"))[
        ...     "c"
        ... ].to_list()
        [0.001, 0.001, 0.003, 0.002, 0.0, 0.002]

        A ``null`` (which voids its own row and the next) and a ``NaN`` make the missing-data handling visible:

        >>> frame = pl.DataFrame({"weight": [0.5, None, -0.5, float("nan"), 0.0]})
        >>> frame.select(cost_slippage(pl.col("weight"), half_spread=0.002).round(4).alias("cost"))["cost"].to_list()
        [0.001, None, None, nan, nan]
    """
    weight = float64_expr(weight)
    validate_positive(half_spread, "half_spread", allow_zero=True)
    return (turnover(weight) * half_spread).name.keep()
