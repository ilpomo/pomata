"""
Price-transform indicators (representative prices).
"""

import polars as pl

from pomata._expr import float64_expr

__all__ = ("price_average", "price_median", "price_typical", "price_weighted_close")


def price_average(
    open: pl.Expr,
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
) -> pl.Expr:
    r"""
    Average Price, the equal-weighted mean of the four OHLC prices.

    The simplest representative price for a bar: the arithmetic mean of its ``open``, ``high``, ``low``, and ``close``,
    weighting each equally.

    .. math::

        \mathrm{AVGPRICE}_t = \frac{\mathrm{open}_t + \mathrm{high}_t + \mathrm{low}_t + \mathrm{close}_t}{4}.

    It is a pure per-bar transform — no window, no recursion, no cross-bar state — so it is defined from row ``0`` and
    each row depends only on its own four prices.

    Args:
        open: Open-price series (e.g. ``pl.col("open")``).
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).

    Returns:
        The average price for each row, the same length as the inputs. There is no window and no warm-up -- every row is
        defined from row ``0``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        **Inputs:**

        ``open``, ``high``, ``low``, and ``close`` are taken as the canonical OHLC roles in that positional order and
        must share a length and alignment (the same row index is one bar).

        **Edge-case behavior:**

        - **Null** — the mean is a plain sum over the four inputs, so a ``null`` in any of them propagates: the row is
          ``null`` whenever at least one of its four prices is ``null`` (``null`` takes precedence over ``NaN``).
        - **NaN** — a ``NaN`` in any input (with no ``null`` at that row) propagates, yielding ``NaN`` for that row.
        - **Partitioning** — the transform is elementwise (each row uses only its own bar), so it is already
          correct on a multi-series panel: ``.over(...)`` partitions identically and is therefore optional here
          (the result is the same either way), unlike the windowed indicators where ``.over`` is required to stop
          a window spanning series boundaries.

    See Also:
        - :func:`price_median`: The midpoint of the bar's range, ``(high + low) / 2``.
        - :func:`price_typical`: The equal-weighted mean of high, low, and close.
        - :func:`price_weighted_close`: The OHLC summary that double-weights the close.

    References:
        - No canonical external source; the indicator is defined by the formula above.

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import price_average
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "open": [10.0, 11.0, 12.0, 11.5, 13.0],
        ...         "high": [11.0, 12.0, 13.0, 12.5, 14.0],
        ...         "low": [9.0, 10.0, 11.0, 11.0, 12.0],
        ...         "close": [10.0, 11.5, 12.5, 11.5, 13.5],
        ...     }
        ... )
        >>> expr = price_average(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")).round(4)
        >>> frame.select(expr.alias("price_average"))["price_average"].to_list()
        [10.0, 11.125, 12.125, 11.625, 13.125]

        On a multi-ticker panel, partition with ``.over`` as the windowed indicators require — for this elementwise
        transform ``.over`` is optional (the result is identical without it) and shown here only for consistency:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "open": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
        ...         "high": [11.0, 12.0, 13.0, 21.0, 22.0, 23.0],
        ...         "low": [9.0, 10.0, 11.0, 19.0, 20.0, 21.0],
        ...         "close": [10.0, 11.5, 12.5, 20.0, 21.5, 22.5],
        ...     }
        ... )
        >>> expr = price_average(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("price_average"))["price_average"].to_list()
        [10.0, 11.125, 12.125, 20.0, 21.125, 22.125]

        A ``null`` then a ``NaN`` in ``close`` (both propagate through the sum) make the missing-data handling visible
        at a glance:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "open": [10.0, 11.0, 12.0, 13.0, 14.0],
        ...         "high": [11.0, 12.0, 13.0, 14.0, 15.0],
        ...         "low": [9.0, 10.0, 11.0, 12.0, 13.0],
        ...         "close": [10.0, None, 12.5, float("nan"), 14.5],
        ...     }
        ... )
        >>> expr = price_average(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")).round(4)
        >>> frame.select(expr.alias("price_average"))["price_average"].to_list()
        [10.0, None, 12.125, nan, 14.125]
    """
    open = float64_expr(open)
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    # Pure elementwise arithmetic: null propagates through `+` (it is NOT skipped as in max_horizontal), NaN propagates.
    return ((open + high + low + close) / 4.0).name.keep()


def price_median(
    high: pl.Expr,
    low: pl.Expr,
) -> pl.Expr:
    r"""
    Median Price, the midpoint of the bar's range.

    The center of the high-low range — the mean of just the two extremes, ignoring open and close:

    .. math::

        \mathrm{MEDPRICE}_t = \frac{\mathrm{high}_t + \mathrm{low}_t}{2}.

    It is a pure per-bar transform — no window, no recursion, no cross-bar state — so it is defined from row ``0`` and
    each row depends only on its own ``high`` and ``low``.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).

    Returns:
        The median price for each row, the same length as the inputs. There is no window and no warm-up -- every row is
        defined from row ``0``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        **Inputs:**

        ``high`` and ``low`` are taken as the canonical OHLC roles in that positional order and must share a length and
        alignment (the same row index is one bar).

        **Edge-case behavior:**

        - **Null** — the midpoint is a plain sum over the two inputs, so a ``null`` in either propagates: the row is
          ``null`` whenever ``high`` or ``low`` is ``null`` (``null`` takes precedence over ``NaN``).
        - **NaN** — a ``NaN`` in either input (with no ``null`` at that row) propagates, yielding ``NaN`` for that row.
        - **Partitioning** — the transform is elementwise (each row uses only its own bar), so it is already
          correct on a multi-series panel: ``.over(...)`` partitions identically and is therefore optional here
          (the result is the same either way), unlike the windowed indicators where ``.over`` is required to stop
          a window spanning series boundaries.

    See Also:
        - :func:`midprice`: The rolling midpoint of the high-low range over a window.
        - :func:`price_average`: The equal-weighted mean of the four OHLC prices.
        - :func:`price_typical`: The equal-weighted mean of high, low, and close.

    References:
        - Achelis, S. B. (2000). *Technical Analysis from A to Z* (2nd ed.). McGraw-Hill.
        - https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/median-price

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import price_median
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [11.0, 12.0, 13.0, 12.5, 14.0],
        ...         "low": [9.0, 10.0, 11.0, 11.0, 12.0],
        ...     }
        ... )
        >>> expr = price_median(pl.col("high"), pl.col("low")).round(4)
        >>> frame.select(expr.alias("price_median"))["price_median"].to_list()
        [10.0, 11.0, 12.0, 11.75, 13.0]

        On a multi-ticker panel, partition with ``.over`` as the windowed indicators require — for this elementwise
        transform ``.over`` is optional (the result is identical without it) and shown here only for consistency:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "high": [11.0, 12.0, 13.0, 21.0, 22.0, 23.0],
        ...         "low": [9.0, 10.0, 11.0, 19.0, 20.0, 21.0],
        ...     }
        ... )
        >>> expr = price_median(pl.col("high"), pl.col("low")).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("price_median"))["price_median"].to_list()
        [10.0, 11.0, 12.0, 20.0, 21.0, 22.0]

        A ``null`` then a ``NaN`` in ``high`` (both propagate through the sum) make the missing-data handling visible
        at a glance:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [11.0, None, 13.0, float("nan"), 15.0],
        ...         "low": [9.0, 10.0, 11.0, 12.0, 13.0],
        ...     }
        ... )
        >>> expr = price_median(pl.col("high"), pl.col("low")).round(4)
        >>> frame.select(expr.alias("price_median"))["price_median"].to_list()
        [10.0, None, 12.0, nan, 14.0]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    # Pure elementwise arithmetic: null propagates through `+` (it is NOT skipped as in max_horizontal), NaN propagates.
    return ((high + low) / 2.0).name.keep()


def price_typical(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
) -> pl.Expr:
    r"""
    Typical Price, the equal-weighted mean of high, low, and close.

    A common single-number summary of a bar — the average of its ``high``, ``low``, and ``close``. It is the price
    series the Commodity Channel Index (:func:`cci`) is built on:

    .. math::

        \mathrm{TYPPRICE}_t = \frac{\mathrm{high}_t + \mathrm{low}_t + \mathrm{close}_t}{3}.

    It is a pure per-bar transform — no window, no recursion, no cross-bar state — so it is defined from row ``0`` and
    each row depends only on its own ``high``, ``low``, and ``close``.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).

    Returns:
        The typical price for each row, the same length as the inputs. There is no window and no warm-up -- every row is
        defined from row ``0``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        **Inputs:**

        ``high``, ``low``, and ``close`` are taken as the canonical OHLC roles in that positional order and must share a
        length and alignment (the same row index is one bar).

        **Edge-case behavior:**

        - **Null** — the mean is a plain sum over the three inputs, so a ``null`` in any of them propagates: the row is
          ``null`` whenever at least one of ``high`` / ``low`` / ``close`` is ``null`` (``null`` takes precedence over
          ``NaN``).
        - **NaN** — a ``NaN`` in any input (with no ``null`` at that row) propagates, yielding ``NaN`` for that row.
        - **Partitioning** — the transform is elementwise (each row uses only its own bar), so it is already
          correct on a multi-series panel: ``.over(...)`` partitions identically and is therefore optional here
          (the result is the same either way), unlike the windowed indicators where ``.over`` is required to stop
          a window spanning series boundaries.

    See Also:
        - :func:`cci`: The Commodity Channel Index, built on the typical price.
        - :func:`price_average`: The equal-weighted mean of the four OHLC prices.
        - :func:`price_weighted_close`: The OHLC summary that double-weights the close.

    References:
        - Achelis, S. B. (2000). *Technical Analysis from A to Z* (2nd ed.). McGraw-Hill.
        - https://en.wikipedia.org/wiki/Typical_price

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import price_typical
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [11.0, 12.0, 13.0, 12.5, 14.0],
        ...         "low": [9.0, 10.0, 11.0, 11.0, 12.0],
        ...         "close": [10.0, 11.5, 12.5, 11.5, 13.5],
        ...     }
        ... )
        >>> expr = price_typical(pl.col("high"), pl.col("low"), pl.col("close")).round(4)
        >>> frame.select(expr.alias("price_typical"))["price_typical"].to_list()
        [10.0, 11.1667, 12.1667, 11.6667, 13.1667]

        On a multi-ticker panel, partition with ``.over`` as the windowed indicators require — for this elementwise
        transform ``.over`` is optional (the result is identical without it) and shown here only for consistency:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "high": [11.0, 12.0, 13.0, 21.0, 22.0, 23.0],
        ...         "low": [9.0, 10.0, 11.0, 19.0, 20.0, 21.0],
        ...         "close": [10.0, 11.5, 12.5, 20.0, 21.5, 22.5],
        ...     }
        ... )
        >>> expr = price_typical(pl.col("high"), pl.col("low"), pl.col("close")).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("price_typical"))["price_typical"].to_list()
        [10.0, 11.1667, 12.1667, 20.0, 21.1667, 22.1667]

        A ``null`` then a ``NaN`` in ``close`` (both propagate through the sum) make the missing-data handling visible
        at a glance:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [11.0, 12.0, 13.0, 14.0, 15.0],
        ...         "low": [9.0, 10.0, 11.0, 12.0, 13.0],
        ...         "close": [10.0, None, 12.5, float("nan"), 14.5],
        ...     }
        ... )
        >>> expr = price_typical(pl.col("high"), pl.col("low"), pl.col("close")).round(4)
        >>> frame.select(expr.alias("price_typical"))["price_typical"].to_list()
        [10.0, None, 12.1667, nan, 14.1667]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    # Pure elementwise arithmetic: null propagates through `+` (it is NOT skipped as in max_horizontal), NaN propagates.
    return ((high + low + close) / 3.0).name.keep()


def price_weighted_close(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
) -> pl.Expr:
    r"""
    Weighted Close Price, the OHLC summary that double-weights the close.

    A representative price that gives the ``close`` twice the weight of the ``high`` and the ``low``, emphasizing where
    the bar settled:

    .. math::

        \mathrm{WCLPRICE}_t = \frac{\mathrm{high}_t + \mathrm{low}_t + 2\,\mathrm{close}_t}{4}.

    It is a pure per-bar transform — no window, no recursion, no cross-bar state — so it is defined from row ``0`` and
    each row depends only on its own ``high``, ``low``, and ``close``.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``); weighted twice.

    Returns:
        The weighted close price for each row, the same length as the inputs. There is no window and no warm-up -- every
        row is defined from row ``0``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        **Inputs:**

        ``high``, ``low``, and ``close`` are taken as the canonical OHLC roles in that positional order and must share a
        length and alignment (the same row index is one bar).

        **Edge-case behavior:**

        - **Null** — the result is a plain weighted sum over the three inputs, so a ``null`` in any of them propagates:
          the row is ``null`` whenever at least one of ``high`` / ``low`` / ``close`` is ``null`` (``null`` takes
          precedence over ``NaN``).
        - **NaN** — a ``NaN`` in any input (with no ``null`` at that row) propagates, yielding ``NaN`` for that row.
        - **Partitioning** — the transform is elementwise (each row uses only its own bar), so it is already
          correct on a multi-series panel: ``.over(...)`` partitions identically and is therefore optional here
          (the result is the same either way), unlike the windowed indicators where ``.over`` is required to stop
          a window spanning series boundaries.

    See Also:
        - :func:`price_average`: The equal-weighted mean of the four OHLC prices.
        - :func:`price_median`: The midpoint of the bar's range, ``(high + low) / 2``.
        - :func:`price_typical`: The equal-weighted mean of high, low, and close.

    References:
        - Achelis, S. B. (2000). *Technical Analysis from A to Z* (2nd ed.). McGraw-Hill.
        - https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/weighted-close

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import price_weighted_close
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [11.0, 12.0, 13.0, 12.5, 14.0],
        ...         "low": [9.0, 10.0, 11.0, 11.0, 12.0],
        ...         "close": [10.0, 11.5, 12.5, 11.5, 13.5],
        ...     }
        ... )
        >>> expr = price_weighted_close(pl.col("high"), pl.col("low"), pl.col("close")).round(4)
        >>> frame.select(expr.alias("price_weighted_close"))["price_weighted_close"].to_list()
        [10.0, 11.25, 12.25, 11.625, 13.25]

        On a multi-ticker panel, partition with ``.over`` as the windowed indicators require — for this elementwise
        transform ``.over`` is optional (the result is identical without it) and shown here only for consistency:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "high": [11.0, 12.0, 13.0, 21.0, 22.0, 23.0],
        ...         "low": [9.0, 10.0, 11.0, 19.0, 20.0, 21.0],
        ...         "close": [10.0, 11.5, 12.5, 20.0, 21.5, 22.5],
        ...     }
        ... )
        >>> expr = price_weighted_close(pl.col("high"), pl.col("low"), pl.col("close")).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("price_weighted_close"))["price_weighted_close"].to_list()
        [10.0, 11.25, 12.25, 20.0, 21.25, 22.25]

        A ``null`` then a ``NaN`` in ``close`` (both propagate through the sum) make the missing-data handling visible
        at a glance:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [11.0, 12.0, 13.0, 14.0, 15.0],
        ...         "low": [9.0, 10.0, 11.0, 12.0, 13.0],
        ...         "close": [10.0, None, 12.5, float("nan"), 14.5],
        ...     }
        ... )
        >>> expr = price_weighted_close(pl.col("high"), pl.col("low"), pl.col("close")).round(4)
        >>> frame.select(expr.alias("price_weighted_close"))["price_weighted_close"].to_list()
        [10.0, None, 12.25, nan, 14.25]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    # Pure elementwise arithmetic: null propagates through `+` (it is NOT skipped as in max_horizontal), NaN propagates.
    return ((high + low + 2.0 * close) / 4.0).name.keep()
