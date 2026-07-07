"""
Directional movement indicators — the Wilder DMI / ADX system.
"""

import polars as pl

from pomata._expr import float64_expr, validate_window
from pomata.indicators.moving_average import rma
from pomata.indicators.volatility import atr, true_range

__all__ = ("adx", "adxr", "di_minus", "di_plus", "dm_minus", "dm_plus", "dx", "vortex")


def adx(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Average Directional Index (ADX).

    The Wilder-smoothed :func:`dx`: a single, non-directional measure of *trend strength* (it says how strongly price is
    trending, not in which direction), bounded in ``[0, 100]`` — low values mean a range, high values a strong trend.
    It is the :func:`rma` of the directional index:

    .. math::

        \mathrm{ADX}_t = \mathrm{RMA}(\mathrm{DX}, n)_t, \qquad n = \text{window}.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        window: Number of observations in the Wilder moving window. Must be ``>= 1``.

    Returns:
        The ADX for each row, the same length as the inputs, in ``[0, 100]``. It carries a deep warm-up — roughly
        ``2 * (window - 1)`` rows of ``null`` — since it smooths the already-smoothed :func:`dx`.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        It is scale-invariant under a positive common rescaling of ``high``, ``low``, and ``close`` (it is built from
        ratios of directional movement to the average true range).

        **Seeding:**

        The warm-up inherits the recursive Wilder seeding of :func:`rma` used throughout the cluster.

        **Edge-case behavior:**

        - **Null** — a ``null`` reaching the recursion yields ``null`` at that row.
        - **NaN** — a ``NaN`` poisons the recursion and yields ``NaN`` for every subsequent non-null row.
        - **Flat directional movement** — when ``di+`` and ``di-`` are both zero the underlying :func:`dx` is ``NaN``
          (``0 / 0``), which then poisons the ADX recursion.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the recursions never span
          series boundaries, e.g. ``adx(pl.col("high"), pl.col("low"), pl.col("close"), 14).over("ticker")``.

    See Also:
        - :func:`dx`: The directional index this smooths.
        - :func:`adxr`: The ADX rating (this averaged with its own past).
        - :func:`di_plus`: The plus directional indicator.

    References:
        - Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
        - https://en.wikipedia.org/wiki/Average_directional_movement_index

    Examples:
        On a small OHLC frame with a short window:

        >>> import polars as pl
        >>> from pomata.indicators import adx
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5],
        ...         "close": [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0],
        ...     }
        ... )
        >>> expr = adx(pl.col("high"), pl.col("low"), pl.col("close"), 2).round(4)
        >>> frame.select(expr.alias("adx"))["adx"].to_list()
        [None, None, 100.0, 60.0, 68.2353, 44.1176, 58.3602, 39.1801, 55.4486, 37.7243]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 6 + ["B"] * 6,
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 20.0, 22.0, 19.0, 23.0, 20.0, 24.0],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 18.0, 20.0, 17.0, 21.0, 18.0, 22.0],
        ...         "close": [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 19.0, 21.0, 18.0, 22.0, 19.0, 23.0],
        ...     }
        ... )
        >>> expr = adx(pl.col("high"), pl.col("low"), pl.col("close"), 2).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("adx"))["adx"].to_list()
        [None, None, 100.0, 60.0, 68.2353, 44.1176, None, None, 75.0, 62.5, 43.75, 45.0893]

        A leading ``null`` ``close`` (absorbed by the true-range maximum) and a later ``NaN`` (which poisons the
        recursion and latches) make the handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5],
        ...         "close": [None, 10.5, 11.5, 11.0, float("nan"), 12.0, 13.5, 13.0],
        ...     }
        ... )
        >>> expr = adx(pl.col("high"), pl.col("low"), pl.col("close"), 2).round(4)
        >>> frame.select(expr.alias("adx"))["adx"].to_list()
        [None, None, 100.0, 60.0, 68.2353, nan, nan, nan]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    validate_window(window)
    return rma(dx(high, low, close, window), window).name.keep()


def adxr(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Average Directional Index Rating (ADXR).

    Wilder's smoothing of the trend-strength reading: the mean of the current :func:`adx` and the ADX from ``window``
    bars ago, which damps the ADX and is often used to compare trend strength across time:

    .. math::

        \mathrm{ADXR}_t = \frac{\mathrm{ADX}_t + \mathrm{ADX}_{t - n}}{2}, \qquad n = \text{window}.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        window: Number of observations in the Wilder moving window, and the look-back for the averaging. Must be
            ``>= 1``.

    Returns:
        The ADXR for each row, the same length as the inputs, in ``[0, 100]``. Its warm-up is the :func:`adx` warm-up
        plus a further ``window`` rows (the look-back of the averaging).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        It is scale-invariant under a positive common rescaling of ``high``, ``low``, and ``close``.

        **Seeding:**

        The warm-up inherits the recursive Wilder seeding of :func:`rma` used throughout the cluster.

        **Edge-case behavior:**

        - **Null / NaN** — inherited from :func:`adx`: a ``null`` yields ``null`` and a ``NaN`` propagates; a row whose
          ADX or whose ``window``-ago ADX is missing is itself missing.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so neither the recursion nor the
          look-back spans series boundaries, e.g. by wrapping the whole call in ``.over("ticker")``.

    See Also:
        - :func:`adx`: The trend-strength index this averages with its own past.
        - :func:`dx`: The directional index the ADX smooths.
        - :func:`di_plus`: A directional indicator at the base of the system.

    References:
        - Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
        - https://en.wikipedia.org/wiki/Average_directional_movement_index

    Examples:
        On a small OHLC frame with a short window:

        >>> import polars as pl
        >>> from pomata.indicators import adxr
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5],
        ...         "close": [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0],
        ...     }
        ... )
        >>> expr = adxr(pl.col("high"), pl.col("low"), pl.col("close"), 2).round(4)
        >>> frame.select(expr.alias("adxr"))["adxr"].to_list()
        [None, None, None, None, 84.1176, 52.0588, 63.2977, 41.6489, 56.9044, 38.4522]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 20.0, 22.0, 19.0, 23.0, 20.0, 24.0, 21.0],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 18.0, 20.0, 17.0, 21.0, 18.0, 22.0, 19.0],
        ...         "close": [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 19.0, 21.0, 18.0, 22.0, 19.0, 23.0, 20.0],
        ...     }
        ... )
        >>> expr = adxr(pl.col("high"), pl.col("low"), pl.col("close"), 2).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("adxr"))["adxr"].to_list()
        [None, None, None, None, 84.1176, 52.0588, 63.2977, None, None, None, None, 59.375, 53.7946, 38.4358]

        A leading ``null`` ``close`` (absorbed by the true-range maximum) and a later ``NaN`` (which propagates)
        make the handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5],
        ...         "close": [None, 10.5, 11.5, 11.0, float("nan"), 12.0, 13.5, 13.0],
        ...     }
        ... )
        >>> expr = adxr(pl.col("high"), pl.col("low"), pl.col("close"), 2).round(4)
        >>> frame.select(expr.alias("adxr"))["adxr"].to_list()
        [None, None, None, None, 84.1176, nan, nan, nan]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    validate_window(window)
    average = adx(high, low, close, window)
    return ((average + average.shift(window)) / 2.0).name.keep()


def di_minus(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Minus Directional Indicator (-DI).

    The Wilder-smoothed minus directional movement (:func:`dm_minus`) as a percentage of the average true range
    (:func:`atr`), so downward trend pressure is comparable across instruments and bounded in ``[0, 100]``:

    .. math::

        -\mathrm{DI}_t = 100 \cdot \frac{\mathrm{dm\_minus}_t}{\mathrm{ATR}_t}.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        window: Number of observations in the Wilder moving window. Must be ``>= 1``.

    Returns:
        The minus directional indicator for each row, the same length as the inputs, in ``[0, 100]`` on complete bars.
        The first ``window - 1`` values are ``null`` (warm-up).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        It is scale-invariant under a positive common rescaling of ``high``, ``low``, and ``close`` (the smoothed
        movement and the average true range scale together).

        **Seeding:**

        The warm-up inherits the recursive Wilder seeding of :func:`rma` used throughout the cluster.

        **Edge-case behavior:**

        - **Flat series** — the ATR is an infinite-memory Wilder RMA, so ``0 / 0`` = ``NaN`` needs the whole series so
          far to be flat (both the ATR and the smoothed movement zero); a merely local flat patch after earlier
          movement leaves the ATR small-but-positive and the DI finite. The ``[0, 100]`` bound holds on complete
          coherent bars; a ``null`` prior close drops the close-based true-range terms and shrinks the ATR, so on a gap
          the ratio can exceed ``100``.
        - **Null** — a ``null`` in the smoothed movement or the ATR at a row yields ``null`` there.
        - **NaN** — a ``NaN`` propagates, yielding ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the recursions never span
          series boundaries, e.g. ``di_minus(pl.col("high"), pl.col("low"), pl.col("close"), 14).over("ticker")``.

    See Also:
        - :func:`di_plus`: The plus counterpart.
        - :func:`dm_minus`: The smoothed minus directional movement in the numerator.
        - :func:`dx`: The directional index built from the two indicators.

    References:
        - Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
        - https://en.wikipedia.org/wiki/Average_directional_movement_index

    Examples:
        On a small OHLC frame with a short window:

        >>> import polars as pl
        >>> from pomata.indicators import di_minus
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5],
        ...         "close": [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0],
        ...     }
        ... )
        >>> expr = di_minus(pl.col("high"), pl.col("low"), pl.col("close"), 2).round(4)
        >>> frame.select(expr.alias("di_minus"))["di_minus"].to_list()
        [None, 0.0, 0.0, 21.0526, 7.8431, 24.0964, 9.4787, 24.7788]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 20.0, 22.0, 19.0, 23.0, 20.0],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 18.0, 20.0, 17.0, 21.0, 18.0],
        ...         "close": [9.5, 10.5, 11.5, 11.0, 12.5, 19.0, 21.0, 18.0, 22.0, 19.0],
        ...     }
        ... )
        >>> expr = di_minus(pl.col("high"), pl.col("low"), pl.col("close"), 2).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("di_minus"))["di_minus"].to_list()
        [None, 0.0, 0.0, 21.0526, 7.8431, None, 0.0, 46.1538, 18.1818, 46.1538]

        A leading ``null`` ``close`` (absorbed by the ATR's true-range maximum) and a later ``NaN`` (which propagates)
        make the handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5],
        ...         "close": [None, 10.5, 11.5, 11.0, float("nan"), 12.0, 13.5, 13.0],
        ...     }
        ... )
        >>> expr = di_minus(pl.col("high"), pl.col("low"), pl.col("close"), 2).round(4)
        >>> frame.select(expr.alias("di_minus"))["di_minus"].to_list()
        [None, 0.0, 0.0, 22.2222, 8.0, nan, nan, nan]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    validate_window(window)
    return (100.0 * dm_minus(high, low, window) / atr(high, low, close, window)).name.keep()


def di_plus(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Plus Directional Indicator (+DI).

    The Wilder-smoothed plus directional movement (:func:`dm_plus`) as a percentage of the average true range
    (:func:`atr`), so upward trend pressure is comparable across instruments and bounded in ``[0, 100]``:

    .. math::

        +\mathrm{DI}_t = 100 \cdot \frac{\mathrm{dm\_plus}_t}{\mathrm{ATR}_t}.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        window: Number of observations in the Wilder moving window. Must be ``>= 1``.

    Returns:
        The plus directional indicator for each row, the same length as the inputs, in ``[0, 100]`` on complete bars.
        The first ``window - 1`` values are ``null`` (warm-up).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        It is scale-invariant under a positive common rescaling of ``high``, ``low``, and ``close`` (the smoothed
        movement and the average true range scale together).

        **Seeding:**

        The warm-up inherits the recursive Wilder seeding of :func:`rma` used throughout the cluster.

        **Edge-case behavior:**

        - **Flat series** — the ATR is an infinite-memory Wilder RMA, so ``0 / 0`` = ``NaN`` needs the whole series so
          far to be flat (both the ATR and the smoothed movement zero); a merely local flat patch after earlier
          movement leaves the ATR small-but-positive and the DI finite. The ``[0, 100]`` bound holds on complete
          coherent bars; a ``null`` prior close drops the close-based true-range terms and shrinks the ATR, so on a gap
          the ratio can exceed ``100``.
        - **Null** — a ``null`` in the smoothed movement or the ATR at a row yields ``null`` there.
        - **NaN** — a ``NaN`` propagates, yielding ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the recursions never span
          series boundaries, e.g. ``di_plus(pl.col("high"), pl.col("low"), pl.col("close"), 14).over("ticker")``.

    See Also:
        - :func:`di_minus`: The minus counterpart.
        - :func:`dm_plus`: The smoothed plus directional movement in the numerator.
        - :func:`dx`: The directional index built from the two indicators.

    References:
        - Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
        - https://en.wikipedia.org/wiki/Average_directional_movement_index

    Examples:
        On a small OHLC frame with a short window:

        >>> import polars as pl
        >>> from pomata.indicators import di_plus
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5],
        ...         "close": [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0],
        ...     }
        ... )
        >>> expr = di_plus(pl.col("high"), pl.col("low"), pl.col("close"), 2).round(4)
        >>> frame.select(expr.alias("di_plus"))["di_plus"].to_list()
        [None, 40.0, 54.5455, 31.5789, 58.8235, 36.1446, 59.7156, 37.1681]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 20.0, 22.0, 19.0, 23.0, 20.0],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 18.0, 20.0, 17.0, 21.0, 18.0],
        ...         "close": [9.5, 10.5, 11.5, 11.0, 12.5, 19.0, 21.0, 18.0, 22.0, 19.0],
        ...     }
        ... )
        >>> expr = di_plus(pl.col("high"), pl.col("low"), pl.col("close"), 2).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("di_plus"))["di_plus"].to_list()
        [None, 40.0, 54.5455, 31.5789, 58.8235, None, 40.0, 15.3846, 54.5455, 27.6923]

        A leading ``null`` ``close`` (absorbed by the ATR's true-range maximum) and a later ``NaN`` (which propagates)
        make the handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5],
        ...         "close": [None, 10.5, 11.5, 11.0, float("nan"), 12.0, 13.5, 13.0],
        ...     }
        ... )
        >>> expr = di_plus(pl.col("high"), pl.col("low"), pl.col("close"), 2).round(4)
        >>> frame.select(expr.alias("di_plus"))["di_plus"].to_list()
        [None, 50.0, 60.0, 33.3333, 60.0, nan, nan, nan]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    validate_window(window)
    return (100.0 * dm_plus(high, low, window) / atr(high, low, close, window)).name.keep()


def dm_minus(
    high: pl.Expr,
    low: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Minus Directional Movement (-DM), Wilder-smoothed.

    Part of J. Welles Wilder's directional-movement system (1978). The raw minus directional movement is the bar's
    downward range expansion — how much further the low fell than the high rose — counted only when the down-move
    leads:

    .. math::

        \mathrm{down}_t &= \mathrm{low}_{t-1} - \mathrm{low}_t, \qquad \mathrm{up}_t = \mathrm{high}_t -
            \mathrm{high}_{t-1}, \\
        -\mathrm{DM}_t &= \begin{cases} \mathrm{down}_t & \mathrm{down}_t > \mathrm{up}_t \ \text{and}\
            \mathrm{down}_t > 0 \\ 0 & \text{otherwise} \end{cases}

    The raw values are then smoothed by Wilder's moving average (:func:`rma`, smoothing factor ``1 / window``).

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        window: Number of observations in the Wilder moving window. Must be ``>= 1``.

    Returns:
        The smoothed minus directional movement for each row, the same length as the inputs. The first ``window - 1``
        values are ``null`` (warm-up), inherited from the :func:`rma`.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        It is homogeneous of degree ``1`` in a positive common rescaling of ``high`` and ``low`` (a range expansion in
        price units).

        **Seeding:**

        The raw directional movement is smoothed by Wilder's :func:`rma`, the mean-scale recursion
        ``m_t = m_{t-1} - m_{t-1} / window + raw_t / window`` (smoothing factor ``1 / window``).
        Wilder's original presentation instead smooths on the sum scale (``S_t = S_{t-1} - S_{t-1} / window + raw_t``,
        seeded from a simple sum of the first ``window`` raw movements), which equals ``window`` times the mean-scale
        value in steady state. That factor of ``window`` is structural and persists for every row — it is not a warm-up
        seed difference that washes out — so this series reads roughly ``window`` times smaller than the sum-scale
        convention throughout. The factor cancels in :func:`di_minus`, :func:`dx`, and :func:`adx`, which are therefore
        unaffected.

        **Edge-case behavior:**

        - **First bar** — row ``0`` has no previous bar, so its raw movement is ``0`` and seeds the smoothing.
        - **Null** — a ``null`` in ``high`` or ``low`` makes the affected raw movement ``0`` for the rows whose
          difference it touches, so the raw movement carries no interior nulls and the only nulls emitted are the
          ``window - 1`` warm-up nulls from :func:`rma`.
        - **NaN** — a ``NaN`` in ``low`` (the own-side input) poisons the recursion and yields ``NaN`` for every
          subsequent non-null row; a ``NaN`` in ``high`` (the opposing side) instead makes the directional comparison
          false, so the affected raw movement is sent to ``0`` and genuine downward movement is silently dropped there.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the differencing and the
          recursion never span series boundaries, e.g. ``dm_minus(pl.col("high"), pl.col("low"), 14).over("ticker")``.

    See Also:
        - :func:`dm_plus`: The plus counterpart.
        - :func:`di_minus`: The minus directional indicator built from this and the :func:`atr`.
        - :func:`rma`: The Wilder moving average that smooths the raw movement.

    References:
        - Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
        - https://en.wikipedia.org/wiki/Average_directional_movement_index

    Examples:
        On a small high/low frame with a short window:

        >>> import polars as pl
        >>> from pomata.indicators import dm_minus
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5],
        ...     }
        ... )
        >>> frame.select(dm_minus(pl.col("high"), pl.col("low"), 2).round(4).alias("dm_minus"))["dm_minus"].to_list()
        [None, 0.0, 0.0, 0.25, 0.125, 0.3125, 0.1562, 0.3281]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 20.0, 22.0, 19.0, 23.0, 20.0],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 18.0, 20.0, 17.0, 21.0, 18.0],
        ...     }
        ... )
        >>> expr = dm_minus(pl.col("high"), pl.col("low"), 2).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("dm_minus"))["dm_minus"].to_list()
        [None, 0.0, 0.0, 0.25, 0.125, None, 0.0, 1.5, 0.75, 1.875]

        On a falling frame, a leading ``null`` ``low`` (which zeroes the raw movement it touches) and a later ``NaN``
        ``low`` (the own side, which poisons the recursion) make the handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [9.0, 8.0, 7.5, 6.5, 7.0, 6.0, 5.5, 5.0],
        ...         "low": [None, 7.0, 6.5, 5.5, float("nan"), 5.0, 4.5, 4.0],
        ...     }
        ... )
        >>> frame.select(dm_minus(pl.col("high"), pl.col("low"), 2).round(4).alias("dm_minus"))["dm_minus"].to_list()
        [None, 0.0, 0.25, 0.625, nan, nan, nan, nan]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    validate_window(window)
    up = high - high.shift(1)
    down = low.shift(1) - low
    raw = pl.when((down > up) & (down > 0)).then(down).otherwise(0.0)
    return rma(raw, window)


def dm_plus(
    high: pl.Expr,
    low: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Plus Directional Movement (+DM), Wilder-smoothed.

    Part of J. Welles Wilder's directional-movement system (1978). The raw plus directional movement is the bar's upward
    range expansion — how much further the high rose than the low fell — counted only when the up-move leads:

    .. math::

        \mathrm{up}_t &= \mathrm{high}_t - \mathrm{high}_{t-1}, \qquad \mathrm{down}_t = \mathrm{low}_{t-1} -
            \mathrm{low}_t, \\
        +\mathrm{DM}_t &= \begin{cases} \mathrm{up}_t & \mathrm{up}_t > \mathrm{down}_t \ \text{and}\ \mathrm{up}_t > 0
            \\ 0 & \text{otherwise} \end{cases}

    The raw values are then smoothed by Wilder's moving average (:func:`rma`, smoothing factor ``1 / window``).

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        window: Number of observations in the Wilder moving window. Must be ``>= 1``.

    Returns:
        The smoothed plus directional movement for each row, the same length as the inputs. The first ``window - 1``
        values are ``null`` (warm-up), inherited from the :func:`rma`.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        It is homogeneous of degree ``1`` in a positive common rescaling of ``high`` and ``low`` (a range expansion in
        price units).

        **Seeding:**

        The raw directional movement is smoothed by Wilder's :func:`rma`, the mean-scale recursion
        ``m_t = m_{t-1} - m_{t-1} / window + raw_t / window`` (smoothing factor ``1 / window``).
        Wilder's original presentation instead smooths on the sum scale (``S_t = S_{t-1} - S_{t-1} / window + raw_t``,
        seeded from a simple sum of the first ``window`` raw movements), which equals ``window`` times the mean-scale
        value in steady state. That factor of ``window`` is structural and persists for every row — it is not a warm-up
        seed difference that washes out — so this series reads roughly ``window`` times smaller than the sum-scale
        convention throughout. The factor cancels in :func:`di_plus`, :func:`dx`, and :func:`adx`, which are therefore
        unaffected.

        **Edge-case behavior:**

        - **First bar** — row ``0`` has no previous bar, so its raw movement is ``0`` and seeds the smoothing.
        - **Null** — a ``null`` in ``high`` or ``low`` makes the affected raw movement ``0`` for the rows whose
          difference it touches, so the raw movement carries no interior nulls and the only nulls emitted are the
          ``window - 1`` warm-up nulls from :func:`rma`.
        - **NaN** — a ``NaN`` in ``high`` (the own-side input) poisons the recursion and yields ``NaN`` for every
          subsequent non-null row; a ``NaN`` in ``low`` (the opposing side) instead makes the directional comparison
          false, so the affected raw movement is sent to ``0`` and genuine upward movement is silently dropped there.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the differencing and the
          recursion never span series boundaries, e.g. ``dm_plus(pl.col("high"), pl.col("low"), 14).over("ticker")``.

    See Also:
        - :func:`dm_minus`: The minus counterpart.
        - :func:`di_plus`: The plus directional indicator built from this and the :func:`atr`.
        - :func:`rma`: The Wilder moving average that smooths the raw movement.

    References:
        - Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
        - https://en.wikipedia.org/wiki/Average_directional_movement_index

    Examples:
        On a small high/low frame with a short window:

        >>> import polars as pl
        >>> from pomata.indicators import dm_plus
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5],
        ...     }
        ... )
        >>> frame.select(dm_plus(pl.col("high"), pl.col("low"), 2).round(4).alias("dm_plus"))["dm_plus"].to_list()
        [None, 0.5, 0.75, 0.375, 0.9375, 0.4688, 0.9844, 0.4922]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 20.0, 22.0, 19.0, 23.0, 20.0],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 18.0, 20.0, 17.0, 21.0, 18.0],
        ...     }
        ... )
        >>> expr = dm_plus(pl.col("high"), pl.col("low"), 2).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("dm_plus"))["dm_plus"].to_list()
        [None, 0.5, 0.75, 0.375, 0.9375, None, 1.0, 0.5, 2.25, 1.125]

        A leading ``null`` ``high`` (which zeroes the raw movement it touches) and a later ``NaN`` ``high`` (the own
        side, which poisons the recursion) make the handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [None, 11.0, 12.0, 11.5, float("nan"), 12.5, 14.0, 13.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5],
        ...     }
        ... )
        >>> frame.select(dm_plus(pl.col("high"), pl.col("low"), 2).round(4).alias("dm_plus"))["dm_plus"].to_list()
        [None, 0.0, 0.5, 0.25, nan, nan, nan, nan]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    validate_window(window)
    up = high - high.shift(1)
    down = low.shift(1) - low
    raw = pl.when((up > down) & (up > 0)).then(up).otherwise(0.0)
    return rma(raw, window)


def dx(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Directional Index (DX).

    The normalized spread between the plus and minus directional indicators — how one-sided the trend is, bounded in
    ``[0, 100]`` (``0`` when up- and down-pressure are equal, ``100`` when only one side moves):

    .. math::

        \mathrm{DX}_t = 100 \cdot \frac{\lvert +\mathrm{DI}_t - (-\mathrm{DI}_t) \rvert}{+\mathrm{DI}_t +
            (-\mathrm{DI}_t)}.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        window: Number of observations in the Wilder moving window. Must be ``>= 1``.

    Returns:
        The directional index for each row, the same length as the inputs, in ``[0, 100]``. The first ``window - 1``
        values are ``null`` (warm-up), inherited from the directional indicators.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        It is scale-invariant under a positive common rescaling of ``high``, ``low``, and ``close``.

        **Seeding:**

        The warm-up inherits the recursive Wilder seeding of :func:`rma` used throughout the cluster.

        **Edge-case behavior:**

        - **Flat directional movement** — when ``+DI`` and ``-DI`` are both zero (no movement either way) the
          denominator is zero, so the result follows IEEE-754: the numerator is also zero, hence ``0 / 0`` is ``NaN``.
        - **Null** — a ``null`` in either indicator at a row yields ``null`` there.
        - **NaN** — a ``NaN`` propagates, yielding ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the recursions never span
          series boundaries, e.g. ``dx(pl.col("high"), pl.col("low"), pl.col("close"), 14).over("ticker")``.

    See Also:
        - :func:`di_plus`: The plus directional indicator.
        - :func:`di_minus`: The minus directional indicator.
        - :func:`adx`: The Wilder-smoothed average of this.

    References:
        - Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
        - https://en.wikipedia.org/wiki/Average_directional_movement_index

    Examples:
        On a small OHLC frame with a short window:

        >>> import polars as pl
        >>> from pomata.indicators import dx
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5],
        ...         "close": [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0],
        ...     }
        ... )
        >>> expr = dx(pl.col("high"), pl.col("low"), pl.col("close"), 2).round(4)
        >>> frame.select(expr.alias("dx"))["dx"].to_list()
        [None, 100.0, 100.0, 20.0, 76.4706, 20.0, 72.6027, 20.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 20.0, 22.0, 19.0, 23.0, 20.0],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 18.0, 20.0, 17.0, 21.0, 18.0],
        ...         "close": [9.5, 10.5, 11.5, 11.0, 12.5, 19.0, 21.0, 18.0, 22.0, 19.0],
        ...     }
        ... )
        >>> expr = dx(pl.col("high"), pl.col("low"), pl.col("close"), 2).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("dx"))["dx"].to_list()
        [None, 100.0, 100.0, 20.0, 76.4706, None, 100.0, 50.0, 50.0, 25.0]

        A leading ``null`` ``close`` (absorbed by the underlying ATR's true-range maximum) and a later ``NaN`` (which
        propagates through the directional indicators) make the handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5],
        ...         "close": [None, 10.5, 11.5, 11.0, float("nan"), 12.0, 13.5, 13.0],
        ...     }
        ... )
        >>> expr = dx(pl.col("high"), pl.col("low"), pl.col("close"), 2).round(4)
        >>> frame.select(expr.alias("dx"))["dx"].to_list()
        [None, 100.0, 100.0, 20.0, 76.4706, nan, nan, nan]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    validate_window(window)
    plus = di_plus(high, low, close, window)
    minus = di_minus(high, low, close, window)
    return (100.0 * (plus - minus).abs() / (plus + minus)).name.keep()


def vortex(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Vortex Indicator (Botes & Siepman) — paired oscillators tracking upward and downward trend movement.

    Etienne Botes and Douglas Siepman's trend gauge: positive vortex movement ``|high_t - low_{t-1}|`` and negative
    ``|low_t - high_{t-1}|``, each summed over the window and normalized by the summed true range. ``VI+`` above ``VI-``
    signals an uptrend, and their crosses mark trend changes:

    .. math::

        \mathrm{VI}^{+}_t = \frac{\sum |\mathrm{high}_i - \mathrm{low}_{i-1}|}{\sum \mathrm{TR}_i}, \qquad
            \mathrm{VI}^{-}_t = \frac{\sum |\mathrm{low}_i - \mathrm{high}_{i-1}|}{\sum \mathrm{TR}_i},

    each sum running over the ``window`` bars ending at :math:`t`, with :math:`\mathrm{TR}` the :func:`true_range`.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        A struct column (one struct per row, the same length as the inputs) with two ``Float64`` fields:

        - ``plus`` — the positive vortex line ``VI+``.
        - ``minus`` — the negative vortex line ``VI-``.

        Read one line with ``.struct.field("plus")`` (etc.) or split both into columns with ``.struct.unnest()``. The
        first ``window`` rows are ``null`` (warm-up): each line needs ``window`` defined vortex movements, and the first
        movement is ``null`` (it reads the previous bar).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        **Inputs:**

        ``high`` / ``low`` / ``close`` must share a length and alignment (the same row index is one bar).

        **Edge-case behavior:**

        - **Null / NaN** — a ``null`` / ``NaN`` in the window (including via the one-bar lag, which makes the first
          movement ``null``) propagates to the affected line at that row.
        - **Flat window** — a flat window (zero summed true range and zero summed movement — the ``0 / 0`` degenerate)
          is detected per line via the residual-free rolling maxima of the true range and the movement, and returns
          ``NaN``. A near-flat window (tiny ranges after a much larger one has slid out) is not silenced: ``VI+`` is
          unbounded above, so the streaming quotient cannot be clipped to a range and, past a sane dynamic range,
          degrades in precision (see the precision note above).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so neither the lag nor the window
          spans series boundaries.

    See Also:
        - :func:`di_plus`: The Wilder directional indicator, the same movement-over-range idea, exponentially smoothed.
        - :func:`true_range`: The per-bar basis of the shared denominator.
        - :func:`di_minus`: The minus directional indicator, the Wilder analog of the negative vortex line.

    References:
        - Botes, E. & Siepman, D. (2010). "The Vortex Indicator." *Technical Analysis of Stocks & Commodities*, 28(1),
          20-30.
        - https://en.wikipedia.org/wiki/Vortex_indicator

    Examples:
        On a small OHLC frame, reading each vortex line with ``.struct.field``:

        >>> import polars as pl
        >>> from pomata.indicators import vortex
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [2.0, 4.0, 6.0, 5.0, 7.0, 6.5, 8.0, 7.5],
        ...         "low": [1.0, 3.0, 4.0, 4.0, 5.0, 5.5, 6.0, 6.5],
        ...         "close": [1.5, 3.5, 5.0, 4.5, 6.0, 6.0, 7.0, 7.0],
        ...     }
        ... )
        >>> bands = vortex(pl.col("high"), pl.col("low"), pl.col("close"), 2)
        >>> frame.select(bands.struct.field("plus").round(4).alias("plus"))["plus"].to_list()
        [None, None, 1.2, 1.1429, 1.1429, 1.2857, 1.3333, 1.3333]
        >>> frame.select(bands.struct.field("minus").round(4).alias("minus"))["minus"].to_list()
        [None, None, 0.2, 0.5714, 0.5714, 0.4286, 0.6667, 0.6667]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "high": [2.0, 4.0, 6.0, 5.0, 7.0, 12.0, 11.0, 13.0, 10.0, 12.0],
        ...         "low": [1.0, 3.0, 4.0, 4.0, 5.0, 10.0, 9.0, 11.0, 8.0, 10.0],
        ...         "close": [1.5, 3.5, 5.0, 4.5, 6.0, 11.0, 10.0, 12.0, 9.0, 11.0],
        ...     }
        ... )
        >>> expr = vortex(pl.col("high"), pl.col("low"), pl.col("close"), 2).over("ticker")
        >>> frame.with_columns(expr.struct.field("plus").round(4).alias("plus"))["plus"].to_list()
        [None, None, 1.2, 1.1429, 1.1429, None, None, 1.0, 0.7143, 0.7143]
        >>> frame.with_columns(expr.struct.field("minus").round(4).alias("minus"))["minus"].to_list()
        [None, None, 0.2, 0.5714, 0.5714, None, None, 0.6, 0.7143, 0.7143]

        A leading ``null`` ``close`` (absorbed by the true-range maximum) and a later ``NaN`` (which contaminates only
        the bars whose window spans it, then clears) make the handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [2.0, 4.0, 6.0, 5.0, 7.0, 6.5, 8.0, 7.5],
        ...         "low": [1.0, 3.0, 4.0, 4.0, 5.0, 5.5, 6.0, 6.5],
        ...         "close": [None, 3.5, 5.0, 4.5, float("nan"), 6.0, 7.0, 7.0],
        ...     }
        ... )
        >>> bands = vortex(pl.col("high"), pl.col("low"), pl.col("close"), 2)
        >>> frame.select(bands.struct.field("plus").round(4).alias("plus"))["plus"].to_list()
        [None, None, 1.7143, 1.1429, 1.1429, nan, nan, 1.3333]
        >>> frame.select(bands.struct.field("minus").round(4).alias("minus"))["minus"].to_list()
        [None, None, 0.2857, 0.5714, 0.5714, nan, nan, 0.6667]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    validate_window(window)
    # Summed vortex movements over the summed true range; the shared denominator is computed once (CSE).
    range_per_bar = true_range(high, low, close)
    plus_per_bar = (high - low.shift(1)).abs()
    minus_per_bar = (low - high.shift(1)).abs()
    range_sum = range_per_bar.rolling_sum(window)
    plus_quotient = plus_per_bar.rolling_sum(window) / range_sum
    minus_quotient = minus_per_bar.rolling_sum(window) / range_sum
    # The genuine 0/0 degenerate (a flat window: zero summed true range AND zero summed movement) is detected
    # residual-free via the rolling maxima and returned as NaN deterministically, per line. A non-zero movement over a
    # zero true range -- reachable only with impossible bars (a close outside the high-low range) -- is left to IEEE-754
    # as inf. The near-flat residual is reported as-is, not clipped: VI+ is unbounded above, so there is no range
    # to clip to (beyond a sane dynamic range the streaming quotient degrades -- see CORRECTNESS.md).
    range_flat = range_per_bar.rolling_max(window) == 0
    plus_flat = range_flat & (plus_per_bar.rolling_max(window) == 0)
    minus_flat = range_flat & (minus_per_bar.rolling_max(window) == 0)
    plus = pl.when(plus_flat).then(float("nan")).otherwise(plus_quotient)
    minus = pl.when(minus_flat).then(float("nan")).otherwise(minus_quotient)
    return pl.struct(plus=plus, minus=minus)
