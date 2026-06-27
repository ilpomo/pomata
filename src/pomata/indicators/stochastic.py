"""
Stochastic oscillators — %K / %D momentum from the high-low range.
"""

import polars as pl

from pomata._expr import float64_expr, validate_window
from pomata.indicators.moving_average import sma

__all__ = ("stochastic_fast", "stochastic_slow")


def _raw_percent_k(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    window_k: int,
) -> pl.Expr:
    """
    Raw stochastic %K: the close's position within the ``window_k`` high-low range, scaled to ``[0, 100]``.

    The shared first step of :func:`stochastic_fast` (which smooths it into %D) and :func:`stochastic_slow` (which
    slows it before smoothing). A flat range (``highest_high == lowest_low``) gives ``0 / 0`` per IEEE, surfaced as
    each factory documents.
    """
    lowest_low = low.rolling_min(window_k)
    highest_high = high.rolling_max(window_k)
    return 100.0 * (close - lowest_low) / (highest_high - lowest_low)


def stochastic_fast(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    *,
    window_k: int,
    window_d: int,
) -> pl.Expr:
    r"""
    Fast Stochastic Oscillator (%K and %D).

    Introduced by George Lane in the late 1950s: a bounded momentum oscillator that locates the close within its recent
    high-low range. The raw line %K is the close as a percentage of the ``window_k`` range, and the signal line %D is
    the :func:`sma` of %K:

    .. math::

        \%\mathrm{K}_t &= 100 \cdot \frac{\mathrm{close}_t - \mathrm{LL}_t}{\mathrm{HH}_t - \mathrm{LL}_t}, \\
        \%\mathrm{D}_t &= \mathrm{SMA}(\%\mathrm{K}, m)_t,

    where :math:`\mathrm{LL}_t` and :math:`\mathrm{HH}_t` are the lowest low and highest high over the ``window_k`` bars
    ending at :math:`t`, and :math:`m` is ``window_d``.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        window_k: Number of observations in the %K look-back range (canonically ``14``). Must be ``>= 1``.
        window_d: Number of observations in the %D moving average of %K (canonically ``3``). Must be ``>= 1``.

    Returns:
        A struct column (one struct per row, the same length as the inputs) with two ``Float64`` fields:

        - ``k`` — the raw %K line, ``100 * (close - LL) / (HH - LL)``.
        - ``d`` — the %D signal line, the :func:`sma` of %K over ``window_d``.

        Read one line with ``.struct.field("k")`` (etc.) or split both into columns with ``.struct.unnest()``. The first
        ``window_k - 1`` rows are ``null`` on ``k`` (the look-back warm-up), and a further ``window_d - 1`` on ``d``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window_k < 1`` or ``window_d < 1``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        Both lines are scale-invariant under a positive common rescaling of ``high``, ``low``, and ``close`` (a ratio of
        price ranges), and lie in ``[0, 100]`` for well-formed bars (``low <= close <= high``).

        **Composition:**

        %D is the :func:`sma` of %K, so it inherits that warm-up and the null / NaN handling on top of %K's own.

        **Edge-case behavior:**

        - **Null** — a ``null`` anywhere in the %K window (a ``high`` / ``low`` over the look-back, or the current
          ``close``) yields ``null`` on ``k`` at that row; a ``null`` reaching the %D average yields ``null`` on ``d``.
        - **NaN** — a ``NaN`` in the window propagates, yielding ``NaN``.
        - **Flat range** — when the highest ``high`` equals the lowest ``low`` (no range over the look-back) the
          denominator is zero, so ``k`` follows IEEE-754: ``0 / 0`` is ``NaN`` when the close sits on that flat level,
          and ``+/-inf`` when it lies outside it (a gap or malformed bar).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so no window spans series
          boundaries, e.g. ``stochastic_fast(pl.col("high"), pl.col("low"), pl.col("close")).over("ticker")``.

    See Also:
        - :func:`stochastic_slow`: The slow variant, %K smoothed once more before %D.
        - :func:`rsi_stochastic`: The same oscillator applied to :func:`rsi` instead of price.
        - :func:`sma`: The moving average that forms %D.

    References:
        - Lane, George C. (1984). "Lane's Stochastics." *Technical Analysis of Stocks & Commodities*, 2(3).
        - https://en.wikipedia.org/wiki/Stochastic_oscillator
        - https://www.investopedia.com/terms/s/stochasticoscillator.asp

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import stochastic_fast
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5],
        ...         "close": [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0],
        ...     }
        ... )
        >>> oscillator = stochastic_fast(pl.col("high"), pl.col("low"), pl.col("close"), window_k=3, window_d=2)
        >>> frame.select(oscillator.struct.field("k").round(4).alias("k"))["k"].to_list()
        [None, None, 83.3333, 50.0, 80.0, 60.0, 80.0, 60.0]
        >>> frame.select(oscillator.struct.field("d").round(4).alias("d"))["d"].to_list()
        [None, None, None, 66.6667, 65.0, 70.0, 70.0, 70.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 4 + ["B"] * 4,
        ...         "high": [10.0, 11.0, 12.0, 11.5, 20.0, 21.0, 22.0, 21.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 19.0, 20.0, 21.0, 20.5],
        ...         "close": [9.5, 10.5, 11.5, 11.0, 19.5, 20.5, 21.5, 21.0],
        ...     }
        ... )
        >>> expr = stochastic_fast(pl.col("high"), pl.col("low"), pl.col("close"), window_k=2, window_d=2)
        >>> frame.with_columns(expr.over("ticker").struct.field("k").round(4).alias("k"))["k"].to_list()
        [None, 75.0, 75.0, 33.3333, None, 75.0, 75.0, 33.3333]

        A ``null`` (yields ``null`` on ``k`` at that row) and a ``NaN`` (which propagates) in ``close`` surface on
        the %K line:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5],
        ...         "close": [9.5, 10.5, 11.5, None, 12.5, float("nan"), 13.5, 13.0],
        ...     }
        ... )
        >>> oscillator = stochastic_fast(pl.col("high"), pl.col("low"), pl.col("close"), window_k=3, window_d=2)
        >>> frame.select(oscillator.struct.field("k").round(4).alias("k"))["k"].to_list()
        [None, None, 83.3333, None, 80.0, nan, 80.0, 60.0]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    validate_window(window_k, name="window_k")
    validate_window(window_d, name="window_d")
    percent_k = _raw_percent_k(high, low, close, window_k)
    return pl.struct(k=percent_k, d=sma(percent_k, window_d))


def stochastic_slow(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    *,
    window_k: int,
    window_slowing: int,
    window_d: int,
) -> pl.Expr:
    r"""
    Slow Stochastic Oscillator (%K and %D).

    The smoothed form of :func:`stochastic_fast`: the raw %K is averaged once to give the slow %K (which damps the noise
    of the fast line), then averaged again to give the slow %D signal:

    .. math::

        \mathrm{raw}_t &= 100 \cdot \frac{\mathrm{close}_t - \mathrm{LL}_t}{\mathrm{HH}_t - \mathrm{LL}_t}, \\
        \%\mathrm{K}_t &= \mathrm{SMA}(\mathrm{raw}, p)_t, \\
        \%\mathrm{D}_t &= \mathrm{SMA}(\%\mathrm{K}, m)_t,

    where :math:`\mathrm{LL}_t` and :math:`\mathrm{HH}_t` are the lowest low and highest high over the ``window_k`` bars
    ending at :math:`t`, :math:`p` is ``window_slowing``, and :math:`m` is ``window_d``.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        window_k: Number of observations in the raw %K look-back range (canonically ``14``). Must be ``>= 1``.
        window_slowing: Number of observations in the slowing average that turns the raw %K into the slow %K
            (canonically ``3``). Must be ``>= 1``.
        window_d: Number of observations in the %D moving average of the slow %K (canonically ``3``). Must be ``>= 1``.

    Returns:
        A struct column (one struct per row, the same length as the inputs) with two ``Float64`` fields:

        - ``k`` — the slow %K line, the :func:`sma` of the raw %K over ``window_slowing``.
        - ``d`` — the %D signal line, the :func:`sma` of the slow %K over ``window_d``.

        Read one line with ``.struct.field("k")`` (etc.) or split both into columns with ``.struct.unnest()``. The first
        ``window_k + window_slowing - 2`` rows are ``null`` on ``k``, and a further ``window_d - 1`` on ``d``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window_k < 1``, ``window_slowing < 1``, or ``window_d < 1``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        Both lines are scale-invariant under a positive common rescaling of ``high``, ``low``, and ``close``, and lie in
        ``[0, 100]`` for well-formed bars (``low <= close <= high``). The slow %K equals the fast %D of
        :func:`stochastic_fast` when ``window_slowing`` matches that call's ``window_d``.

        **Composition:**

        The slow %K is the :func:`sma` of the raw %K, and %D is the :func:`sma` of the slow %K, so each averaging
        inherits the warm-up and null / NaN handling on top of the raw %K's own.

        **Edge-case behavior:**

        - **Null** — a ``null`` anywhere in a window yields ``null`` on the dependent field at that row.
        - **NaN** — a ``NaN`` in a window propagates, yielding ``NaN``.
        - **Flat range** — when the highest ``high`` equals the lowest ``low`` (no range over the look-back) the
          raw %K is ``0 / 0 = NaN`` when the close sits on that flat level (``+/-inf`` when it lies outside it), which
          then propagates through both averages.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so no window spans series
          boundaries, e.g. ``stochastic_slow(pl.col("high"), pl.col("low"), pl.col("close")).over("ticker")``.

    See Also:
        - :func:`stochastic_fast`: The unsmoothed variant, whose raw %K this smooths.
        - :func:`rsi_stochastic`: The stochastic applied to :func:`rsi` instead of price.
        - :func:`sma`: The moving average behind both the slowing and %D.

    References:
        - Lane, George C. (1984). "Lane's Stochastics." *Technical Analysis of Stocks & Commodities*, 2(3).
        - https://en.wikipedia.org/wiki/Stochastic_oscillator
        - https://www.investopedia.com/terms/s/stochasticoscillator.asp

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import stochastic_slow
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5],
        ...         "close": [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0],
        ...     }
        ... )
        >>> oscillator = stochastic_slow(
        ...     pl.col("high"), pl.col("low"), pl.col("close"), window_k=3, window_slowing=2, window_d=2
        ... )
        >>> frame.select(oscillator.struct.field("k").round(4).alias("k"))["k"].to_list()
        [None, None, None, 66.6667, 65.0, 70.0, 70.0, 70.0]
        >>> frame.select(oscillator.struct.field("d").round(4).alias("d"))["d"].to_list()
        [None, None, None, None, 65.8333, 67.5, 70.0, 70.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 20.0, 21.0, 22.0, 21.5, 23.0],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 19.0, 20.0, 21.0, 20.5, 22.0],
        ...         "close": [9.5, 10.5, 11.5, 11.0, 12.5, 19.5, 20.5, 21.5, 21.0, 22.5],
        ...     }
        ... )
        >>> expr = stochastic_slow(
        ...     pl.col("high"), pl.col("low"), pl.col("close"), window_k=2, window_slowing=2, window_d=2
        ... )
        >>> frame.with_columns(expr.over("ticker").struct.field("k").round(4).alias("k"))["k"].to_list()
        [None, None, 75.0, 54.1667, 56.6667, None, None, 75.0, 54.1667, 56.6667]

        A ``null`` (nulls every slow %K window it falls in) and a ``NaN`` (which propagates the same way) in
        ``close`` surface on the slow %K line:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5],
        ...         "close": [9.5, 10.5, 11.5, None, 12.5, float("nan"), 13.5, 13.0],
        ...     }
        ... )
        >>> oscillator = stochastic_slow(
        ...     pl.col("high"), pl.col("low"), pl.col("close"), window_k=2, window_slowing=2, window_d=2
        ... )
        >>> frame.select(oscillator.struct.field("k").round(4).alias("k"))["k"].to_list()
        [None, None, 75.0, None, None, nan, nan, 56.6667]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    validate_window(window_k, name="window_k")
    validate_window(window_slowing, name="window_slowing")
    validate_window(window_d, name="window_d")
    raw_k = _raw_percent_k(high, low, close, window_k)
    slow_k = sma(raw_k, window_slowing)
    return pl.struct(k=slow_k, d=sma(slow_k, window_d))
