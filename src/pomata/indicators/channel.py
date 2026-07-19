"""
Range / channel indicators (the midline of a price range over a window).
"""

import polars as pl

from pomata._expr import float64_expr, validate_positive, validate_window
from pomata.indicators.moving_average import ema
from pomata.indicators.volatility import atr

__all__ = ("donchian_channels", "ichimoku", "keltner_channels", "midpoint", "midprice")


def donchian_channels(
    high: pl.Expr,
    low: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Donchian Channels — the highest high and lowest low over a trailing window, with their midline.

    Introduced by Richard Donchian: an upper band tracking the window's highest ``high``, a lower band tracking its
    lowest ``low``, and a middle band halfway between. A breakout system reads price against a channel that widens only
    to admit a new extreme and never narrows within the window:

    .. math::

        \mathrm{upper}_t &= \max(\mathrm{high}_{t-n+1 \ldots t}), \\
        \mathrm{lower}_t &= \min(\mathrm{low}_{t-n+1 \ldots t}), \\
        \mathrm{middle}_t &= \frac{\mathrm{upper}_t + \mathrm{lower}_t}{2}, \qquad n = \text{window}.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        window: Number of observations in the moving window (canonically ``20``, the Donchian period). Must be ``>= 1``.

    Returns:
        A struct ``pl.Expr`` with three ``Float64`` fields, the same length as the inputs:

        - ``lower`` — the lowest ``low`` over the window.
        - ``middle`` — the channel midline, ``(upper + lower) / 2`` (identical to :func:`midprice`).
        - ``upper`` — the highest ``high`` over the window.

        Read one band with ``.struct.field("upper")`` (etc.) or split all three into columns with ``.struct.unnest()``.
        The first ``window - 1`` rows are ``null`` (warm-up).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Inputs**

        ``high`` and ``low`` must share a length and alignment (the same row index is one bar). The channel does not
        assume ``high >= low``: a malformed bar where ``high < low`` flows through unchanged (the upper band can then
        sit below the lower band) rather than being silently reordered.

        **Edge-case behavior**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values) —
          a ``null`` in the ``high`` window nulls ``upper`` and ``middle``, a ``null`` in the ``low`` window nulls
          ``lower`` and ``middle``; a fully missing bar nulls all three.
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there — per band, with ``null`` still
          taking precedence over ``NaN``.
        - **window == 1** — the bands are the bar's own ``high`` and ``low``, and the middle is its
          :func:`price_median`.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`midprice`: The channel's middle band on its own.
        - :func:`keltner_channels`: The same band shape, an EMA midline with ATR width instead of window extremes.
        - :func:`bollinger_bands`: Volatility bands around a moving average rather than around the window's extremes.

    References:
        - https://en.wikipedia.org/wiki/Donchian_channel

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import donchian_channels
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [11.0, 12.0, 13.0, 12.5, 14.0],
        ...         "low": [9.0, 10.0, 11.0, 11.0, 12.0],
        ...     }
        ... )
        >>> expr = donchian_channels(pl.col("high"), pl.col("low"), window=3)
        >>> frame.select(upper=expr.struct.field("upper").round(4))["upper"].to_list()
        [None, None, 13.0, 13.0, 14.0]
        >>> frame.select(lower=expr.struct.field("lower").round(4))["lower"].to_list()
        [None, None, 9.0, 10.0, 11.0]
        >>> frame.select(middle=expr.struct.field("middle").round(4))["middle"].to_list()
        [None, None, 11.0, 11.5, 12.5]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker's bands warm up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 3 + ["NVDA"] * 3,
        ...         "high": [11.0, 12.0, 13.0] + [21.0, 22.0, 23.0],
        ...         "low": [9.0, 10.0, 11.0] + [19.0, 20.0, 21.0],
        ...     }
        ... )
        >>> expr = donchian_channels(pl.col("high"), pl.col("low"), window=2)
        >>> frame.with_columns(middle=expr.over("ticker").struct.field("middle"))["middle"].to_list()
        [None, 10.5, 11.5, None, 20.5, 21.5]

        A ``null`` (nulling every band whose window reads it — here ``upper`` and ``middle``, while ``lower`` stays
        defined) and a ``NaN`` (which propagates) make the handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [11.0, None, 13.0, float("nan"), 15.0],
        ...         "low": [9.0, 10.0, 11.0, 12.0, 13.0],
        ...     }
        ... )
        >>> expr = donchian_channels(pl.col("high"), pl.col("low"), window=2)
        >>> frame.select(middle=expr.struct.field("middle"))["middle"].to_list()
        [None, None, None, nan, nan]

        **window == 1** — a single-bar window makes the upper and lower bands the bar's own ``high`` and ``low``, and
        the middle their mean, with no warm-up:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [11.0, 12.0, 13.0],
        ...         "low": [9.0, 10.0, 11.0],
        ...     }
        ... )
        >>> expr = donchian_channels(pl.col("high"), pl.col("low"), window=1)
        >>> frame.select(lower=expr.struct.field("lower"))["lower"].to_list()
        [9.0, 10.0, 11.0]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    validate_window(window)
    # Window extremes and their midline; rolling_max/min share min_samples=window, so null/NaN propagate per window.
    upper = high.rolling_max(window)
    lower = low.rolling_min(window)
    return (pl.struct(lower=lower, middle=(upper + lower) / 2.0, upper=upper)).name.keep()


def ichimoku(
    high: pl.Expr,
    low: pl.Expr,
    *,
    window_tenkan: int,
    window_kijun: int,
    window_senkou: int,
) -> pl.Expr:
    r"""
    Ichimoku Kinkō Hyō (Ichimoku Cloud) — rolling high-low midpoints over three horizons: two lines and the cloud.

    Introduced by Goichi Hosoda: a one-glance equilibrium chart built from rolling midpoints of the high-low range over
    three horizons. The conversion line (*tenkan-sen*) and base line (*kijun-sen*) are short and medium midpoints, and
    the two leading spans (*senkou span A* and *senkou span B*) that bound the cloud are the midpoint of those two and a
    long midpoint:

    .. math::

        \mathrm{tenkan}_t &= \frac{\max(\mathrm{high}_{t-a+1 \ldots t}) + \min(\mathrm{low}_{t-a+1 \ldots t})}{2}, \\
        \mathrm{kijun}_t &= \frac{\max(\mathrm{high}_{t-b+1 \ldots t}) + \min(\mathrm{low}_{t-b+1 \ldots t})}{2}, \\
        \mathrm{senkou\_a}_t &= \frac{\mathrm{tenkan}_t + \mathrm{kijun}_t}{2}, \\
        \mathrm{senkou\_b}_t &= \frac{\max(\mathrm{high}_{t-c+1 \ldots t}) + \min(\mathrm{low}_{t-c+1 \ldots t})}{2},

    with ``a = window_tenkan``, ``b = window_kijun``, ``c = window_senkou``.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        window_tenkan: Conversion-line window (canonically ``9``). Must be ``>= 1``.
        window_kijun: Base-line window (canonically ``26``). Must be ``>= 1`` and ``>= window_tenkan``.
        window_senkou: Leading-span-B window (canonically ``52``). Must be ``>= 1`` and ``>= window_kijun``.

    Returns:
        A struct ``pl.Expr`` with four ``Float64`` fields, the same length as the inputs:

        - ``tenkan`` — the conversion line; ``null`` through its ``window_tenkan - 1``-row warm-up.
        - ``kijun`` — the base line; ``null`` through ``window_kijun - 1`` rows.
        - ``senkou_a`` — the cloud's first bound; ``null`` through ``window_kijun - 1`` rows (it needs both lines).
        - ``senkou_b`` — the cloud's second bound; ``null`` through ``window_senkou - 1`` rows.

        Read a field with ``.struct.field("tenkan")`` or split all four with ``.struct.unnest()``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If any window is ``< 1``, or the windows are not ordered
            ``window_tenkan <= window_kijun <= window_senkou`` (equality is allowed and collapses the corresponding
            lines onto each other).

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        Every line is homogeneous of degree ``1`` under a positive common rescaling of ``high`` and ``low`` (each is a
        midpoint of price extremes).

        **Displacement (no lookahead)**

        Each line is emitted aligned to the row it is computed from -- zero displacement -- so the output never reads a
        future bar and is safe to feed a backtest directly. The traditional chart instead plots the two leading spans
        ``window_kijun`` bars into the future and a *chikou* (lagging) span ``window_kijun`` bars into the past; that is
        a presentation choice, applied on the user's side with ``.shift(...)`` -- e.g.
        ``...struct.field("senkou_a") .shift(window_kijun)`` to lead, ``pl.col("close").shift(-window_kijun)`` to lag.
        The chikou span is deliberately not emitted: un-displaced it is identical to ``close``, and its backward shift
        reads future bars, which must never enter a backtest.

        **Edge-case behavior**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window_tenkan`` non-null
          values) — each line nulls only where its own window touches the ``null``, in either ``high`` or ``low``.
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there — per line, via the underlying
          rolling extremes, with ``null`` still taking precedence over ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`midprice`: The single rolling high-low midpoint each Ichimoku line is built from.
        - :func:`donchian_channels`: The same window extremes kept as separate bands rather than midpoints.
        - :func:`keltner_channels`: Another channel, an EMA midline with ATR bands rather than rolling midpoints.

    References:
        - Hosoda, G. (1969). *Ichimoku Kinkō Hyō*.
        - https://en.wikipedia.org/wiki/Ichimoku_Kink%C5%8D_Hy%C5%8D

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import ichimoku
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0, 13.0],
        ...         "low": [8.0, 9.0, 10.0, 11.0, 12.0, 10.0, 12.0, 11.0],
        ...     }
        ... )
        >>> expr = ichimoku(pl.col("high"), pl.col("low"), window_tenkan=2, window_kijun=3, window_senkou=4)
        >>> frame.select(tenkan=expr.struct.field("tenkan").round(4))["tenkan"].to_list()
        [None, 10.0, 10.5, 11.5, 12.5, 12.0, 12.5, 13.0]
        >>> frame.select(senkou_b=expr.struct.field("senkou_b").round(4))["senkou_b"].to_list()
        [None, None, None, 10.5, 11.5, 12.0, 12.5, 12.5]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker's lines warm up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 5 + ["NVDA"] * 5,
        ...         "high": [10.0, 12.0, 11.0, 13.0, 14.0] + [20.0, 22.0, 21.0, 23.0, 24.0],
        ...         "low": [8.0, 9.0, 10.0, 11.0, 12.0] + [18.0, 19.0, 20.0, 21.0, 22.0],
        ...     }
        ... )
        >>> expr = ichimoku(pl.col("high"), pl.col("low"), window_tenkan=2, window_kijun=3, window_senkou=4)
        >>> frame.with_columns(kijun=expr.over("ticker").struct.field("kijun").round(4))["kijun"].to_list()
        [None, None, 10.0, 11.0, 12.0, None, None, 20.0, 21.0, 22.0]

        A ``null`` (any line whose window touches it is ``null``) and a ``NaN`` (which propagates) make it visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 12.0, None, 13.0, float("nan"), 12.0, 15.0],
        ...         "low": [8.0, 9.0, 10.0, 11.0, 12.0, 10.0, 12.0],
        ...     }
        ... )
        >>> expr = ichimoku(pl.col("high"), pl.col("low"), window_tenkan=2, window_kijun=3, window_senkou=4)
        >>> frame.select(tenkan=expr.struct.field("tenkan").round(4))["tenkan"].to_list()
        [None, 10.0, None, None, nan, nan, 12.5]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    validate_window(window_tenkan, name="window_tenkan")
    validate_window(window_kijun, name="window_kijun")
    validate_window(window_senkou, name="window_senkou")
    # 3-way chain: validate_window_order in the hub is pairwise by design, so this ordering guard stays inline.
    if not window_tenkan <= window_kijun <= window_senkou:
        raise ValueError(
            f"windows must be ordered window_tenkan <= window_kijun <= window_senkou, "
            f"got window_tenkan={window_tenkan}, window_kijun={window_kijun}, window_senkou={window_senkou}"
        )
    tenkan = midprice(high, low, window_tenkan)
    kijun = midprice(high, low, window_kijun)
    return (
        pl.struct(
            tenkan=tenkan,
            kijun=kijun,
            senkou_a=(tenkan + kijun) / 2.0,
            senkou_b=midprice(high, low, window_senkou),
        )
    ).name.keep()


def keltner_channels(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    *,
    window: int,
    window_atr: int,
    multiplier: float = 2.0,
) -> pl.Expr:
    r"""
    Keltner Channels — an EMA midline with bands an ATR multiple away (the modern Linda Raschke form).

    Chester Keltner's channel in its modern form (popularized by Linda Raschke): a center band that is the :func:`ema`
    of ``close``, with upper and lower bands set ``multiplier`` average true ranges (:func:`atr`) away. Unlike Bollinger
    Bands, whose width tracks the standard deviation, Keltner's width tracks the ATR -- it breathes with the trading
    range rather than the dispersion:

    .. math::

        \mathrm{middle}_t &= \mathrm{EMA}(\mathrm{close}, n)_t, \\
        \mathrm{upper}_t &= \mathrm{middle}_t + m \cdot \mathrm{ATR}(n_a)_t, \\
        \mathrm{lower}_t &= \mathrm{middle}_t - m \cdot \mathrm{ATR}(n_a)_t,

    where :math:`n` is ``window``, :math:`n_a` is ``window_atr``, and :math:`m` is ``multiplier``.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        window: Number of observations in the EMA midline window (canonically ``20``). Must be ``>= 1``.
        window_atr: Number of observations in the ATR window (canonically ``10``). Must be ``>= 1``.
        multiplier: Band half-width as a multiple of the ATR (canonically ``2.0``). Must be a finite number ``> 0``.

    Returns:
        A struct ``pl.Expr`` with three ``Float64`` fields, the same length as the inputs:

        - ``lower`` — the lower band, ``middle - multiplier * atr``.
        - ``middle`` — the center band, the :func:`ema` of ``close``.
        - ``upper`` — the upper band, ``middle + multiplier * atr``.

        Read one band with ``.struct.field("middle")`` (etc.) or split all three into columns with ``.struct.unnest()``.
        Each band is ``null`` through its own warm-up: the midline's first ``window - 1`` rows, the outer bands' first
        ``max(window, window_atr) - 1`` rows (they also need the ATR).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``, ``window_atr < 1``, or ``multiplier`` is not a finite number ``> 0`` (i.e.
            ``<= 0``, ``NaN``, or ``±inf``).

    Note:
        **Precision**

        Agrees with its independent reference oracle (a composition of the :func:`ema` and :func:`atr` references) to
        ten significant figures (a ``1e-10`` band); the documentation's *Correctness* page gives the method.

        **Inputs**

        ``high``, ``low``, and ``close`` must share a length and alignment (the same row index is one bar). The original
        high-low band variant (Keltner's 1960 form) is not provided; compose it from :func:`ema` and the bar range if
        ever needed.

        **Edge-case behavior**

        - **Null** — a leading ``null`` run stays ``null`` until the first non-null seed; an interior ``null`` yields
          ``null`` at that position while the recursion continues across the gap — inherited from the recursive
          :func:`ema` (midline) and :func:`atr` (band width) legs, exactly as documented for each.
        - **NaN** — a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null position
          — inherited from the same :func:`ema` and :func:`atr` legs, exactly as documented for each.
        - **Degenerate denominator** — a constant ``high == low == close`` run has zero ATR, so the half-width vanishes
          and all three bands collapse onto the EMA.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`ema`: The midline.
        - :func:`atr`: The basis of the band half-width.
        - :func:`bollinger_bands`: The same idea with standard-deviation width instead of ATR.

    References:
        - Keltner, C. W. (1960). *How to Make Money in Commodities*.
        - https://en.wikipedia.org/wiki/Keltner_channel

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import keltner_channels
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [3.0, 4.0, 5.0, 6.0],
        ...         "low": [1.0, 2.0, 3.0, 4.0],
        ...         "close": [2.0, 3.0, 4.0, 5.0],
        ...     }
        ... )
        >>> expr = keltner_channels(pl.col("high"), pl.col("low"), pl.col("close"), window=2, window_atr=2)
        >>> frame.select(middle=expr.struct.field("middle").round(4))["middle"].to_list()
        [None, 2.5, 3.5, 4.5]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker's bands warm up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 4 + ["NVDA"] * 4,
        ...         "high": [3.0, 4.0, 5.0, 6.0] + [13.0, 14.0, 15.0, 16.0],
        ...         "low": [1.0, 2.0, 3.0, 4.0] + [11.0, 12.0, 13.0, 14.0],
        ...         "close": [2.0, 3.0, 4.0, 5.0] + [12.0, 13.0, 14.0, 15.0],
        ...     }
        ... )
        >>> expr = keltner_channels(pl.col("high"), pl.col("low"), pl.col("close"), window=2, window_atr=2)
        >>> frame.with_columns(middle=expr.over("ticker").struct.field("middle").round(4))["middle"].to_list()
        [None, 2.5, 3.5, 4.5, None, 12.5, 13.5, 14.5]

        A ``null`` (yields ``null`` at that row) and a ``NaN`` (which propagates) in ``close`` flow through the midline:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ...         "low": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ...         "close": [2.0, 3.0, None, 5.0, float("nan"), 7.0, 8.0],
        ...     }
        ... )
        >>> expr = keltner_channels(pl.col("high"), pl.col("low"), pl.col("close"), window=2, window_atr=2)
        >>> frame.select(middle=expr.struct.field("middle").round(4))["middle"].to_list()
        [None, 2.5, None, 4.6429, nan, nan, nan]

        **Degenerate denominator** — a flat series has zero ATR, so all three bands collapse onto the EMA of the close:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [4.0, 4.0, 4.0, 4.0],
        ...         "low": [4.0, 4.0, 4.0, 4.0],
        ...         "close": [4.0, 4.0, 4.0, 4.0],
        ...     }
        ... )
        >>> expr = keltner_channels(pl.col("high"), pl.col("low"), pl.col("close"), window=2, window_atr=2)
        >>> frame.select(lower=expr.struct.field("lower"))["lower"].to_list()
        [None, 4.0, 4.0, 4.0]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    validate_window(window)
    validate_window(window_atr, name="window_atr")
    validate_positive(multiplier, "multiplier")
    # Compose the EMA midline and ATR half-width; null/NaN propagate per band through each leg.
    middle = ema(close, window)
    half_width = multiplier * atr(high, low, close, window_atr)
    return (pl.struct(lower=middle - half_width, middle=middle, upper=middle + half_width)).name.keep()


def midpoint(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Midpoint over a window — the mean of the highest and lowest values in the trailing window.

    The center of the window's range: halfway between its rolling maximum and rolling minimum. Unlike a moving average
    it ignores everything between the extremes, tracking only the band the series has traversed:

    .. math::

        \mathrm{MIDPOINT}_t = \frac{\max(x_{t-n+1 \ldots t}) + \min(x_{t-n+1 \ldots t})}{2}, \qquad n = \text{window}.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        The window midpoint for each row, the same length as the input. The first ``window - 1`` values are ``null``
        (warm-up): the window must hold ``window`` non-null values before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Edge-case behavior**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **window == 1** — the max and min are the single value, so the midpoint reproduces the input.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`midprice`: The same midpoint taken across a bar's high and low instead of one series.
        - :func:`sma`: The moving mean of the window, which uses every value rather than only the extremes.
        - :func:`donchian_channels`: The high-low band system built from the same rolling extremes.

    References:
        - No canonical external source; the indicator is defined by the formula above.

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import midpoint
        >>>
        >>> frame = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        >>> frame.select(midpoint=midpoint(pl.col("x"), window=3).round(4))["midpoint"].to_list()
        [None, None, 2.0, 3.0, 4.0, 5.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 3 + ["NVDA"] * 3,
        ...         "x": [1.0, 2.0, 3.0] + [10.0, 20.0, 30.0],
        ...     }
        ... )
        >>> frame.with_columns(midpoint=midpoint(pl.col("x"), window=2).over("ticker").round(4))["midpoint"].to_list()
        [None, 1.5, 2.5, None, 15.0, 25.0]

        A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make the handling visible:

        >>> frame = pl.DataFrame({"x": [1.0, None, 3.0, float("nan"), 5.0, 6.0]})
        >>> frame.select(midpoint=midpoint(pl.col("x"), window=2).round(4))["midpoint"].to_list()
        [None, None, None, nan, nan, 5.5]

        **window == 1** — a single-observation window makes the max and min the value itself, so the midpoint reproduces
        the input with no warm-up:

        >>> frame = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        >>> frame.select(midpoint=midpoint(pl.col("x"), window=1))["midpoint"].to_list()
        [1.0, 2.0, 3.0]
    """
    expr = float64_expr(expr)
    validate_window(window)
    # Center of the rolling range; rolling_max/min share min_samples=window, so null/NaN propagate per window.
    return ((expr.rolling_max(window) + expr.rolling_min(window)) / 2.0).name.keep()


def midprice(
    high: pl.Expr,
    low: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Midprice over a window — the mean of the highest high and the lowest low in the trailing window.

    The center of the price range a bar series has covered: halfway between the rolling maximum of ``high`` and the
    rolling minimum of ``low``. It is the two-input analog of :func:`midpoint`, reading a bar's extremes rather than
    a single series:

    .. math::

        \mathrm{MIDPRICE}_t = \frac{\max(\mathrm{high}_{t-n+1 \ldots t}) + \min(\mathrm{low}_{t-n+1 \ldots t})}{2},
            \qquad n = \text{window}.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        The window midprice for each row, the same length as the inputs. The first ``window - 1`` values are ``null``
        (warm-up): the window must hold ``window`` non-null values before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Inputs**

        ``high`` and ``low`` must share a length and alignment (the same row index is one bar).

        **Edge-case behavior**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values) —
          in either ``high`` or ``low``.
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **window == 1** — the extremes are the bar's own ``high`` and ``low``, so the midprice reduces to the per-bar
          :func:`price_median`.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`midpoint`: The same midpoint over a single series instead of a bar's high and low.
        - :func:`price_median`: The per-bar ``(high + low) / 2`` this collapses to at ``window == 1``.
        - :func:`donchian_channels`: The channel whose middle band is exactly this midprice.

    References:
        - No canonical external source; the indicator is defined by the formula above.

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import midprice
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [11.0, 12.0, 13.0, 12.5, 14.0],
        ...         "low": [9.0, 10.0, 11.0, 11.0, 12.0],
        ...     }
        ... )
        >>> frame.select(midprice=midprice(pl.col("high"), pl.col("low"), window=3).round(4))["midprice"].to_list()
        [None, None, 11.0, 11.5, 12.5]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 3 + ["NVDA"] * 3,
        ...         "high": [11.0, 12.0, 13.0] + [21.0, 22.0, 23.0],
        ...         "low": [9.0, 10.0, 11.0] + [19.0, 20.0, 21.0],
        ...     }
        ... )
        >>> expr = midprice(pl.col("high"), pl.col("low"), window=2)
        >>> frame.with_columns(midprice=expr.over("ticker").round(4))["midprice"].to_list()
        [None, 10.5, 11.5, None, 20.5, 21.5]

        A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make the handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [11.0, None, 13.0, float("nan"), 15.0],
        ...         "low": [9.0, 10.0, 11.0, 12.0, 13.0],
        ...     }
        ... )
        >>> frame.select(midprice=midprice(pl.col("high"), pl.col("low"), window=2).round(4))["midprice"].to_list()
        [None, None, None, nan, nan]

        **window == 1** — a single-bar window makes the extremes the bar's own ``high`` and ``low``, so the midprice
        reduces to the per-bar :func:`price_median` with no warm-up:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [11.0, 12.0, 13.0],
        ...         "low": [9.0, 10.0, 11.0],
        ...     }
        ... )
        >>> frame.select(midprice=midprice(pl.col("high"), pl.col("low"), window=1))["midprice"].to_list()
        [10.0, 11.0, 12.0]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    validate_window(window)
    # Center of the rolling high-low range; rolling_max/min share min_samples=window, so null/NaN propagate per window.
    return ((high.rolling_max(window) + low.rolling_min(window)) / 2.0).name.keep()
