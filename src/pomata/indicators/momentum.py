"""
Momentum indicators.
"""

import math
from typing import Final

import polars as pl

from pomata._expr import float64_expr, validate_window, validate_window_order
from pomata.indicators.moving_average import ema, rma, sma
from pomata.indicators.price_transform import price_median, price_typical

__all__ = (
    "absolute_price_oscillator",
    "aroon",
    "aroon_oscillator",
    "awesome_oscillator",
    "balance_of_power",
    "cci",
    "chande_momentum_oscillator",
    "fisher_transform",
    "macd",
    "mom",
    "percentage_price_oscillator",
    "roc",
    "rsi",
    "rsi_stochastic",
    "trix",
    "ultimate_oscillator",
    "williams_r",
)

_FISHER_CLAMP: Final = 0.999


def absolute_price_oscillator(
    expr: pl.Expr,
    *,
    window_fast: int,
    window_slow: int,
) -> pl.Expr:
    r"""
    Absolute Price Oscillator, also known as APO — the gap between a fast and a slow exponential moving average.

    A momentum oscillator built from the difference of two :func:`ema` of the close, a fast one minus a slow one. It is
    the line that underlies :func:`macd`, expressed in price units:

    .. math::

        \mathrm{APO}_t = \mathrm{EMA}(\mathrm{close}, n_{\mathrm{fast}})_t
            - \mathrm{EMA}(\mathrm{close}, n_{\mathrm{slow}})_t.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window_fast: Span of the fast EMA (canonically ``12``). Must be ``>= 1``.
        window_slow: Span of the slow EMA (canonically ``26``). Must be ``>= 1`` and ``>= window_fast``.

    Returns:
        The oscillator for each row, the same length as the input. Values are ``null`` until both EMAs leave their
        warm-up (the first ``max(window_fast, window_slow) - 1`` rows).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window_fast < 1``, ``window_slow < 1``, or ``window_fast > window_slow`` (the fast leg must be
            the shorter one; ``window_fast == window_slow`` is allowed and gives an identically-zero oscillator).

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Moving average**

        Both legs use the exponential :func:`ema` (not a simple average), so APO is the MACD line without the signal;
        compose :func:`sma` directly for a simple-average oscillator.

        **Edge-case behavior**

        - **Null** — a leading ``null`` run stays ``null`` until the first non-null seed; an interior ``null`` yields
          ``null`` at that position while the recursion continues across the gap.
        - **NaN** — a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null
          position.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`percentage_price_oscillator`: The same gap expressed as a percentage of the slow EMA.
        - :func:`macd`: The oscillator this line underlies, adding a signal and histogram.
        - :func:`ema`: The exponential moving average each leg is built from.

    References:
        - Appel, G. (2005). *Technical Analysis: Power Tools for Active Investors*. FT Press.

    Examples:
        Basic usage on a single price series:

        >>> import polars as pl
        >>> from pomata.indicators import absolute_price_oscillator
        >>>
        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0]})
        >>> expr = absolute_price_oscillator(pl.col("close"), window_fast=2, window_slow=3)
        >>> frame.select(absolute_price_oscillator=expr.round(4))["absolute_price_oscillator"].to_list()
        [None, None, 0.5, 0.1667, 0.3889, 0.463, 0.1543, 0.3848]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker's EMAs warm up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 4 + ["NVDA"] * 4,
        ...         "close": [10.0, 11.0, 12.0, 11.0] + [20.0, 22.0, 24.0, 22.0],
        ...     }
        ... )
        >>> expr = absolute_price_oscillator(pl.col("close"), window_fast=2, window_slow=3)
        >>> frame.with_columns(absolute_price_oscillator=expr.over("ticker").round(4))[
        ...     "absolute_price_oscillator"
        ... ].to_list()
        [None, None, 0.5, 0.1667, None, None, 1.0, 0.3333]

        A ``null`` (which the recursive EMA bridges) and a ``NaN`` (which latches) make the handling visible:

        >>> frame = pl.DataFrame({"close": [10.0, 11.0, None, 13.0, float("nan"), 15.0]})
        >>> expr = absolute_price_oscillator(pl.col("close"), window_fast=2, window_slow=3)
        >>> frame.select(absolute_price_oscillator=expr.round(4))["absolute_price_oscillator"].to_list()
        [None, None, None, 1.3095, nan, nan]
    """
    expr = float64_expr(expr)
    validate_window(window_fast, name="window_fast")
    validate_window(window_slow, name="window_slow")
    validate_window_order(window_fast, window_slow)
    # The MACD line in price units: fast EMA minus slow EMA (both exponential, not simple).
    return (ema(expr, window_fast) - ema(expr, window_slow)).name.keep()


def aroon(
    high: pl.Expr,
    low: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Aroon (up and down) — two lines measuring how recently the window's high and low each occurred, in [0, 100].

    Tushar Chande's trend indicator (1995): each line measures how recently the window's extreme occurred, reported as a
    percentage of the window so it sits in ``[0, 100]`` (``100`` when the extreme is the current bar, ``0`` when it is
    the oldest bar in the look-back). With a look-back of ``window + 1`` bars and :math:`n = \text{window}`:

    .. math::

        \mathrm{up}_t &= 100 \cdot \frac{n - (\text{bars since highest high})}{n}, \\
        \mathrm{down}_t &= 100 \cdot \frac{n - (\text{bars since lowest low})}{n}.

    A rising Aroon Up with a falling Aroon Down signals an uptrend (recent highs, stale lows), and vice versa. On ties
    the most recent extreme is used. The lines depend only on the *positions* of the extremes, so they are invariant
    under any positive rescaling of ``high`` and ``low``.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        window: Look-back length; the extreme is sought over the last ``window + 1`` bars. Must be ``>= 1``.

    Returns:
        A struct ``pl.Expr`` with two ``Float64`` fields, the same length as the inputs:

        - ``up`` — the Aroon Up line, in ``[0, 100]``.
        - ``down`` — the Aroon Down line, in ``[0, 100]``.

        Both are ``null`` for the first ``window`` rows (warm-up: a full ``window + 1``-bar look-back is needed). Access
        the fields with ``.struct.field("up")`` / ``"down"`` or ``.struct.unnest()``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Tie-break and seeding**

        When the extreme is attained more than once in the look-back, the most recent occurrence is used, so the line
        reads higher.

        **Edge-case behavior**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there (it is not treated as an extreme).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`aroon_oscillator`: The difference ``up - down`` as a single line.
        - :func:`donchian_channels`: The rolling high/low extremes Aroon locates in time.
        - :func:`williams_r`: Another windowed high-low range oscillator.

    References:
        - Chande, T. S. (1995). "The Time Price Oscillator." *Technical Analysis of Stocks & Commodities*, 13(9),
          369-374.
        - https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/aroon

    Examples:
        Basic usage on high-low bars:

        >>> import polars as pl
        >>> from pomata.indicators import aroon
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0],
        ...         "low": [9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0],
        ...     }
        ... )
        >>> expr = aroon(pl.col("high"), pl.col("low"), window=3)
        >>> frame.select(up=expr.struct.field("up").round(4))["up"].to_list()
        [None, None, None, 66.6667, 100.0, 66.6667, 100.0, 66.6667]
        >>> frame.select(down=expr.struct.field("down").round(4))["down"].to_list()
        [None, None, None, 0.0, 66.6667, 33.3333, 0.0, 33.3333]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker's channel warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 5 + ["NVDA"] * 5,
        ...         "high": [10.0, 11.0, 12.0, 11.0, 13.0] + [20.0, 22.0, 24.0, 22.0, 26.0],
        ...         "low": [9.0, 10.0, 11.0, 10.0, 12.0] + [19.0, 21.0, 23.0, 21.0, 25.0],
        ...     }
        ... )
        >>> expr = aroon(pl.col("high"), pl.col("low"), window=3)
        >>> frame.with_columns(up=expr.over("ticker").struct.field("up").round(4))["up"].to_list()
        [None, None, None, 66.6667, 100.0, None, None, None, 66.6667, 100.0]

        A ``null`` (which nulls the affected line) and a ``NaN`` (which propagates) in ``high`` make the handling
        visible on the ``up`` line:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 13.0, None, 15.0, 16.0, 17.0, 18.0, float("nan"), 20.0, 21.0],
        ...         "low": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        ...     }
        ... )
        >>> expr = aroon(pl.col("high"), pl.col("low"), window=3)
        >>> frame.select(up=expr.struct.field("up").round(4))["up"].to_list()
        [None, None, None, 100.0, None, None, None, None, 100.0, nan, nan, nan]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    validate_window(window)
    span = window + 1
    rolling_high = high.rolling_max(span)
    rolling_low = low.rolling_min(span)
    # Polars has no rolling arg-max: the most recent bar whose value equals the window extreme is the smallest shift for
    # which the shifted value equals the rolling extreme (ties resolve to the most recent). A NaN window yields NaN.
    periods_since_high = pl.min_horizontal(
        [pl.when(high.shift(offset) == rolling_high).then(float(offset)).otherwise(None) for offset in range(span)]
    )
    periods_since_low = pl.min_horizontal(
        [pl.when(low.shift(offset) == rolling_low).then(float(offset)).otherwise(None) for offset in range(span)]
    )
    up_value = 100.0 * (window - periods_since_high) / window
    down_value = 100.0 * (window - periods_since_low) / window
    up = pl.when(rolling_high.is_nan()).then(float("nan")).otherwise(up_value)
    down = pl.when(rolling_low.is_nan()).then(float("nan")).otherwise(down_value)
    return (pl.struct(up=up, down=down)).name.keep()


def aroon_oscillator(
    high: pl.Expr,
    low: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Aroon Oscillator — Aroon Up minus Aroon Down as a single line, swinging within [-100, 100].

    The single-line form of :func:`aroon`: Aroon Up minus Aroon Down, so it swings within ``[-100, 100]`` (positive when
    highs are more recent than lows — an uptrend — and negative in a downtrend):

    .. math::

        \mathrm{AroonOsc}_t = \mathrm{up}_t - \mathrm{down}_t,

    with ``up`` and ``down`` the :func:`aroon` lines over the same ``window``.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        window: Look-back length; the extremes are sought over the last ``window + 1`` bars. Must be ``>= 1``.

    Returns:
        The oscillator for each row, the same length as the inputs, in ``[-100, 100]``. The first ``window`` rows are
        ``null`` (warm-up), inherited from :func:`aroon`.

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
        - **window == 1** — the look-back collapses to the last two bars, so each Aroon line is ``0`` or ``100`` and the
          oscillator takes only ``-100``, ``0``, or ``+100``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`aroon`: The two-line indicator this collapses into one.
        - :func:`donchian_channels`: The rolling high/low extremes the lines are built from.
        - :func:`williams_r`: Another windowed high-low range oscillator.

    References:
        - Chande, T. S. (1995). "The Time Price Oscillator." *Technical Analysis of Stocks & Commodities*, 13(9),
          369-374.
        - https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/aroon

    Examples:
        Basic usage on high-low bars:

        >>> import polars as pl
        >>> from pomata.indicators import aroon_oscillator
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0],
        ...         "low": [9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0],
        ...     }
        ... )
        >>> expr = aroon_oscillator(pl.col("high"), pl.col("low"), window=3)
        >>> frame.select(aroon_oscillator=expr.round(4))["aroon_oscillator"].to_list()
        [None, None, None, 66.6667, 33.3333, 33.3333, 100.0, 33.3333]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 5 + ["NVDA"] * 5,
        ...         "high": [10.0, 11.0, 12.0, 11.0, 13.0] + [20.0, 22.0, 24.0, 22.0, 26.0],
        ...         "low": [9.0, 10.0, 11.0, 10.0, 12.0] + [19.0, 21.0, 23.0, 21.0, 25.0],
        ...     }
        ... )
        >>> expr = aroon_oscillator(pl.col("high"), pl.col("low"), window=3)
        >>> frame.with_columns(aroon_oscillator=expr.over("ticker").round(4))["aroon_oscillator"].to_list()
        [None, None, None, 66.6667, 33.3333, None, None, None, 66.6667, 33.3333]

        A ``null`` (which nulls the oscillator) and a ``NaN`` (which propagates) in ``high`` make the handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 13.0, None, 15.0, 16.0, 17.0, 18.0, float("nan"), 20.0, 21.0],
        ...         "low": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        ...     }
        ... )
        >>> expr = aroon_oscillator(pl.col("high"), pl.col("low"), window=3)
        >>> frame.select(aroon_oscillator=expr.round(4))["aroon_oscillator"].to_list()
        [None, None, None, 100.0, None, None, None, None, 100.0, nan, nan, nan]

        **window == 1** — a one-bar look-back collapses each Aroon line to the last two bars, so a strictly alternating
        series saturates the oscillator at ``+100`` or ``-100`` from the second row on:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0],
        ...         "low": [9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0],
        ...     }
        ... )
        >>> expr = aroon_oscillator(pl.col("high"), pl.col("low"), window=1)
        >>> frame.select(aroon_oscillator=expr)["aroon_oscillator"].to_list()
        [None, 100.0, 100.0, -100.0, 100.0, -100.0, 100.0, -100.0, 100.0, -100.0]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    validate_window(window)
    bands = aroon(high, low, window)
    return (bands.struct.field("up") - bands.struct.field("down")).name.keep()


def awesome_oscillator(
    high: pl.Expr,
    low: pl.Expr,
    *,
    window_fast: int,
    window_slow: int,
) -> pl.Expr:
    r"""
    Awesome Oscillator (Bill Williams) — the gap between a fast and a slow simple average of the median price.

    Bill Williams' momentum gauge: a fast simple moving average of each bar's median price minus a slow one. It reads
    momentum off the midpoint of the bar rather than the close, crossing zero as the short-term average overtakes the
    long-term:

    .. math::

        \mathrm{AO}_t = \mathrm{SMA}(\mathrm{median}, n_f)_t - \mathrm{SMA}(\mathrm{median}, n_s)_t, \qquad
            \mathrm{median}_t = \frac{\mathrm{high}_t + \mathrm{low}_t}{2},

    where :math:`n_f` is ``window_fast`` and :math:`n_s` is ``window_slow``.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        window_fast: Window of the fast simple moving average (canonically ``5``). Must be ``>= 1``.
        window_slow: Window of the slow simple moving average (canonically ``34``). Must be ``>= 1`` and
            ``>= window_fast``.

    Returns:
        The oscillator for each row, the same length as the inputs. The first ``window_slow - 1`` values are ``null``
        (warm-up): both averages must be defined before their difference is.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window_fast < 1``, ``window_slow < 1``, or ``window_fast > window_slow`` (the fast leg must be
            the shorter one; ``window_fast == window_slow`` is allowed and gives an identically-zero oscillator).

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Inputs**

        ``high`` and ``low`` must share a length and alignment (the same row index is one bar).

        **Edge-case behavior**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window_fast`` non-null
          values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **Insufficient sample** — a slow window longer than the series never completes, so the result is ``null``.
        - **Degenerate denominator** — over a constant median run there is no spread between the two averages, so the
          oscillator is exactly ``0``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`absolute_price_oscillator`: The same fast-minus-slow shape on the close, with exponential averages.
        - :func:`macd`: The exponential oscillator with an added signal line.
        - :func:`price_median`: The bar median each average is taken over.

    References:
        - Williams, B. (1998). *New Trading Dimensions*. Wiley.

    Examples:
        Basic usage on high-low bars:

        >>> import polars as pl
        >>> from pomata.indicators import awesome_oscillator
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [2.0, 4.0, 6.0, 8.0, 10.0],
        ...         "low": [0.0, 2.0, 4.0, 6.0, 8.0],
        ...     }
        ... )
        >>> expr = awesome_oscillator(pl.col("high"), pl.col("low"), window_fast=2, window_slow=3)
        >>> frame.select(awesome_oscillator=expr.round(4))["awesome_oscillator"].to_list()
        [None, None, 1.0, 1.0, 1.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 5 + ["NVDA"] * 5,
        ...         "high": [11.0, 12.0, 13.0, 12.5, 14.0] + [21.0, 22.0, 23.0, 22.5, 24.0],
        ...         "low": [9.0, 10.0, 11.0, 11.0, 12.0] + [19.0, 20.0, 21.0, 21.0, 22.0],
        ...     }
        ... )
        >>> expr = awesome_oscillator(pl.col("high"), pl.col("low"), window_fast=2, window_slow=3)
        >>> frame.with_columns(awesome_oscillator=expr.over("ticker").round(4))["awesome_oscillator"].to_list()
        [None, None, 0.5, 0.2917, 0.125, None, None, 0.5, 0.2917, 0.125]

        A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make it visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [11.0, 12.0, 13.0, 12.5, 14.0, None, 15.0, float("nan"), 16.0, 17.0],
        ...         "low": [9.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0, 14.0, 15.0],
        ...     }
        ... )
        >>> expr = awesome_oscillator(pl.col("high"), pl.col("low"), window_fast=2, window_slow=3)
        >>> frame.select(awesome_oscillator=expr.round(4))["awesome_oscillator"].to_list()
        [None, None, 0.5, 0.2917, 0.125, None, None, None, nan, nan]

        **Insufficient sample** — a fast and slow window both collapsed to a single bar leave no warm-up to wait out, so
        a one-row series reads a well-defined ``0``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [2.0],
        ...         "low": [0.0],
        ...     }
        ... )
        >>> expr = awesome_oscillator(pl.col("high"), pl.col("low"), window_fast=1, window_slow=1)
        >>> frame.select(awesome_oscillator=expr)["awesome_oscillator"].to_list()
        [0.0]

        **Degenerate denominator** — a constant median price makes the fast and slow averages equal, so the oscillator
        is exactly ``0``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [5.0, 5.0, 5.0, 5.0, 5.0],
        ...         "low": [5.0, 5.0, 5.0, 5.0, 5.0],
        ...     }
        ... )
        >>> expr = awesome_oscillator(pl.col("high"), pl.col("low"), window_fast=2, window_slow=3)
        >>> frame.select(awesome_oscillator=expr)["awesome_oscillator"].to_list()
        [None, None, 0.0, 0.0, 0.0]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    validate_window(window_fast, name="window_fast")
    validate_window(window_slow, name="window_slow")
    validate_window_order(window_fast, window_slow)
    # Fast minus slow simple average of the bar's median price; CSE shares the single median sub-expression.
    median = price_median(high, low)
    return (sma(median, window_fast) - sma(median, window_slow)).name.keep()


def balance_of_power(
    open: pl.Expr,
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
) -> pl.Expr:
    r"""
    Balance of Power, also known as BOP — where each bar closed within its range, relative to where it opened.

    A per-bar momentum gauge: the close-minus-open move as a fraction of the bar's whole high-low range, reporting how
    decisively buyers or sellers controlled the bar (positive = buyers, negative = sellers; bounded in ``[-1, 1]`` for a
    well-formed bar whose open and close sit inside its range):

    .. math::

        \mathrm{BOP}_t = \frac{\mathrm{close}_t - \mathrm{open}_t}{\mathrm{high}_t - \mathrm{low}_t}.

    Args:
        open: Open-price series (e.g. ``pl.col("open")``).
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).

    Returns:
        The balance of power for each row, the same length as the inputs. There is no window and no warm-up -- every row
        is defined from row ``0``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Inputs**

        ``open``, ``high``, ``low``, and ``close`` are the canonical OHLC roles in that positional order and must share
        a length and alignment (the same row index is one bar). ``balance_of_power`` is scale-invariant: multiplying all
        four by a common factor leaves it unchanged.

        **Edge-case behavior**

        - **Null** — a ``null`` price makes that row ``null`` (``null`` takes precedence over ``NaN``).
        - **NaN** — a ``NaN`` price yields ``NaN`` for that row.
        - **Degenerate denominator** — when ``high == low`` the range is zero — the ``0 / 0`` degenerate — but the
          result is ``0`` by convention (no range, no directional power); the zero-range branch fires first, so a finite
          flat bar reads ``0`` even when ``open`` or ``close`` is ``null``, and only a ``null`` ``high`` or ``low``
          (which leaves the range itself ``null``) still yields ``null`` on a flat bar.
        - **Partitioning** — already correct on a multi-series panel: ``.over(...)`` partitions identically and is
          therefore optional here.

    See Also:
        - :func:`price_average`: Another per-bar OHLC summary, the equal-weighted mean of the four prices.
        - :func:`price_weighted_close`: A per-bar OHLC summary that leans on the close.
        - :func:`price_typical`: The per-bar high-low-close average.

    References:
        - Livshin, I. (2001). "Using the Balance of Power Indicator." *Technical Analysis of Stocks & Commodities*.
        - https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/balance-of-power-bop

    Examples:
        Basic usage on a small OHLC frame:

        >>> import polars as pl
        >>> from pomata.indicators import balance_of_power
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "open": [10.0, 11.0, 12.0, 11.0],
        ...         "high": [11.0, 13.0, 12.0, 13.0],
        ...         "low": [9.0, 10.0, 11.0, 10.0],
        ...         "close": [10.5, 12.0, 11.5, 12.0],
        ...     }
        ... )
        >>> expr = balance_of_power(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        >>> frame.select(balance_of_power=expr.round(4))["balance_of_power"].to_list()
        [0.25, 0.3333, -0.5, 0.3333]

        Balance of Power is elementwise, so ``.over`` is optional; each ticker yields the same per-bar reading:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 4 + ["NVDA"] * 4,
        ...         "open": [10.0, 11.0, 12.0, 11.0] + [20.0, 21.0, 22.0, 21.0],
        ...         "high": [11.0, 13.0, 12.0, 13.0] + [21.0, 23.0, 22.0, 23.0],
        ...         "low": [9.0, 10.0, 11.0, 10.0] + [19.0, 20.0, 21.0, 20.0],
        ...         "close": [10.5, 12.0, 11.5, 12.0] + [20.5, 22.0, 21.5, 22.0],
        ...     }
        ... )
        >>> expr = balance_of_power(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        >>> frame.with_columns(balance_of_power=expr.over("ticker").round(4))["balance_of_power"].to_list()
        [0.25, 0.3333, -0.5, 0.3333, 0.25, 0.3333, -0.5, 0.3333]

        A flat bar (``high == low``, giving ``0``), then a ``null`` and a ``NaN`` in ``close`` make the edge handling
        visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "open": [10.0, 12.0, 11.0, 12.0, 12.0],
        ...         "high": [11.0, 12.0, 13.0, 14.0, 13.0],
        ...         "low": [9.0, 12.0, 11.0, 12.0, 11.0],
        ...         "close": [10.5, 12.0, None, 13.0, float("nan")],
        ...     }
        ... )
        >>> expr = balance_of_power(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        >>> frame.select(balance_of_power=expr.round(4))["balance_of_power"].to_list()
        [0.25, 0.0, None, 0.5, nan]

        **Degenerate denominator** — a flat bar with ``high`` equal to ``low`` has an exact zero range, the bare
        ``0/0``, but returns ``0`` by convention:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "open": [10.0, 12.0],
        ...         "high": [11.0, 12.0],
        ...         "low": [9.0, 12.0],
        ...         "close": [10.5, 11.0],
        ...     }
        ... )
        >>> expr = balance_of_power(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
        >>> frame.select(balance_of_power=expr)["balance_of_power"].to_list()
        [0.25, 0.0]
    """
    open = float64_expr(open)
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    # Flat bar (range == 0) -> 0 by convention; use high - low == 0 (not high == low) since Polars treats NaN == NaN as
    # true, which would wrongly flatten a NaN bar. Otherwise (close - open) / (high - low) propagates null / NaN.
    return pl.when((high - low) == 0).then(0.0).otherwise((close - open) / (high - low)).name.keep()


def cci(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Commodity Channel Index (CCI) — the typical price's gap from its rolling mean, scaled by mean absolute deviation.

    A momentum oscillator introduced by Donald Lambert (1980) that measures how far the current typical price has
    strayed from its statistical mean, scaled by the typical price's own mean absolute deviation so the result is
    comparable across instruments and price levels.

    With the typical price :math:`\mathrm{TP} = (H + L + C) / 3` and :math:`n = \text{window}`:

    .. math::

        \mathrm{CCI}_t = \frac{\mathrm{TP}_t - \mathrm{SMA}(\mathrm{TP}, n)_t}{0.015 \cdot \mathrm{MAD}_t},
        \qquad
        \mathrm{MAD}_t = \frac{1}{n} \sum_{i=0}^{n-1} \bigl\lvert \mathrm{TP}_{t-i} - \mathrm{SMA}(\mathrm{TP}, n)_t
        \bigr\rvert.

    The denominator is the rolling **mean absolute deviation about the rolling mean**: every observation in the window
    is measured against the *same* current :math:`\mathrm{SMA}(\mathrm{TP}, n)_t`, not against its own moving average.
    Lambert fixed the constant at :math:`0.015` so that roughly 70-80% of values fall in :math:`[-100, +100]`; the index
    is unbounded and routinely overshoots that band on strong moves. It is scale-invariant under a positive common
    rescaling of ``high``, ``low``, and ``close`` (numerator and denominator scale together) and flips sign under a
    negative one.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        The CCI for each row, the same length as the inputs. The first ``window - 1`` values are ``null`` (warm-up),
        inherited from the :func:`sma` of the typical price: the value is defined only once a full window of typical
        prices is available.

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
        - **Insufficient sample** — a window longer than the series never completes, so the result is ``null``.
        - **Degenerate denominator** — when every typical price in the window is equal there is no spread to normalize
          by, so the result is a ``0 / 0``, i.e. ``NaN``, detected exactly via the rolling extremes (its rolling maximum
          equals its rolling minimum) rather than the rounding noise a sub-ULP denominator residual would produce.
        - **window == 1** — every one-bar window is trivially flat, so every non-null result is ``NaN`` (a ``null`` row
          stays ``null``).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`price_typical`: The typical price the index is built on.
        - :func:`sma`: The simple moving average of the typical price it composes.
        - :func:`rsi`: A bounded momentum oscillator.

    References:
        - Lambert, D. R. (1980). "Commodity Channel Index: Tools for Trading Cyclic Trends." *Commodities* (now
          *Futures*) magazine.
        - https://en.wikipedia.org/wiki/Commodity_channel_index

    Examples:
        Basic usage on high-low-close bars:

        >>> import polars as pl
        >>> from pomata.indicators import cci
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [24.2, 24.3, 24.7, 25.0, 24.8, 24.5, 24.6],
        ...         "low": [23.9, 24.1, 24.3, 24.6, 24.4, 24.1, 24.2],
        ...         "close": [24.0, 24.2, 24.5, 24.8, 24.6, 24.3, 24.4],
        ...     }
        ... )
        >>> frame.select(cci=cci(pl.col("high"), pl.col("low"), pl.col("close"), window=3).round(4))["cci"].to_list()
        [None, None, 100.0, 100.0, -20.0, -100.0, -20.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 4 + ["NVDA"] * 4,
        ...         "high": [11.0, 12.0, 13.0, 12.0] + [21.0, 23.0, 22.0, 24.0],
        ...         "low": [9.0, 10.0, 11.0, 10.0] + [19.0, 21.0, 20.0, 22.0],
        ...         "close": [10.0, 11.0, 12.0, 11.0] + [20.0, 22.0, 21.0, 23.0],
        ...     }
        ... )
        >>> expr = cci(pl.col("high"), pl.col("low"), pl.col("close"), window=2)
        >>> frame.with_columns(cci=expr.over("ticker").round(4))["cci"].to_list()
        [None, 66.6667, 66.6667, -66.6667, None, 66.6667, -66.6667, 66.6667]

        A ``null`` and a ``NaN`` in ``close`` (each voiding every window that covers it) make the exact handling visible
        at a glance:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        ...         "low": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        ...         "close": [10.0, 11.0, 12.0, None, 14.0, 15.0, float("nan"), 17.0],
        ...     }
        ... )
        >>> frame.select(cci=cci(pl.col("high"), pl.col("low"), pl.col("close"), window=2).round(4))["cci"].to_list()
        [None, 66.6667, 66.6667, None, None, 66.6667, nan, nan]

        **Insufficient sample** — a single-row series with a one-bar window is trivially flat, so the result is ``NaN``
        immediately:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0],
        ...         "low": [8.0],
        ...         "close": [9.0],
        ...     }
        ... )
        >>> frame.select(cci=cci(pl.col("high"), pl.col("low"), pl.col("close"), window=1))["cci"].to_list()
        [nan]

        **Degenerate denominator** — a constant series gives a zero mean deviation, the flat-window ``0/0`` degenerate,
        so the result is ``NaN``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 10.0, 10.0, 10.0, 10.0],
        ...         "low": [8.0, 8.0, 8.0, 8.0, 8.0],
        ...         "close": [9.0, 9.0, 9.0, 9.0, 9.0],
        ...     }
        ... )
        >>> frame.select(cci=cci(pl.col("high"), pl.col("low"), pl.col("close"), window=3))["cci"].to_list()
        [None, None, nan, nan, nan]

        **window == 1** — a one-bar window makes every window trivially flat, the ``0/0`` boundary, so every result is
        ``NaN``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 12.0, 11.0],
        ...         "low": [8.0, 9.0, 9.0],
        ...         "close": [9.0, 11.0, 10.0],
        ...     }
        ... )
        >>> frame.select(cci=cci(pl.col("high"), pl.col("low"), pl.col("close"), window=1))["cci"].to_list()
        [nan, nan, nan]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    validate_window(window)
    typical_price = price_typical(high, low, close)
    typical_mean = sma(typical_price, window)
    # MAD is about the *current* window mean SMA(TP)_t (same value for every term), not each point's own mean — hence
    # the explicit shifted-term sum, not (typical_price - typical_mean).abs().rolling_mean(window), which would
    # measure each point against its own mean (the standard but incorrect shortcut).
    deviation_total = (typical_price - typical_mean).abs()
    for offset in range(1, window):
        deviation_total = deviation_total + (typical_price.shift(offset) - typical_mean).abs()
    mean_deviation = deviation_total / window
    raw = (typical_price - typical_mean) / (0.015 * mean_deviation)
    # A flat window (every typical price equal) is the 0/0 degenerate: detect it exactly via the rolling extremes, so a
    # sub-ULP residual in the rolling-mean denominator cannot fake a finite reading, and return NaN as documented.
    is_flat = typical_price.rolling_max(window) == typical_price.rolling_min(window)
    return pl.when(is_flat).then(float("nan")).otherwise(raw).name.keep()


def chande_momentum_oscillator(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Chande Momentum Oscillator, also known as CMO.

    A momentum oscillator introduced by Tushar Chande (1994) that measures the net of up-moves versus down-moves over a
    window as a fraction of the total movement, so it swings within ``[-100, +100]`` (positive when gains dominate,
    negative when losses do). With one-step changes :math:`\Delta_t = x_t - x_{t-1}` split into gains
    :math:`U_t = \max(\Delta_t, 0)` and losses :math:`D_t = \max(-\Delta_t, 0)`, and :math:`n = \text{window}`:

    .. math::

        \mathrm{CMO}_t = 100 \cdot \frac{\sum_{i=0}^{n-1} U_{t-i} - \sum_{i=0}^{n-1} D_{t-i}}
        {\sum_{i=0}^{n-1} U_{t-i} + \sum_{i=0}^{n-1} D_{t-i}}.

    Unlike the RSI, which Wilder-smooths the gains and losses, the CMO sums them over a fixed window and keeps both
    signs in the numerator, so it crosses zero rather than oscillating around 50. It is scale-invariant under a
    positive common rescaling of ``expr`` (gains and losses scale together) and flips sign under a negative one.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of one-step changes summed in the window. Must be ``>= 1``.

    Returns:
        The oscillator (in percent) for each row, the same length as the input. The first ``window`` rows are ``null``
        (warm-up): row ``0`` has no change, and the rolling sums need ``window`` non-null changes before emitting.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it. For this oscillator the limit is concrete: the windowed gain / loss sums ride Polars'
        incremental sliding kernel, so a window whose scale sits tens of orders of magnitude below a value that has
        already slid out can inherit a stale residue; the clamp keeps the output inside ``[-100, 100]``, and no real
        market series builds that spread.

        **Documented TA-Lib divergence**

        TA-Lib Wilder-smooths the gains and losses, so its CMO equals ``2 * RSI - 100``; pomata sums them over a fixed
        window, Chande's original 1994 construction, so the two never agree even at steady state and the differential
        tier holds the CMO out as a documented divergence.

        **Edge-case behavior**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **Degenerate denominator** — an exactly-flat window has every change zero, detected via the residual-free
          rolling maximum of ``|change|``, so the result is a ``0 / 0``, i.e. ``NaN``.
        - **Stability** — a near-flat window (tiny changes after a much larger one has slid out of the streaming sums)
          is not silenced: its quotient is clipped to ``[-100, +100]``, so it stays in range but, past a sane dynamic
          range, degrades in precision (see the precision note above).
        - **window == 1** — the rolling sums collapse to the single change, so each row reports ``+100`` on an up move,
          ``-100`` on a down move, and ``NaN`` on no move.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`rsi`: The Wilder-smoothed sibling, bounded in ``[0, 100]``.
        - :func:`roc`: A simpler single-horizon momentum measure.
        - :func:`mom`: The absolute-difference momentum sibling.

    References:
        - Chande, T. S. & Kroll, S. (1994). *The New Technical Trader*. Wiley.

    Examples:
        Basic usage on a single price series:

        >>> import polars as pl
        >>> from pomata.indicators import chande_momentum_oscillator
        >>>
        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0]})
        >>> expr = chande_momentum_oscillator(pl.col("close"), window=3)
        >>> frame.select(chande_momentum_oscillator=expr.round(4))["chande_momentum_oscillator"].to_list()
        [None, None, None, 33.3333, 50.0, 50.0, 50.0, 50.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 5 + ["NVDA"] * 5,
        ...         "close": [10.0, 11.0, 12.0, 11.0, 13.0] + [20.0, 19.0, 21.0, 22.0, 20.0],
        ...     }
        ... )
        >>> expr = chande_momentum_oscillator(pl.col("close"), window=3)
        >>> frame.with_columns(chande_momentum_oscillator=expr.over("ticker").round(4))[
        ...     "chande_momentum_oscillator"
        ... ].to_list()
        [None, None, None, 33.3333, 50.0, None, None, None, 50.0, 20.0]

        A ``null`` (any window it touches yields ``null``) and a ``NaN`` (which propagates) make the handling visible:

        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, None, 14.0, float("nan"), 16.0, 17.0]})
        >>> expr = chande_momentum_oscillator(pl.col("close"), window=3)
        >>> frame.select(chande_momentum_oscillator=expr.round(4))["chande_momentum_oscillator"].to_list()
        [None, None, None, None, None, None, None, nan]

        **Degenerate denominator** — an all-flat window has every change exactly zero, the ``0/0`` degenerate, so the
        result is ``NaN``:

        >>> frame = pl.DataFrame({"close": [10.0, 10.0, 10.0, 10.0, 10.0]})
        >>> expr = chande_momentum_oscillator(pl.col("close"), window=3)
        >>> frame.select(chande_momentum_oscillator=expr)["chande_momentum_oscillator"].to_list()
        [None, None, None, nan, nan]

        **window == 1** — a one-bar window collapses the rolling gain/loss sums to the raw move direction, so each row
        reports ``+100`` on an up move and ``-100`` on a down move:

        >>> frame = pl.DataFrame({"close": [1.0, 3.0, 2.0, 5.0]})
        >>> expr = chande_momentum_oscillator(pl.col("close"), window=1)
        >>> frame.select(chande_momentum_oscillator=expr)["chande_momentum_oscillator"].to_list()
        [None, 100.0, -100.0, 100.0]
    """
    expr = float64_expr(expr)
    validate_window(window)
    delta = expr.diff()
    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)
    sum_gain = gain.rolling_sum(window)
    sum_loss = loss.rolling_sum(window)
    # An exactly-flat window (every change zero) is the 0/0 degenerate: detect it via the residual-free rolling maximum
    # of |change| -- zero only when every change is exactly zero -- and return NaN, matching the oracle. The clip bounds
    # the near-flat residual overshoot to [-100, 100]: beyond a sane dynamic range the streaming quotient degrades but
    # stays in range (see the documentation's Correctness page); the bound holds without a non-decaying floor
    # that NaN-s legitimate data.
    raw = (100.0 * (sum_gain - sum_loss) / (sum_gain + sum_loss)).clip(-100.0, 100.0)
    is_flat = delta.abs().rolling_max(window) == 0
    return pl.when(is_flat).then(float("nan")).otherwise(raw).name.keep()


def _fisher_kernel(
    series: pl.Series,
) -> pl.Series:
    """
    Sequential Fisher-transform recurrence over one Series of normalized channel positions (the pure-Python kernel).

    The input ``series`` is ``y = 2 * (price - min) / (max - min) - 1``, the median price's position in its rolling
    channel mapped to ``[-1, 1]`` (``None`` on the warm-up and on any window touching a ``null``; ``NaN`` on a window
    touching a ``NaN`` or on a flat ``max == min`` window). Both running states seed at ``0``; a ``None`` row is skipped
    and a ``NaN`` row yields ``NaN``, the state bridging either gap so a transient gap does not latch.
    """
    positions: list[float | None] = series.to_list()
    result: list[float | None] = [None] * len(positions)
    smoothed = 0.0
    fisher = 0.0
    for index, position in enumerate(positions):
        if position is None:
            continue
        if math.isnan(position):
            result[index] = math.nan
            continue
        smoothed = 0.33 * position + 0.67 * smoothed
        smoothed = -_FISHER_CLAMP if smoothed < -_FISHER_CLAMP else min(smoothed, _FISHER_CLAMP)
        fisher = 0.5 * math.log((1.0 + smoothed) / (1.0 - smoothed)) + 0.5 * fisher
        result[index] = fisher
    return pl.Series(result, dtype=pl.Float64)


def _fisher_signal_kernel(
    series: pl.Series,
) -> pl.Series:
    """
    Run the Fisher recurrence once and emit the ``{fisher, signal}`` struct, ``signal`` being ``fisher`` lagged one bar.

    Producing both lines from a single pass avoids re-running the sequential recurrence for the lagged signal line.
    """
    fisher = _fisher_kernel(series)
    return pl.DataFrame({"fisher": fisher, "signal": fisher.shift(1)}).to_struct()


def fisher_transform(
    high: pl.Expr,
    low: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Fisher Transform — the median price's rolling channel position mapped through the inverse hyperbolic tangent.

    Introduced by John F. Ehlers (2002): it presses the price into a sharply-peaked, near-Gaussian oscillator so that
    turning points stand out as decisive extremes rather than the rounded humps of a raw price channel. The median
    price ``(high + low) / 2`` is first placed in its rolling channel and mapped to ``[-1, 1]``, smoothed with Ehlers'
    fixed ``0.33 / 0.67`` recursion, held just inside ``\pm 1`` by a clamp, then run through the Fisher transform --
    the inverse hyperbolic tangent, which stretches the tails toward ``\pm\infty``:

    .. math::

        x_t &= 0.33 \left( 2 \, \frac{p_t - \min_w p}{\max_w p - \min_w p} - 1 \right) + 0.67 \, x_{t-1},
        \qquad x_t \leftarrow \operatorname{clamp}(x_t, -0.999, 0.999), \\
        \mathrm{Fisher}_t &= 0.5 \, \ln\!\frac{1 + x_t}{1 - x_t} + 0.5 \, \mathrm{Fisher}_{t-1},

    where :math:`p_t = (\mathrm{high}_t + \mathrm{low}_t)/2` is the median price and :math:`\min_w, \max_w` are its
    rolling minimum and maximum over ``window`` bars. The ``signal`` line is the Fisher value lagged one bar, the
    trigger a crossover system reads against the Fisher line.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        window: Number of observations in the moving window (canonically ``10``). Must be ``>= 1``.

    Returns:
        A struct ``pl.Expr`` with two ``Float64`` fields, the same length as the inputs:

        - ``fisher`` — the Fisher Transform.
        - ``signal`` — ``fisher`` lagged one bar (``null`` for one further row).

        The first ``window - 1`` rows are ``null`` (the channel's warm-up). Read a field with
        ``.struct.field("fisher")`` or split both with ``.struct.unnest()``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        It is invariant under a positive affine rescaling of the inputs: the channel normalization
        ``(p - \min)/(\max - \min)`` cancels any common scale, so the transform depends only on the price's *shape*, not
        its level or units.

        **Clamp convention**

        The smoothed position is held to a symmetric ``[-0.999, 0.999]`` -- a monotone clamp at the threshold, keeping
        the argument strictly inside the log's domain. Ehlers' original snaps any value past ``\pm 0.99`` straight to
        ``\pm 0.999``; pomata uses the monotone form (the modern convention), which agrees everywhere except on the thin
        ``(0.99, 0.999)`` band the original discontinuously lifts (at ``0.999`` both map to ``0.999``).

        **Seeding**

        Both recursions start from ``0`` (the bar before the first defined row contributes ``0`` to each), matching
        Ehlers' zero-initialized series; the smoothing then washes the seed out geometrically.

        **Edge-case behavior**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values) —
          the recursion bridges those rows and resumes once the window clears.
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there, likewise bridged.
        - **Insufficient sample** — a window longer than the series never completes, so the result is ``null``.
        - **Degenerate denominator** — when ``max == min`` over the window the channel has no range to normalize by, so
          the result is a ``0 / 0``, i.e. ``NaN``.
        - **window == 1** — the channel spans a single bar, so ``max == min`` makes it flat by construction and
          ``fisher`` is ``NaN`` from the first row.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`williams_r`: The raw channel position the transform sharpens.
        - :func:`rsi_stochastic`: Another channel-normalized momentum oscillator, bounded rather than tail-stretched.
        - :func:`stochastic_fast`: The %K channel position, the same normalization before the transform.

    References:
        - Ehlers, J. F. (2002). "Using the Fisher Transform." *Technical Analysis of Stocks & Commodities*, 20(11).

    Examples:
        Basic usage on high-low bars:

        >>> import polars as pl
        >>> from pomata.indicators import fisher_transform
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 13.0, 14.0, 15.0],
        ...         "low": [9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 12.0, 13.0, 14.0],
        ...     }
        ... )
        >>> expr = fisher_transform(pl.col("high"), pl.col("low"), window=3)
        >>> frame.select(fisher=expr.struct.field("fisher").round(4))["fisher"].to_list()
        [None, None, 0.3428, 0.7914, 1.2615, 0.7701, 0.1432, 0.2444, 0.6002, 1.038]
        >>> frame.select(signal=expr.struct.field("signal").round(4))["signal"].to_list()
        [None, None, None, 0.3428, 0.7914, 1.2615, 0.7701, 0.1432, 0.2444, 0.6002]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker's channel warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 5 + ["NVDA"] * 5,
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0] + [20.0, 21.0, 22.0, 21.5, 23.0],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0] + [19.0, 20.0, 21.0, 20.5, 22.0],
        ...     }
        ... )
        >>> expr = fisher_transform(pl.col("high"), pl.col("low"), window=3)
        >>> frame.with_columns(fisher=expr.over("ticker").struct.field("fisher").round(4))["fisher"].to_list()
        [None, None, 0.3428, 0.3962, 0.7187, None, None, 0.3428, 0.3962, 0.7187]

        A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make it visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, None, 13.0, float("nan"), 15.0],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 14.0],
        ...     }
        ... )
        >>> expr = fisher_transform(pl.col("high"), pl.col("low"), window=3)
        >>> frame.select(fisher=expr.struct.field("fisher").round(4))["fisher"].to_list()
        [None, None, 0.3428, None, None, None, nan]

        **Insufficient sample** — a one-bar window is flat by construction (``max == min``), so ``fisher`` is ``NaN``
        from the first row:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [11.0],
        ...         "low": [9.0],
        ...     }
        ... )
        >>> expr = fisher_transform(pl.col("high"), pl.col("low"), window=1)
        >>> frame.select(fisher=expr.struct.field("fisher"))["fisher"].to_list()
        [nan]

        **Degenerate denominator** — a constant series has ``max == min`` over every window, so the channel
        normalization is the ``0/0`` boundary, which bridges through the recursion as ``NaN``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        ...         "low": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        ...     }
        ... )
        >>> expr = fisher_transform(pl.col("high"), pl.col("low"), window=3)
        >>> frame.select(fisher=expr.struct.field("fisher"))["fisher"].to_list()
        [None, None, nan, nan, nan, nan]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    validate_window(window)
    price = price_median(high, low)
    lowest = price.rolling_min(window)
    highest = price.rolling_max(window)
    position = 2.0 * (price - lowest) / (highest - lowest) - 1.0
    return position.map_batches(
        _fisher_signal_kernel, return_dtype=pl.Struct({"fisher": pl.Float64, "signal": pl.Float64})
    ).name.keep()


def macd(
    expr: pl.Expr,
    *,
    window_fast: int,
    window_slow: int,
    window_signal: int,
) -> pl.Expr:
    r"""
    Moving Average Convergence/Divergence (MACD) — the fast-minus-slow EMA gap, its signal EMA, and their histogram.

    Gerald Appel's trend-and-momentum oscillator (late 1970s): the gap between a fast and a slow :func:`ema` of the
    close (the MACD line), a further EMA of that gap (the signal line), and their difference (the histogram), returned
    together as one struct. With :math:`n_f`, :math:`n_s`, :math:`n_g` the fast, slow, and signal spans:

    .. math::

        \mathrm{MACD}_t &= \mathrm{EMA}(\mathrm{close}, n_f)_t - \mathrm{EMA}(\mathrm{close}, n_s)_t, \\
        \mathrm{signal}_t &= \mathrm{EMA}(\mathrm{MACD}, n_g)_t, \\
        \mathrm{histogram}_t &= \mathrm{MACD}_t - \mathrm{signal}_t.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window_fast: Span of the fast EMA (canonically ``12``). Must be ``>= 1``.
        window_slow: Span of the slow EMA (canonically ``26``). Must be ``>= 1`` and ``>= window_fast``.
        window_signal: Span of the signal EMA over the MACD line (canonically ``9``). Must be ``>= 1``.

    Returns:
        A struct ``pl.Expr`` with three ``Float64`` fields, the same length as ``expr``:

        - ``macd`` — the fast-minus-slow EMA gap, ``null`` for its ``max(window_fast, window_slow) - 1`` warm-up rows.
        - ``signal`` — the EMA of the MACD line, carrying the additional ``window_signal - 1`` warm-up rows on top.
        - ``histogram`` — ``macd`` minus ``signal``, sharing the signal line's warm-up.

        Access the fields with ``.struct.field("macd")`` / ``"signal"`` / ``"histogram"`` or ``.struct.unnest()``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window_fast < 1``, ``window_slow < 1``, ``window_signal < 1``, or ``window_fast > window_slow``
            (the fast leg must be the shorter one; ``window_fast == window_slow`` is allowed and gives an
            identically-zero MACD line, signal, and histogram).

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Scaling**

        Every field is homogeneous of degree ``1`` in ``expr`` (the EMAs and their differences all scale with the
        price), so multiplying the close by ``k`` scales all three fields by ``k``.

        **Edge-case behavior**

        - **Null** — a leading ``null`` run stays ``null`` until the first non-null seed; an interior ``null`` yields
          ``null`` at that position while the recursion continues across the gap.
        - **NaN** — a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null
          position.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`absolute_price_oscillator`: The Absolute Price Oscillator, the MACD line without the signal and
          histogram.
        - :func:`percentage_price_oscillator`: The percentage counterpart of the MACD line.
        - :func:`ema`: The exponential moving average all three lines are built from.

    References:
        - Appel, G. (2005). *Technical Analysis: Power Tools for Active Investors*. FT Press.
        - https://en.wikipedia.org/wiki/MACD

    Examples:
        Basic usage on a single price series:

        >>> import polars as pl
        >>> from pomata.indicators import macd
        >>>
        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0]})
        >>> expr = macd(pl.col("close"), window_fast=2, window_slow=3, window_signal=2)
        >>> frame.select(macd=expr.struct.field("macd").round(4))["macd"].to_list()
        [None, None, 0.5, 0.1667, 0.3889, 0.463, 0.1543, 0.3848]
        >>> frame.select(signal=expr.struct.field("signal").round(4))["signal"].to_list()
        [None, None, None, 0.3333, 0.3704, 0.4321, 0.2469, 0.3388]
        >>> frame.select(histogram=expr.struct.field("histogram").round(4))["histogram"].to_list()
        [None, None, None, -0.1667, 0.0185, 0.0309, -0.0926, 0.046]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker's EMAs warm up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 4 + ["NVDA"] * 4,
        ...         "close": [10.0, 11.0, 12.0, 11.0] + [20.0, 22.0, 24.0, 22.0],
        ...     }
        ... )
        >>> expr = macd(pl.col("close"), window_fast=2, window_slow=3, window_signal=2)
        >>> frame.with_columns(macd=expr.over("ticker").struct.field("macd").round(4))["macd"].to_list()
        [None, None, 0.5, 0.1667, None, None, 1.0, 0.3333]

        A ``null`` (which the recursive EMAs bridge) and a ``NaN`` (which latches) make the handling visible on the MACD
        line:

        >>> frame = pl.DataFrame({"close": [10.0, 11.0, None, 13.0, float("nan"), 15.0]})
        >>> expr = macd(pl.col("close"), window_fast=2, window_slow=3, window_signal=2)
        >>> frame.select(macd=expr.struct.field("macd").round(4))["macd"].to_list()
        [None, None, None, 1.3095, nan, nan]
    """
    expr = float64_expr(expr)
    validate_window(window_fast, name="window_fast")
    validate_window(window_slow, name="window_slow")
    validate_window(window_signal, name="window_signal")
    validate_window_order(window_fast, window_slow)
    macd_line = ema(expr, window_fast) - ema(expr, window_slow)
    signal = ema(macd_line, window_signal)
    histogram = macd_line - signal
    return (pl.struct(macd=macd_line, signal=signal, histogram=histogram)).name.keep()


def mom(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Momentum, also known as MOM or the rate-of-change in absolute (price-difference) form.

    The oldest and simplest momentum oscillator: the signed difference between the current observation and the
    observation ``window`` periods earlier, measuring how far price has traveled over the look-back horizon:

    .. math:: \mathrm{MOM}_t = x_t - x_{t - n}, \qquad n = \text{window}.

    A positive value means price is higher than ``window`` periods ago (upward momentum), a negative value the reverse,
    and zero a flat look-back; the magnitude is in the same units as ``expr`` and scales linearly with price, so it is
    not comparable across instruments at different price levels (use a ratio-based rate-of-change for that). The result
    is homogeneous of degree 1 in ``expr`` (``mom(k * x) == k * mom(x)``) and invariant to an additive
    constant only at the per-element level once both endpoints are shifted by the same amount.

    It is the unbounded, absolute-difference sibling of the percentage rate-of-change.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of observations to look back. Must be ``>= 1``.

    Returns:
        The momentum for each row, the same length as ``expr``. The first ``window`` values are ``null`` (warm-up),
        clamped to the series length: unlike the moving-average family, whose warm-up is ``window - 1`` rows, the value
        at row ``t`` needs the observation at row ``t - window``, which first exists at ``t == window``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Edge-case behavior**

        - **Null** — a ``null`` value makes that row ``null`` (``null`` takes precedence over ``NaN``).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there — a fixed-lag difference, not a
          recurrence, so a ``null`` or ``NaN`` contaminates only the (at most two) positions that reference it and never
          latches onto the rest of the series.
        - **Degenerate denominator** — a flat look-back leaves ``x_t == x_{t-n}``, so the difference is exactly ``0``.
        - **window == 1** — the look-back is a single bar, so ``mom`` is the one-step first difference
          ``x_t - x_{t-1}``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`roc`: The percentage-change sibling (scale-invariant).
        - :func:`rsi`: A bounded momentum oscillator.
        - :func:`chande_momentum_oscillator`: A bounded net-of-gains-and-losses momentum oscillator.

    References:
        - https://en.wikipedia.org/wiki/Momentum_%28technical_analysis%29

    Examples:
        Basic usage on a single price series:

        >>> import polars as pl
        >>> from pomata.indicators import mom
        >>>
        >>> frame = pl.DataFrame({"close": [2.0, 4.0, 6.0, 8.0, 10.0]})
        >>> frame.select(mom=mom(pl.col("close"), window=2).round(4))["mom"].to_list()
        [None, None, 4.0, 4.0, 4.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 5 + ["NVDA"] * 5,
        ...         "close": [10.0, 11.0, 12.0, 11.0, 13.0] + [20.0, 22.0, 21.0, 23.0, 22.0],
        ...     }
        ... )
        >>> frame.with_columns(mom=mom(pl.col("close"), window=2).over("ticker").round(4))["mom"].to_list()
        [None, None, 2.0, 0.0, 1.0, None, None, 1.0, 1.0, 1.0]

        A ``null`` (voiding the rows that reference it) and a ``NaN`` (which propagates) make the exact handling visible
        at a glance:

        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, None, 15.0, float("nan"), 17.0, 18.0, 19.0]})
        >>> frame.select(mom=mom(pl.col("close"), window=2).round(4))["mom"].to_list()
        [None, None, 2.0, 2.0, None, 2.0, None, 2.0, nan, 2.0]

        **Degenerate denominator** — the momentum of a constant series is exactly ``0`` once warmed up:

        >>> frame = pl.DataFrame({"close": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]})
        >>> frame.select(mom=mom(pl.col("close"), window=3))["mom"].to_list()
        [None, None, None, 0.0, 0.0, 0.0]

        **window == 1** — a one-bar window is the first difference with a single leading ``null``:

        >>> frame = pl.DataFrame({"close": [2.0, 4.0, 6.0, 8.0]})
        >>> frame.select(mom=mom(pl.col("close"), window=1))["mom"].to_list()
        [None, 2.0, 2.0, 2.0]
    """
    expr = float64_expr(expr)
    validate_window(window)
    return (expr - expr.shift(window)).name.keep()


def percentage_price_oscillator(
    expr: pl.Expr,
    *,
    window_fast: int,
    window_slow: int,
) -> pl.Expr:
    r"""
    Percentage Price Oscillator, also known as PPO — the fast/slow EMA gap as a percentage of the slow EMA.

    The scale-free sibling of :func:`absolute_price_oscillator`: the same fast-minus-slow :func:`ema` gap, divided
    by the slow EMA and put on a percentage scale, so oscillators of differently-priced series are comparable:

    .. math::

        \mathrm{PPO}_t = 100 \cdot \frac{\mathrm{EMA}(\mathrm{close}, n_{\mathrm{fast}})_t
            - \mathrm{EMA}(\mathrm{close}, n_{\mathrm{slow}})_t}{\mathrm{EMA}(\mathrm{close}, n_{\mathrm{slow}})_t}.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window_fast: Span of the fast EMA (canonically ``12``). Must be ``>= 1``.
        window_slow: Span of the slow EMA (canonically ``26``). Must be ``>= 1`` and ``>= window_fast``.

    Returns:
        The oscillator (in percent) for each row, the same length as the input. Values are ``null`` until both EMAs
        leave their warm-up (the first ``max(window_fast, window_slow) - 1`` rows).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window_fast < 1``, ``window_slow < 1``, or ``window_fast > window_slow`` (the fast leg must be
            the shorter one; ``window_fast == window_slow`` is allowed and gives an identically-zero oscillator).

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Moving average**

        Both legs use the exponential :func:`ema`. Being scale-free, PPO is invariant to the price's unit — multiplying
        the close by a constant leaves it unchanged.

        **Edge-case behavior**

        - **Null** — a leading ``null`` run stays ``null`` until the first non-null seed; an interior ``null`` yields
          ``null`` at that position while the recursion continues across the gap.
        - **NaN** — a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null
          position.
        - **Degenerate denominator** — when both the gap and the slow EMA are ``0`` the ratio is indeterminate, so the
          result is a ``0 / 0``, i.e. ``NaN`` — a non-zero gap over a zero slow EMA is ``+/-inf``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`absolute_price_oscillator`: The same gap in price units, before dividing by the slow EMA.
        - :func:`macd`: The oscillator built on this gap, with an added signal line.
        - :func:`ema`: The exponential moving average each leg is built from.

    References:
        - Appel, G. (2005). *Technical Analysis: Power Tools for Active Investors*. FT Press.

    Examples:
        Basic usage on a single price series:

        >>> import polars as pl
        >>> from pomata.indicators import percentage_price_oscillator
        >>>
        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0]})
        >>> expr = percentage_price_oscillator(pl.col("close"), window_fast=2, window_slow=3)
        >>> frame.select(percentage_price_oscillator=expr.round(4))["percentage_price_oscillator"].to_list()
        [None, None, 4.5455, 1.5152, 3.2407, 3.5613, 1.1871, 2.7484]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker's EMAs warm up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 4 + ["NVDA"] * 4,
        ...         "close": [10.0, 11.0, 12.0, 11.0] + [20.0, 22.0, 24.0, 22.0],
        ...     }
        ... )
        >>> expr = percentage_price_oscillator(pl.col("close"), window_fast=2, window_slow=3)
        >>> frame.with_columns(percentage_price_oscillator=expr.over("ticker").round(4))[
        ...     "percentage_price_oscillator"
        ... ].to_list()
        [None, None, 4.5455, 1.5152, None, None, 4.5455, 1.5152]

        A ``null`` (which the recursive EMA bridges) and a ``NaN`` (which latches) make the handling visible:

        >>> frame = pl.DataFrame({"close": [10.0, 11.0, None, 13.0, float("nan"), 15.0]})
        >>> expr = percentage_price_oscillator(pl.col("close"), window_fast=2, window_slow=3)
        >>> frame.select(percentage_price_oscillator=expr.round(4))["percentage_price_oscillator"].to_list()
        [None, None, None, 11.5546, nan, nan]

        **Degenerate denominator** — an all-zero series drives both EMAs to exactly ``0.0``, so the ``0/0`` boundary
        surfaces as ``NaN``:

        >>> frame = pl.DataFrame({"close": [0.0, 0.0, 0.0, 0.0]})
        >>> expr = percentage_price_oscillator(pl.col("close"), window_fast=2, window_slow=3)
        >>> frame.select(percentage_price_oscillator=expr)["percentage_price_oscillator"].to_list()
        [None, None, nan, nan]

        **Degenerate denominator** — a window summing to zero seeds the slow EMA at exactly ``0.0`` while the fast EMA
        stays non-zero, so the non-zero gap over the zero slow EMA is ``+/-inf``:

        >>> frame = pl.DataFrame({"close": [1.0, 1.0, -2.0]})
        >>> expr = percentage_price_oscillator(pl.col("close"), window_fast=2, window_slow=3)
        >>> frame.select(percentage_price_oscillator=expr)["percentage_price_oscillator"].to_list()
        [None, None, -inf]
    """
    expr = float64_expr(expr)
    validate_window(window_fast, name="window_fast")
    validate_window(window_slow, name="window_slow")
    validate_window_order(window_fast, window_slow)
    slow = ema(expr, window_slow)
    # APO normalized by the slow EMA and put on a percentage scale, so it is invariant to the price's unit.
    return ((ema(expr, window_fast) - slow) / slow * 100.0).name.keep()


def roc(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Rate of Change (ROC), also known as the Price Rate of Change (PROC) or n-day momentum (percent).

    The percentage change of the series relative to its value ``window`` observations ago — a pure momentum oscillator
    that is unbounded above and below and crosses zero when price returns to its lagged level:

    .. math::

        \mathrm{ROC}_t = 100 \cdot \frac{x_t - x_{t-n}}{x_{t-n}}, \qquad n = \text{window},

    where :math:`x_{t-n}` is the value ``window`` rows earlier (``expr.shift(window)``). It is the simple return over
    ``window`` periods expressed in percent; a positive value means the series rose over the lookback, a negative value
    that it fell, and ``0`` that it is unchanged.

    ROC is a ratio, so it is scale-invariant rather than scale-homogeneous: multiplying the input by a non-zero constant
    leaves the output unchanged (``roc(k * x) == roc(x)``), because the constant cancels between numerator and
    denominator.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of observations to look back. Must be ``>= 1``.

    Returns:
        The ROC for each row, the same length as ``expr``. The first ``window`` values are ``null`` (warm-up) -- the
        lagged term ``expr.shift(window)`` is undefined for the first ``window`` rows, so no change can be measured
        there.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Edge-case behavior**

        - **Null** — a ``null`` value makes that row ``null`` (``null`` takes precedence over ``NaN``).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **Insufficient sample** — a look-back longer than the series never completes, so the result is ``null``.
        - **Degenerate denominator** — when both the change and the lagged value are ``0`` the ratio is indeterminate,
          so the result is a ``0 / 0``, i.e. ``NaN`` — a non-zero change over a zero lagged value is ``+/-inf`` (the
          sign tracks the change relative to the signed zero).
        - **window == 1** — the look-back is a single bar, so ``roc`` is the one-period simple return in percent.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`mom`: The absolute-difference sibling.
        - :func:`trix`: The one-period rate of change of a triple-smoothed EMA.
        - :func:`rsi`: A bounded momentum oscillator.

    References:
        - https://en.wikipedia.org/wiki/Momentum_%28technical_analysis%29

    Examples:
        Basic usage on a single price series:

        >>> import polars as pl
        >>> from pomata.indicators import roc
        >>>
        >>> frame = pl.DataFrame({"close": [2.0, 4.0, 6.0, 8.0, 10.0]})
        >>> frame.select(roc=roc(pl.col("close"), window=2).round(4))["roc"].to_list()
        [None, None, 200.0, 100.0, 66.6667]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 5 + ["NVDA"] * 5,
        ...         "close": [10.0, 11.0, 12.0, 11.0, 13.0] + [20.0, 22.0, 21.0, 23.0, 22.0],
        ...     }
        ... )
        >>> frame.with_columns(roc=roc(pl.col("close"), window=2).over("ticker").round(4))["roc"].to_list()
        [None, None, 20.0, 0.0, 8.3333, None, None, 5.0, 4.5455, 4.7619]

        A ``null`` (voiding the rows that reference it) and a ``NaN`` (which propagates) make the exact handling visible
        at a glance:

        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, None, 15.0, float("nan"), 17.0, 18.0, 19.0]})
        >>> frame.select(roc=roc(pl.col("close"), window=2).round(4))["roc"].to_list()
        [None, None, 20.0, 18.1818, None, 15.3846, None, 13.3333, nan, 11.7647]

        **Insufficient sample** — a one-element series with a one-bar window stays undefined:

        >>> frame = pl.DataFrame({"close": [42.0]})
        >>> frame.select(roc=roc(pl.col("close"), window=1))["roc"].to_list()
        [None]

        **Degenerate denominator** — ROC of a constant non-zero series is exactly ``0`` once warmed up:

        >>> frame = pl.DataFrame({"close": [5.0, 5.0, 5.0, 5.0]})
        >>> frame.select(roc=roc(pl.col("close"), window=1))["roc"].to_list()
        [None, 0.0, 0.0, 0.0]

        **Degenerate denominator** — a non-zero change over a zero lagged value is ``+/-inf``, the sign tracking the
        change direction:

        >>> frame = pl.DataFrame({"close": [0.0, 5.0, 0.0, -5.0]})
        >>> frame.select(roc=roc(pl.col("close"), window=1))["roc"].to_list()
        [None, inf, -100.0, -inf]

        **Degenerate denominator** — a zero change over zero is ``NaN`` (``0/0``), while a non-zero change over zero is
        ``+inf``:

        >>> frame = pl.DataFrame({"close": [0.0, 0.0, 5.0]})
        >>> frame.select(roc=roc(pl.col("close"), window=1))["roc"].to_list()
        [None, nan, inf]
    """
    expr = float64_expr(expr)
    validate_window(window)
    past = expr.shift(window)
    return (100.0 * (expr - past) / past).name.keep()


def rsi(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Relative Strength Index (RSI), also known as Wilder's RSI.

    A bounded momentum oscillator (J. Welles Wilder, 1978) that measures the magnitude of recent up moves relative to
    recent down moves, mapped onto a fixed ``[0, 100]`` scale.

    With one-step price changes :math:`\Delta_t = x_t - x_{t-1}` split into gains and losses
    :math:`U_t = \max(\Delta_t, 0)`, :math:`D_t = \max(-\Delta_t, 0)`, smoothed by the Wilder moving average :func:`rma`
    (an exponential average with :math:`\alpha = 1 / n`):

    .. math::

        \mathrm{RS}_t = \frac{\mathrm{RMA}(U)_t}{\mathrm{RMA}(D)_t},
        \qquad
        \mathrm{RSI}_t = 100 - \frac{100}{1 + \mathrm{RS}_t},
        \qquad n = \text{window}.

    Equivalently :math:`\mathrm{RSI}_t = 100 \cdot \mathrm{RMA}(U)_t / (\mathrm{RMA}(U)_t + \mathrm{RMA}(D)_t)`, which
    makes the bounds explicit: a window with no losses gives :math:`\mathrm{RSI} = 100`, a window with no gains gives
    :math:`\mathrm{RSI} = 0`, and the value lives in :math:`[0, 100]` everywhere in between.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of observations in the Wilder moving window. Must be ``>= 1``.

    Returns:
        The RSI for each row, the same length as ``expr``. The first ``window`` values are ``null`` (warm-up) --
        Wilder's RSI needs ``window + 1`` prices for its first value, since row ``0`` has no difference and the gain /
        loss averages count ``window`` non-null differences before emitting.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Seeding**

        The gain and loss averages use Wilder's :func:`rma`, seeded with the simple average of the first ``window``
        gains and losses -- Wilder's canonical initialization, exact from the first emitted value.

        **Edge-case behavior**

        - **Null** — a leading ``null`` run stays ``null`` until the first non-null seed; an interior ``null`` yields
          ``null`` at that position while the recursion continues across the gap — the one-bar difference reads the
          missing observation twice, so an interior ``null`` voids its own row and the next.
        - **NaN** — a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null
          position.
        - **Insufficient sample** — a series shorter than ``window + 1`` never completes the first difference, so the
          result is ``null``.
        - **Degenerate denominator** — no up and no down move leaves the relative strength indeterminate, so the result
          is a ``0 / 0``, i.e. ``NaN`` (genuinely undefined, not a conventional ``50`` or ``100``).
        - **window == 1** — the smoothing vanishes: each row reports ``100`` on an up move, ``0`` on a down move, and
          ``NaN`` on no move.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`rma`: Wilder's moving average that smooths the gains and losses RSI is built on.
        - :func:`money_flow_index`: The volume-weighted analog — the same oscillator on raw money flow.
        - :func:`chande_momentum_oscillator`: The unsmoothed sibling that sums gains and losses over a fixed window.

    References:
        - Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
        - https://en.wikipedia.org/wiki/Relative_strength_index

    Examples:
        Basic usage on a single price series:

        >>> import polars as pl
        >>> from pomata.indicators import rsi
        >>>
        >>> frame = pl.DataFrame({"close": [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.1, 45.42]})
        >>> frame.select(rsi=rsi(pl.col("close"), window=3).round(4))["rsi"].to_list()
        [None, None, None, 7.0588, 59.0674, 74.1408, 80.0819, 85.8581]

        On a multi-ticker panel, wrap the call in ``.over`` so the difference and the recursion restart per group --
        note that each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 5 + ["NVDA"] * 5,
        ...         "close": [10.0, 11.0, 10.5, 11.5, 12.5] + [50.0, 49.0, 51.0, 50.5, 52.0],
        ...     }
        ... )
        >>> frame.with_columns(rsi=rsi(pl.col("close"), window=3).over("ticker").round(2))["rsi"].to_list()
        [None, None, None, 80.0, 87.5, None, None, None, 57.14, 73.91]

        A ``null`` (skipped, and any window it touches yields ``null``) and a ``NaN`` (which propagates) make the exact
        handling visible at a glance:

        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, None, 15.0, float("nan"), 17.0, 18.0, 19.0]})
        >>> frame.select(rsi=rsi(pl.col("close"), window=2).round(4))["rsi"].to_list()
        [None, None, 100.0, 100.0, None, None, nan, nan, nan, nan]

        **Insufficient sample** — a one-element series has no difference to seed the recursion:

        >>> frame = pl.DataFrame({"close": [42.0]})
        >>> frame.select(rsi=rsi(pl.col("close"), window=1))["rsi"].to_list()
        [None]

        **Degenerate denominator** — zero gain and zero loss is the indeterminate ``0/0`` relative strength, surfaced as
        ``NaN``:

        >>> frame = pl.DataFrame({"close": [5.0, 5.0, 5.0, 5.0]})
        >>> frame.select(rsi=rsi(pl.col("close"), window=2))["rsi"].to_list()
        [None, None, nan, nan]

        **window == 1** — a one-bar window collapses the Wilder smoothing to the raw move direction: ``100`` up, ``0``
        down:

        >>> frame = pl.DataFrame({"close": [1.0, 3.0, 2.0, 5.0]})
        >>> frame.select(rsi=rsi(pl.col("close"), window=1))["rsi"].to_list()
        [None, 100.0, 0.0, 100.0]
    """
    expr = float64_expr(expr)
    validate_window(window)
    delta = expr.diff()
    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)
    average_gain = rma(gain, window)
    average_loss = rma(loss, window)
    relative_strength = average_gain / average_loss
    return (100.0 - 100.0 / (1.0 + relative_strength)).name.keep()


def rsi_stochastic(
    expr: pl.Expr,
    *,
    window_rsi: int,
    window_k: int,
    window_d: int,
) -> pl.Expr:
    r"""
    Stochastic Relative Strength Index, also known as the Stochastic RSI.

    Introduced by Tushar Chande and Stanley Kroll (1994): the Fast Stochastic Oscillator applied to :func:`rsi` instead
    of price, which sharpens the bounded RSI into a faster ``[0, 100]`` oscillator by locating it within its own recent
    range. The raw line %K places the RSI within its ``window_k`` range, and %D is the :func:`sma` of %K:

    .. math::

        \%\mathrm{K}_t &= 100 \cdot \frac{\mathrm{RSI}_t - \mathrm{RSImin}_t}{\mathrm{RSImax}_t - \mathrm{RSImin}_t}, \\
        \%\mathrm{D}_t &= \mathrm{SMA}(\%\mathrm{K}, m)_t,

    where :math:`\mathrm{RSI}` is the :func:`rsi` over ``window_rsi``, :math:`\mathrm{RSImin}_t` and
    :math:`\mathrm{RSImax}_t` are its lowest and highest values over the ``window_k`` bars ending at :math:`t`, and
    :math:`m` is ``window_d``.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window_rsi: Number of observations in the underlying :func:`rsi` (canonically ``14``). Must be ``>= 1``.
        window_k: Number of observations in the %K look-back range over the RSI (canonically ``14``). Must be ``>= 1``.
        window_d: Number of observations in the %D moving average of %K (canonically ``3``). Must be ``>= 1``.

    Returns:
        A struct ``pl.Expr`` with two ``Float64`` fields, the same length as ``expr``:

        - ``k`` — the raw %K line, ``100 * (rsi - RSImin) / (RSImax - RSImin)``.
        - ``d`` — the %D signal line, the :func:`sma` of %K over ``window_d``.

        Read one line with ``.struct.field("k")`` (etc.) or split both into columns with ``.struct.unnest()``. The
        warm-up stacks the :func:`rsi` warm-up (``window_rsi`` rows), the ``window_k - 1`` range look-back, and the
        ``window_d - 1`` of %D.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window_rsi < 1``, ``window_k < 1``, or ``window_d < 1``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        Both lines lie in ``[0, 100]``. Because the underlying :func:`rsi` is already scale-invariant, so is this; there
        is no homogeneity to test.

        **Composition**

        Built from :func:`rsi` (whose recursive Wilder seeding it inherits — see that function's ``Seeding`` note), then
        the %K range ratio, then the :func:`sma` of %K, so every stage's warm-up and ``null`` / ``NaN`` handling stacks.

        **Edge-case behavior**

        - **Null** — a leading ``null`` run stays ``null`` until the first non-null seed; an interior ``null`` yields
          ``null`` at that position while the recursion continues across the gap.
        - **NaN** — a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null
          position.
        - **Degenerate denominator** — when the RSI does not move over the look-back (its highest equals its lowest,
          e.g. a sustained trend pinning the RSI) the ``%K`` normalization is indeterminate, so the result is a
          ``0 / 0``, i.e. ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`rsi`: The oscillator this is the stochastic of.
        - :func:`stochastic_fast`: The same %K / %D construction applied to price.
        - :func:`stochastic_slow`: The smoothed %K / %D stochastic variant.

    References:
        - Chande, T. S. & Kroll, S. (1994). *The New Technical Trader*. Wiley.

    Examples:
        Basic usage on a single price series:

        >>> import polars as pl
        >>> from pomata.indicators import rsi_stochastic
        >>>
        >>> frame = pl.DataFrame({"close": [50.0, 51.0, 50.5, 52.0, 51.5, 53.0, 52.0, 54.0, 53.5, 55.0]})
        >>> expr = rsi_stochastic(pl.col("close"), window_rsi=3, window_k=3, window_d=2)
        >>> frame.select(k=expr.struct.field("k").round(4))["k"].to_list()
        [None, None, None, None, None, 94.7368, 0.0, 81.5861, 44.2237, 100.0]
        >>> frame.select(d=expr.struct.field("d").round(4))["d"].to_list()
        [None, None, None, None, None, None, 47.3684, 40.793, 62.9049, 72.1118]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 8 + ["NVDA"] * 8,
        ...         "close": [50.0, 51.0, 50.5, 52.0, 51.5, 53.0, 52.0, 54.0]
        ...         + [40.0, 41.0, 40.5, 42.0, 41.5, 43.0, 42.0, 44.0],
        ...     }
        ... )
        >>> expr = rsi_stochastic(pl.col("close"), window_rsi=3, window_k=3, window_d=2)
        >>> frame.with_columns(k=expr.over("ticker").struct.field("k").round(4))["k"].to_list()
        [None, None, None, None, None, 94.7368, 0.0, 81.5861, None, None, None, None, None, 94.7368, 0.0, 81.5861]

        A ``null`` (which nulls the dependent %K) and a ``NaN`` (which propagates) make the handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "close": [
        ...             50.0,
        ...             51.0,
        ...             50.5,
        ...             52.0,
        ...             51.5,
        ...             53.0,
        ...             52.0,
        ...             54.0,
        ...             None,
        ...             55.0,
        ...             float("nan"),
        ...             56.0,
        ...             57.0,
        ...             58.0,
        ...         ],
        ...     }
        ... )
        >>> expr = rsi_stochastic(pl.col("close"), window_rsi=3, window_k=3, window_d=2)
        >>> frame.select(k=expr.struct.field("k").round(4))["k"].to_list()
        [None, None, None, None, None, 94.7368, 0.0, 81.5861, None, None, None, None, nan, nan]

        **Degenerate denominator** — a monotone run gives an exactly-flat RSI, so the %K channel normalization is the
        ``0/0`` degenerate ``NaN``:

        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, 14.0]})
        >>> expr = rsi_stochastic(pl.col("close"), window_rsi=2, window_k=2, window_d=1)
        >>> frame.select(k=expr.struct.field("k"))["k"].to_list()
        [None, None, None, nan, nan]
    """
    expr = float64_expr(expr)
    validate_window(window_rsi, name="window_rsi")
    validate_window(window_k, name="window_k")
    validate_window(window_d, name="window_d")
    strength = rsi(expr, window_rsi)
    lowest = strength.rolling_min(window_k)
    highest = strength.rolling_max(window_k)
    percent_k = 100.0 * (strength - lowest) / (highest - lowest)
    return (pl.struct(k=percent_k, d=sma(percent_k, window_d))).name.keep()


def trix(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    TRIX — the one-period rate of change of a triple-smoothed exponential moving average.

    A momentum oscillator that triple-smooths the close with chained :func:`ema` passes to strip out cycles shorter than
    ``window``, then takes the one-period percentage :func:`roc` of that smoothed line, so it oscillates around zero and
    filters minor noise:

    .. math::

        \mathrm{TRIX}_t = 100 \cdot \frac{\mathrm{TE}_t - \mathrm{TE}_{t-1}}{\mathrm{TE}_{t-1}},
            \qquad \mathrm{TE} = \mathrm{EMA}\bigl(\mathrm{EMA}(\mathrm{EMA}(\mathrm{close}, n), n), n\bigr).

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Span of each of the three EMA passes. Must be ``>= 1``.

    Returns:
        The oscillator (in percent) for each row, the same length as the input. The first ``3 * (window - 1) + 1`` rows
        are ``null`` (warm-up): three chained EMAs plus the one-period rate of change.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Edge-case behavior**

        - **Null** — a leading ``null`` run stays ``null`` until the first non-null seed; an interior ``null`` yields
          ``null`` at that position while the recursion continues across the gap.
        - **NaN** — a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null
          position.
        - **Degenerate denominator** — a zero-valued history drives the triple EMA to exactly ``0``, so the one-period
          rate of change is a ``0 / 0``, i.e. ``NaN`` while the EMA holds at zero, and ``+/-inf`` — reported, not
          clipped — the moment it moves off it.
        - **window == 1** — each EMA pass is the identity, so TRIX is the one-period rate of change of ``expr``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`ema`: The exponential moving average chained three times.
        - :func:`roc`: The one-period rate of change applied to the smoothed line.
        - :func:`tema`: Another triple-EMA construction, blending the three passes differently.

    References:
        - Hutson, J. K. (1983). "Good Trix." *Technical Analysis of Stocks & Commodities*.
        - https://en.wikipedia.org/wiki/Trix_%28technical_analysis%29

    Examples:
        Basic usage on a single price series:

        >>> import polars as pl
        >>> from pomata.indicators import trix
        >>>
        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0]})
        >>> frame.select(trix=trix(pl.col("close"), window=2).round(4))["trix"].to_list()
        [None, None, None, None, 5.4718, 7.4466, 2.989, 5.4253]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker's EMA chain warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 6 + ["NVDA"] * 6,
        ...         "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0] + [20.0, 22.0, 24.0, 26.0, 28.0, 30.0],
        ...     }
        ... )
        >>> frame.with_columns(trix=trix(pl.col("close"), window=2).over("ticker").round(4))["trix"].to_list()
        [None, None, None, None, 8.6957, 8.0, None, None, None, None, 8.6957, 8.0]

        A ``null`` (which the EMA chain bridges) and a ``NaN`` (which latches) make the handling visible:

        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, None, 14.0, float("nan"), 16.0, 17.0]})
        >>> frame.select(trix=trix(pl.col("close"), window=2).round(4))["trix"].to_list()
        [None, None, None, None, None, nan, nan, nan]

        **Degenerate denominator** — a zero-valued history drives the triple EMA to exactly zero, so the rate of change
        divides by that zero: a ``0/0`` while the EMA holds at zero, ``+/-inf`` the moment it moves off it:

        >>> frame = pl.DataFrame({"close": [0.0, 0.0, 0.0, 0.0, 0.0, 5.0]})
        >>> frame.select(trix=trix(pl.col("close"), window=2))["trix"].to_list()
        [None, None, None, None, nan, inf]

        **window == 1** — a one-bar window makes every EMA pass the identity, degenerating TRIX to the one-period
        percentage ROC of the raw input:

        >>> frame = pl.DataFrame({"close": [100.0, 120.0, 90.0, 108.0]})
        >>> frame.select(trix=trix(pl.col("close"), window=1))["trix"].to_list()
        [None, 20.0, -25.0, 20.0]
    """
    expr = float64_expr(expr)
    validate_window(window)
    # Triple-smoothed EMA, then its one-period percentage rate of change.
    triple_ema = ema(ema(ema(expr, window), window), window)
    return roc(triple_ema, 1).name.keep()


def ultimate_oscillator(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    *,
    window_short: int,
    window_medium: int,
    window_long: int,
) -> pl.Expr:
    r"""
    Ultimate Oscillator — buying pressure blended over three windows with weights 4:2:1, scaled to [0, 100].

    Introduced by Larry Williams (1976): a momentum oscillator that blends buying pressure over three time frames to
    damp the false divergences a single-period oscillator throws off. Each bar's buying pressure is the close above its
    true low, normalized by the true range; the three period sums are averaged with weights ``4 : 2 : 1`` (the shortest
    matters most) and scaled to ``[0, 100]``:

    .. math::

        \mathrm{BP}_t &= \mathrm{close}_t - \min(\mathrm{low}_t, \mathrm{close}_{t-1}), \\
        \mathrm{TR}_t &= \max(\mathrm{high}_t, \mathrm{close}_{t-1}) - \min(\mathrm{low}_t, \mathrm{close}_{t-1}), \\
        \mathrm{avg}_n &= \frac{\sum_{i} \mathrm{BP}_{t-i}}{\sum_{i} \mathrm{TR}_{t-i}} \quad (i = 0 \dots n - 1), \\
        \mathrm{UO}_t &= 100 \cdot \frac{4\,\mathrm{avg}_{n_s} + 2\,\mathrm{avg}_{n_m} + \mathrm{avg}_{n_l}}{7},

    where :math:`n_s`, :math:`n_m`, :math:`n_l` are ``window_short`` / ``window_medium`` / ``window_long``.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        window_short: Number of observations in the short averaging window (weight ``4``, canonically ``7``). Must be
            ``>= 1``.
        window_medium: Number of observations in the medium averaging window (weight ``2``, canonically ``14``). Must be
            ``>= 1``.
        window_long: Number of observations in the long averaging window (weight ``1``, canonically ``28``). Must be
            ``>= 1``.

    Returns:
        The Ultimate Oscillator for each row, the same length as the inputs, in ``[0, 100]`` for well-formed bars. The
        first ``max(window_short, window_medium, window_long) - 1`` values are ``null`` (warm-up). The bound is not
        guaranteed for an incoherent bar: a missing or ``NaN`` ``low`` on a down bar (the documented fallback below)
        substitutes the previous ``close`` into the true low, which can make the buying pressure negative and push the
        value outside ``[0, 100]``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window_short < 1``, ``window_medium < 1``, ``window_long < 1``, or the periods are not ordered
            ``window_short <= window_medium <= window_long`` (the three windows must run shortest to longest).

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        It is scale-invariant under a positive common rescaling of ``high``, ``low``, and ``close`` (each averaged term
        is a ratio of price ranges).

        **Seeding**

        Row ``0`` has no previous close, so the true low and true high fall back to that bar's own low and high.

        **Edge-case behavior**

        - **Null** — a ``null`` in a single ``high`` / ``low`` / ``close`` drops only the terms that reference it (the
          true low / high follow ``pl.min_horizontal`` / ``pl.max_horizontal``, which skip nulls); a ``null`` reaching a
          period sum yields ``null`` for the rows whose window touches it.
        - **NaN** — the per-field behavior is asymmetric. A ``NaN`` in ``high`` or ``close`` propagates
          (``pl.max_horizontal`` treats it as the largest value, and a corrupt close poisons the next bar's true range),
          yielding ``NaN``. A ``NaN`` in ``low`` on a bar with a finite previous close is instead treated as absent:
          ``pl.min_horizontal`` skips it and the true low falls back to the previous close, so the bar reports a finite
          value computed from the substituted close (only at row ``0``, where there is no previous close, does a ``NaN``
          ``low`` propagate).
        - **Degenerate denominator** — an exactly-flat true range with zero buying pressure — the genuine degenerate,
          detected via the residual-free rolling maxima of the true range and the buying pressure — is indeterminate, so
          the result is a ``0 / 0``, i.e. ``NaN``; a finite buying pressure over an exactly-zero true range (the
          missing-``low`` fallback) is left to IEEE-754 as ``+/-inf``, and a near-flat range is reported, not clipped
          (the ``[0, 100]`` bound is conditional on well-formed bars, so past a sane dynamic range its precision
          degrades — see the precision note above).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`rsi`: The single-period momentum oscillator this generalizes across three.
        - :func:`williams_r`: Another high-low-range momentum oscillator.
        - :func:`true_range`: The per-bar true range the buying pressure is normalized by.

    References:
        - Williams, L. (1985). "The Ultimate Oscillator." *Technical Analysis of Stocks & Commodities*.
        - https://en.wikipedia.org/wiki/Ultimate_oscillator

    Examples:
        Basic usage on high-low-close bars:

        >>> import polars as pl
        >>> from pomata.indicators import ultimate_oscillator
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5],
        ...         "close": [9.5, 10.5, 11.5, 11.0, 12.5, 12.0],
        ...     }
        ... )
        >>> expr = ultimate_oscillator(
        ...     pl.col("high"), pl.col("low"), pl.col("close"), window_short=2, window_medium=3, window_long=4
        ... )
        >>> frame.select(ultimate_oscillator=expr.round(4))["ultimate_oscillator"].to_list()
        [None, None, None, 60.7143, 66.6667, 65.0433]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 6 + ["NVDA"] * 6,
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5] + [20.0, 21.0, 22.0, 21.5, 23.0, 22.5],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5] + [19.0, 20.0, 21.0, 20.5, 22.0, 21.5],
        ...         "close": [9.5, 10.5, 11.5, 11.0, 12.5, 12.0] + [19.5, 20.5, 21.5, 21.0, 22.5, 22.0],
        ...     }
        ... )
        >>> expr = ultimate_oscillator(
        ...     pl.col("high"), pl.col("low"), pl.col("close"), window_short=2, window_medium=3, window_long=4
        ... )
        >>> frame.with_columns(ultimate_oscillator=expr.over("ticker").round(4))["ultimate_oscillator"].to_list()
        [None, None, None, 60.7143, 66.6667, 65.0433, None, None, None, 60.7143, 66.6667, 65.0433]

        A ``null`` (which nulls the windows that cover it) and a ``NaN`` (which propagates, also poisoning the next
        bar's true range) in ``close`` make the handling visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5, 16.0, 15.5, 17.0],
        ...         "low": [9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5, 16.0],
        ...         "close": [9.5, 10.5, 11.5, 11.0, 12.5, 12.0, None, 13.0, 12.5, 13.5, float("nan"), 14.0, 15.0],
        ...     }
        ... )
        >>> expr = ultimate_oscillator(
        ...     pl.col("high"), pl.col("low"), pl.col("close"), window_short=2, window_medium=3, window_long=4
        ... )
        >>> frame.select(ultimate_oscillator=expr.round(4))["ultimate_oscillator"].to_list()
        [None, None, None, 60.7143, 66.6667, 65.0433, None, None, None, None, nan, nan, nan]

        **Degenerate denominator** — the ``0/0`` degenerate on a flat well-formed series, detected via residual-free
        rolling maxima:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 10.0, 10.0],
        ...         "low": [10.0, 10.0, 10.0],
        ...         "close": [10.0, 10.0, 10.0],
        ...     }
        ... )
        >>> expr = ultimate_oscillator(
        ...     pl.col("high"), pl.col("low"), pl.col("close"), window_short=1, window_medium=1, window_long=2
        ... )
        >>> frame.select(ultimate_oscillator=expr)["ultimate_oscillator"].to_list()
        [None, nan, nan]

        **Degenerate denominator** — a missing low sends the true range to zero through the prior-close fallback while
        the buying pressure stays positive, so the quotient is ``+/-inf``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 8.0],
        ...         "low": [10.0, None],
        ...         "close": [10.0, 12.0],
        ...     }
        ... )
        >>> expr = ultimate_oscillator(
        ...     pl.col("high"), pl.col("low"), pl.col("close"), window_short=1, window_medium=1, window_long=2
        ... )
        >>> frame.select(ultimate_oscillator=expr)["ultimate_oscillator"].to_list()
        [None, inf]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    validate_window(window_short, name="window_short")
    validate_window(window_medium, name="window_medium")
    validate_window(window_long, name="window_long")
    # 3-way chain: validate_window_order in the hub is pairwise by design, so this ordering guard stays inline.
    if not window_short <= window_medium <= window_long:
        raise ValueError(
            f"windows must be ordered window_short <= window_medium <= window_long, "
            f"got window_short={window_short}, window_medium={window_medium}, window_long={window_long}"
        )
    previous_close = close.shift(1)
    true_low = pl.min_horizontal(low, previous_close)
    true_high = pl.max_horizontal(high, previous_close)
    buying_pressure = close - true_low
    true_range = true_high - true_low

    def averaged(
        window: int,
    ) -> pl.Expr:
        # The genuine 0/0 degenerate (an exactly-flat true range with no buying pressure) is detected residual-free via
        # the rolling maxima and returned as NaN deterministically. A finite buying pressure over a zero true range --
        # reachable only through the documented missing-low fallback -- is left to IEEE-754 as +/-inf, the deliberate
        # malformed-bar signal. The near-flat residual is reported as-is, not clipped: the [0, 100] bound is conditional
        # on well-formed bars, so beyond a sane dynamic range the streaming quotient degrades but is reported, not
        # silenced (see the documentation's Correctness page).
        raw = buying_pressure.rolling_sum(window) / true_range.rolling_sum(window)
        is_zero_over_zero = (true_range.rolling_max(window) == 0) & (buying_pressure.abs().rolling_max(window) == 0)
        return pl.when(is_zero_over_zero).then(float("nan")).otherwise(raw)

    weighted = 4.0 * averaged(window_short) + 2.0 * averaged(window_medium) + averaged(window_long)
    return (100.0 * weighted / 7.0).name.keep()


def williams_r(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Williams %R (Williams Percent Range), also known as %R / Williams Overbought-Oversold Index.

    A bounded momentum oscillator introduced by Larry Williams that reports where the current close sits inside the
    high-low range of the last ``window`` bars, expressed on a :math:`[-100, 0]` scale.

    It is effectively the inverse of the Fast Stochastic %K: a reading near ``0`` means the close is at the top of the
    recent range (overbought), while a reading near ``-100`` means it is at the bottom (oversold):

    .. math::

        \%R_t = -100 \cdot \frac{\mathrm{HH}_t - C_t}{\mathrm{HH}_t - \mathrm{LL}_t},
        \qquad
        \mathrm{HH}_t = \max_{0 \le i < n} H_{t-i}, \quad
        \mathrm{LL}_t = \min_{0 \le i < n} L_{t-i}, \quad n = \text{window},

    where :math:`H`, :math:`L`, :math:`C` are ``high``, ``low``, ``close``, :math:`\mathrm{HH}` is the highest high and
    :math:`\mathrm{LL}` the lowest low over the window. For well-formed bars (:math:`L \le C \le H`) the close lies
    inside the windowed range, so :math:`\%R \in [-100, 0]`. It is invariant to a common positive rescaling of ``high``,
    ``low``, and ``close`` (it is a ratio of price differences) and to a common additive shift of all three, so it
    carries no price units. The original convention writes the oscillator as ``-100 * (HH - C) / (HH - LL)``; some
    charting packages flip the sign to plot it on ``[0, 100]``, which is the same information mirrored about zero.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        The Williams %R for each row, the same length as the inputs. The first ``window - 1`` values are ``null``
        (warm-up), matching the rolling moving-average family: the value is defined only once ``window`` observations
        have been seen.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Warm-up**

        The warm-up is the canonical ``window - 1`` leading nulls of the rolling family, and the ``null`` / ``NaN``
        contract below matches the simple moving average.

        **Edge-case behavior**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values) —
          this covers ``high`` and ``low``; the ``close`` enters elementwise, so a ``null`` ``close`` nulls only its own
          bar (``null`` takes precedence over ``NaN``).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there; a ``NaN`` ``close`` yields ``NaN``
          only at its own bar.
        - **Insufficient sample** — a window longer than the series never completes, so the result is ``null``.
        - **Degenerate denominator** — when the windowed range collapses (:math:`\mathrm{HH} = \mathrm{LL}`, e.g. a flat
          high-low over the whole window) with the close on that level the ratio is indeterminate, so the result is a
          ``0 / 0``, i.e. ``NaN`` — a non-zero numerator over the zero range is ``+/-inf``.
        - **window == 1** — the highest high and lowest low collapse to the single bar's own ``high`` and ``low``, so
          :math:`\%R = -100\,(H - C) / (H - L)`.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`stochastic_fast`: The Fast Stochastic %K this oscillator inverts.
        - :func:`rsi`: A bounded momentum oscillator on the same [0, 100]-style scale.
        - :func:`cci`: Another bounded oscillator over a rolling window.

    References:
        - Williams, L. (1973). *How I Made One Million Dollars Last Year Trading Commodities*.
        - https://en.wikipedia.org/wiki/Williams_%25R

    Examples:
        Basic usage on high-low-close bars:

        >>> import polars as pl
        >>> from pomata.indicators import williams_r
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 12.0, 11.0, 13.0, 15.0, 14.0],
        ...         "low": [8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
        ...         "close": [9.0, 11.0, 10.5, 12.0, 14.0, 13.5],
        ...     }
        ... )
        >>> expr = williams_r(pl.col("high"), pl.col("low"), pl.col("close"), window=3)
        >>> frame.select(williams_r=expr.round(4))["williams_r"].to_list()
        [None, None, -37.5, -25.0, -20.0, -37.5]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 4 + ["NVDA"] * 4,
        ...         "high": [11.0, 12.0, 13.0, 12.0] + [21.0, 23.0, 22.0, 24.0],
        ...         "low": [9.0, 10.0, 11.0, 10.0] + [19.0, 21.0, 20.0, 22.0],
        ...         "close": [10.0, 11.0, 12.0, 11.0] + [20.0, 22.0, 21.0, 23.0],
        ...     }
        ... )
        >>> expr = williams_r(pl.col("high"), pl.col("low"), pl.col("close"), window=2)
        >>> frame.with_columns(williams_r=expr.over("ticker").round(4))["williams_r"].to_list()
        [None, -33.3333, -33.3333, -66.6667, None, -25.0, -66.6667, -25.0]

        A ``null`` and a ``NaN`` in ``close`` (each confined to its own bar, since the close enters elementwise) make
        the exact handling visible at a glance:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        ...         "low": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        ...         "close": [10.0, 11.0, 12.0, None, 14.0, 15.0, float("nan"), 17.0],
        ...     }
        ... )
        >>> expr = williams_r(pl.col("high"), pl.col("low"), pl.col("close"), window=2)
        >>> frame.select(williams_r=expr.round(4))["williams_r"].to_list()
        [None, -33.3333, -33.3333, None, -33.3333, -33.3333, nan, -33.3333]

        **Insufficient sample** — a single bar with a one-bar window:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0],
        ...         "low": [8.0],
        ...         "close": [9.0],
        ...     }
        ... )
        >>> expr = williams_r(pl.col("high"), pl.col("low"), pl.col("close"), window=1)
        >>> frame.select(williams_r=expr)["williams_r"].to_list()
        [-50.0]

        **Degenerate denominator** — a flat window with close on that level is the ``0/0`` IEEE degenerate:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [5.0, 5.0, 5.0],
        ...         "low": [5.0, 5.0, 5.0],
        ...         "close": [5.0, 5.0, 5.0],
        ...     }
        ... )
        >>> expr = williams_r(pl.col("high"), pl.col("low"), pl.col("close"), window=2)
        >>> frame.select(williams_r=expr)["williams_r"].to_list()
        [None, nan, nan]

        **Degenerate denominator** — a flat window with close off that level is a non-zero numerator over a zero
        denominator, signed ``inf``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [5.0, 5.0],
        ...         "low": [5.0, 5.0],
        ...         "close": [3.0, 3.0],
        ...     }
        ... )
        >>> expr = williams_r(pl.col("high"), pl.col("low"), pl.col("close"), window=2)
        >>> frame.select(williams_r=expr)["williams_r"].to_list()
        [None, -inf]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    validate_window(window)
    highest_high = high.rolling_max(window)
    lowest_low = low.rolling_min(window)
    return (-100.0 * (highest_high - close) / (highest_high - lowest_low)).name.keep()
