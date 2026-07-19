"""
Volatility indicators.
"""

import polars as pl

from pomata._expr import float64_expr, validate_positive, validate_window
from pomata.indicators.moving_average import rma, sma
from pomata.indicators.statistic import standard_deviation_rolling

__all__ = ("atr", "atr_normalized", "bollinger_bands", "true_range")


def atr(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Average True Range (ATR), Wilder's volatility measure.

    Introduced by J. Welles Wilder in *New Concepts in Technical Trading Systems* (1978) as the Wilder-smoothed average
    of the :func:`true_range`. The true range captures the largest of the three candidate moves on each bar — the
    current high-low spread and the two gaps from the previous close — and the ATR smooths that series with Wilder's
    running average:

    .. math::

        \mathrm{TR}_t &= \max\!\bigl(\,\mathrm{high}_t - \mathrm{low}_t,\;
            \lvert \mathrm{high}_t - \mathrm{close}_{t-1} \rvert,\;
            \lvert \mathrm{low}_t - \mathrm{close}_{t-1} \rvert \,\bigr), \\
        \mathrm{ATR}_t &= \mathrm{RMA}(\mathrm{TR})_t,
            \qquad \alpha = \frac{1}{n}, \quad n = \text{window}.

    It is computed by composing the public :func:`true_range` and :func:`rma`, so the Wilder smoothing
    (:math:`\alpha = 1 / n`) is shared with the rest of Wilder's family (RSI, ADX, DMI) and the result is
    unit-consistent with price.

    Because every true-range candidate is a non-negative magnitude (the ``high - low`` spread of a well-formed bar, or
    one of the two absolute gap terms), the true range is non-negative, and the Wilder average of a non-negative series
    is itself non-negative for well-formed bars (``high >= low``).

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``); the previous close supplies the two gap terms.
        window: Number of observations in the Wilder moving window. Must be ``>= 1``.

    Returns:
        The ATR for each row, the same length as the inputs. The first ``window - 1`` values are ``null`` (warm-up),
        inherited from the :func:`rma` over the true-range series: the running average emits only once ``window``
        non-null true ranges have been counted, independent of where any interior ``null`` falls.

        The true range itself is defined from row ``0`` (the first bar has no previous close, so it degenerates to
        ``high - low`` with the two gap terms dropped), so the ATR warm-up is exactly the ``rma`` warm-up of
        ``window - 1``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Scaling**

        Scaling is homogeneous of degree ``1`` only for a positive factor: multiplying every price by ``k > 0`` scales
        the ATR by ``k``. A negative factor makes the bar incoherent (``high`` falls below ``low``), so it is not a
        clean rescale.

        **Seeding**

        The Wilder smoothing (:func:`rma`) is seeded with the simple average of the first ``window`` true ranges --
        Wilder's canonical initialization. The first true range is the bar's high-low range (no prior close extends it),
        so the seed and warm-up include it.

        **Edge-case behavior**

        - **Null** — a leading ``null`` run stays ``null`` until the first non-null seed; an interior ``null`` yields
          ``null`` at that position while the recursion continues across the gap — the true range is ``null`` only when
          every :func:`true_range` candidate term is ``null``.
        - **NaN** — a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null position
          — except at ``window == 1``, where the smoothing is the identity and no recursion exists to latch: the ``NaN``
          clears once it leaves the true range's one-bar reach.
        - **window == 1** — the smoothing factor is ``1`` and the warm-up vanishes, so the ATR reproduces the true range
          exactly: the ``max_horizontal``-reduced true range (not a textbook three-term true range whenever a candidate
          term is dropped by a ``null``).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`true_range`: The per-bar range this Wilder-smooths.
        - :func:`rma`: The Wilder moving average used for the smoothing.
        - :func:`atr_normalized`: The same ATR expressed as a percent of the current close.

    References:
        - Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
        - https://en.wikipedia.org/wiki/Average_true_range

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import atr
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 12.0, 13.0, 12.0, 14.0],
        ...         "low": [9.0, 10.0, 11.0, 10.0, 12.0],
        ...         "close": [9.5, 11.0, 12.0, 11.0, 13.0],
        ...     }
        ... )
        >>> frame.select(atr=atr(pl.col("high"), pl.col("low"), pl.col("close"), window=3).round(4))["atr"].to_list()
        [None, None, 1.8333, 1.8889, 2.2593]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 4 + ["B"] * 4,
        ...         "high": [12.0, 13.0, 12.5, 14.0, 22.0, 24.0, 23.0, 25.0],
        ...         "low": [10.0, 11.0, 11.0, 12.0, 20.0, 21.0, 21.0, 23.0],
        ...         "close": [11.0, 12.5, 11.5, 13.5, 21.5, 21.5, 22.5, 24.0],
        ...     }
        ... )
        >>> expr = atr(pl.col("high"), pl.col("low"), pl.col("close"), window=2)
        >>> frame.with_columns(atr=expr.over("ticker").round(4))["atr"].to_list()
        [None, 2.0, 1.75, 2.125, None, 2.5, 2.25, 2.375]

        A ``null`` ``close`` (absorbed, so the next bar falls back to ``high - low``) then a ``NaN`` ``close`` (which
        the Wilder recursion latches from the next bar on) make the exact handling visible at a glance:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        ...         "low": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        ...         "close": [11.5, 12.5, None, 14.5, 15.5, float("nan"), 17.5, 18.0],
        ...     }
        ... )
        >>> frame.select(atr=atr(pl.col("high"), pl.col("low"), pl.col("close"), window=2).round(4))["atr"].to_list()
        [None, 2.0, 2.0, 2.0, 2.0, 2.0, nan, nan]

        **window == 1** — window=1 makes the Wilder smoothing the identity, so the ATR reproduces the true range:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 12.0, 11.0, 13.0],
        ...         "low": [8.0, 9.0, 9.5, 10.0],
        ...         "close": [9.0, 11.0, 10.0, 12.0],
        ...     }
        ... )
        >>> frame.select(atr=atr(pl.col("high"), pl.col("low"), pl.col("close"), window=1))["atr"].to_list()
        [2.0, 3.0, 1.5, 3.0]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    validate_window(window)
    return (rma(true_range(high, low, close), window)).name.keep()


def atr_normalized(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Normalized Average True Range (NATR) — the ATR as a percentage of the current close.

    The :func:`atr` expressed as a percentage of the current close, so volatility is comparable across instruments and
    price levels (unlike the raw ATR, which is in price units). With the ATR over the same ``window``:

    .. math::

        \mathrm{NATR}_t = 100 \cdot \frac{\mathrm{ATR}_t}{\mathrm{close}_t}, \qquad n = \text{window}.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        window: Number of observations in the Wilder moving window. Must be ``>= 1``.

    Returns:
        The NATR (in percent) for each row, the same length as the inputs. The first ``window - 1`` values are ``null``
        (warm-up), inherited from the :func:`atr`.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        It is scale-invariant under a positive common rescaling of ``high``, ``low``, and ``close`` (the ATR and the
        close scale together).

        **Edge-case behavior**

        - **Null** — a leading ``null`` run stays ``null`` until the first non-null seed; an interior ``null`` yields
          ``null`` at that position while the recursion continues across the gap — inherited from :func:`atr`, with a
          ``null`` ``close`` also nulling the ratio at that row.
        - **NaN** — a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null position
          — inherited from :func:`atr`, with a ``NaN`` ``close`` also yielding ``NaN`` for the ratio at that row.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`atr`: The raw (price-unit) average true range this normalizes.
        - :func:`true_range`: The per-bar range underlying the ATR.
        - :func:`bollinger_bands`: Another volatility view, standard-deviation bands around a moving average.

    References:
        - Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import atr_normalized
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.2, 10.5, 10.7, 10.3, 10.8],
        ...         "low": [9.8, 10.0, 10.2, 9.9, 10.3],
        ...         "close": [10.0, 10.3, 10.5, 10.1, 10.6],
        ...     }
        ... )
        >>> expr = atr_normalized(pl.col("high"), pl.col("low"), pl.col("close"), window=2)
        >>> frame.select(atr_normalized=expr.round(4))["atr_normalized"].to_list()
        [None, 4.3689, 4.5238, 5.3218, 5.8373]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 4 + ["B"] * 4,
        ...         "high": [10.2, 10.5, 10.7, 10.3, 20.4, 21.0, 21.4, 20.6],
        ...         "low": [9.8, 10.0, 10.2, 9.9, 19.6, 20.0, 20.4, 19.8],
        ...         "close": [10.0, 10.3, 10.5, 10.1, 20.0, 20.6, 21.0, 20.2],
        ...     }
        ... )
        >>> expr = atr_normalized(pl.col("high"), pl.col("low"), pl.col("close"), window=2)
        >>> frame.with_columns(atr_normalized=expr.over("ticker").round(4))["atr_normalized"].to_list()
        [None, 4.3689, 4.5238, 5.3218, None, 4.3689, 4.5238, 5.3218]

        A ``null`` ``close`` (voiding the ratio at that row) then a ``NaN`` ``close`` (which propagates through the
        ratio and the latched ATR) make the missing-data handling visible at a glance:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.2, 10.5, 10.7, 10.9, 11.1, 11.3, 11.5, 11.7],
        ...         "low": [9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2],
        ...         "close": [10.0, 10.3, None, 10.7, float("nan"), 11.1, 11.3, 11.5],
        ...     }
        ... )
        >>> expr = atr_normalized(pl.col("high"), pl.col("low"), pl.col("close"), window=2)
        >>> frame.select(atr_normalized=expr.round(4))["atr_normalized"].to_list()
        [None, 4.3689, None, 4.5561, nan, nan, nan, nan]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    validate_window(window)
    return (100.0 * atr(high, low, close, window) / close).name.keep()


def bollinger_bands(
    expr: pl.Expr,
    window: int,
    *,
    multiplier: float = 2.0,
) -> pl.Expr:
    r"""
    Bollinger Bands, volatility bands around a moving average.

    Introduced by John Bollinger in the 1980s: a center band that is the :func:`sma` of ``expr``, with an upper and a
    lower band placed ``multiplier`` population standard deviations away. The bands widen as volatility rises and
    contract as it falls, so price is read relative to a band that breathes with the market:

    .. math::

        \mathrm{middle}_t &= \mathrm{SMA}(\mathrm{expr}, n)_t, \\
        \mathrm{upper}_t &= \mathrm{middle}_t + k \, \sigma_t, \\
        \mathrm{lower}_t &= \mathrm{middle}_t - k \, \sigma_t,

    where :math:`n` is the window, :math:`k` is ``multiplier``, and :math:`\sigma_t` is the population rolling
    :func:`standard_deviation_rolling` of ``expr`` over the same window.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of observations in the moving window. Must be ``>= 1``.
        multiplier: Number of standard deviations between the center band and each outer band (default ``2.0``). Must be
            a finite number ``> 0`` (a non-positive width would collapse or invert the bands). The bands are symmetric;
            for asymmetric bands compose :func:`sma` and :func:`standard_deviation_rolling` directly.

    Returns:
        A struct ``pl.Expr`` with three ``Float64`` fields, the same length as ``expr``:

        - ``lower`` — the lower band, ``middle - multiplier * sigma``.
        - ``middle`` — the center band, the :func:`sma` of ``expr``.
        - ``upper`` — the upper band, ``middle + multiplier * sigma``.

        Read one band with ``.struct.field("middle")`` (etc.) or split all three into columns with ``.struct.unnest()``.
        For the first ``window - 1`` rows (warm-up) every field of the struct is ``null`` (the struct row itself stays a
        valid struct).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``, or if ``multiplier`` is not a finite number ``> 0`` (i.e. ``<= 0``, ``NaN``, or
            ``±inf``).

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Composition**

        The bands are built from :func:`sma` (center) and the population :func:`standard_deviation_rolling` (width), so
        they inherit the warm-up and missing-data behavior of both — identically on every field of the struct.

        **Edge-case behavior**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values) —
          on all three fields.
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there — on all three fields.
        - **Degenerate denominator** — a window of equal values has zero standard deviation (see
          :func:`standard_deviation_rolling`), so all three bands collapse onto the constant — even at ``window == 1``,
          or just after a much larger value has left the window.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`sma`: The center band.
        - :func:`standard_deviation_rolling`: The band half-width, before scaling by ``multiplier``.
        - :func:`keltner_channels`: The same band shape with ATR width instead of a standard deviation.

    References:
        - Bollinger, J. (2001). *Bollinger on Bollinger Bands*. McGraw-Hill.
        - https://en.wikipedia.org/wiki/Bollinger_Bands

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import bollinger_bands
        >>>
        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 11.0, 13.0]})
        >>> expr = bollinger_bands(pl.col("close"), window=3)
        >>> frame.select(lower=expr.struct.field("lower").round(4))["lower"].to_list()
        [None, None, 9.367, 10.3905, 10.367]
        >>> frame.select(middle=expr.struct.field("middle").round(4))["middle"].to_list()
        [None, None, 11.0, 11.3333, 12.0]
        >>> frame.select(upper=expr.struct.field("upper").round(4))["upper"].to_list()
        [None, None, 12.633, 12.2761, 13.633]

        Split the struct into three columns with ``.struct.unnest()``:

        >>> frame.select(bollinger_bands=expr).unnest("bollinger_bands").columns
        ['lower', 'middle', 'upper']

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "close": [10.0, 11.0, 12.0, 20.0, 22.0, 21.0],
        ...     }
        ... )
        >>> expr = bollinger_bands(pl.col("close"), window=2)
        >>> frame.with_columns(middle=expr.over("ticker").struct.field("middle").round(4))["middle"].to_list()
        [None, 10.5, 11.5, None, 21.0, 21.5]

        A ``null`` and a ``NaN`` propagate to every band; the middle band makes the handling visible:

        >>> frame = pl.DataFrame({"close": [10.0, None, 12.0, float("nan"), 14.0, 15.0]})
        >>> expr = bollinger_bands(pl.col("close"), window=2)
        >>> frame.select(middle=expr.struct.field("middle").round(4))["middle"].to_list()
        [None, None, None, nan, nan, 14.5]

        **Degenerate denominator** — a constant window has zero deviation, so all three bands collapse onto the middle
        even after a much larger value has left the window, where the rolling kernel would otherwise leave a residue:

        >>> frame = pl.DataFrame({"close": [1000000.0, 0.1, 0.1, 0.1, 0.1]})
        >>> expr = bollinger_bands(pl.col("close"), window=3)
        >>> frame.select(lower=expr.struct.field("lower").round(4))["lower"].to_list()
        [None, None, -609475.5473, 0.1, 0.1]
    """
    expr = float64_expr(expr)
    validate_window(window)
    validate_positive(multiplier, "multiplier")
    # Center = SMA; bands = ± multiplier * population rolling std (composed, so the warm-up/null/NaN behavior matches).
    middle = sma(expr, window)
    half_width = multiplier * standard_deviation_rolling(expr, window)
    return (pl.struct(lower=middle - half_width, middle=middle, upper=middle + half_width)).name.keep()


def true_range(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
) -> pl.Expr:
    r"""
    True Range (TR), also known as Wilder's True Range.

    The single-bar volatility primitive J. Welles Wilder introduced as the building block of the Average True Range and
    the Directional Movement system. It generalizes the bar's high-low spread to account for gaps relative to the prior
    close, taking the largest of three distances:

    .. math::

        \mathrm{TR}_t = \max\!\Bigl(
            h_t - l_t,\;
            \lvert h_t - c_{t-1} \rvert,\;
            \lvert l_t - c_{t-1} \rvert
        \Bigr),

    where :math:`h`, :math:`l`, :math:`c` are ``high``, ``low``, ``close`` and :math:`c_{t-1}` is the previous
    ``close``. The first row has no previous ``close``, so the two gap terms vanish and
    :math:`\mathrm{TR}_0 = h_0 - l_0` (Wilder's original definition). TR is a base building block: it composes nothing
    and is itself the input to :func:`atr` and the volatility-normalized directional indicators.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``); the previous close supplies the two gap terms.

    Returns:
        The True Range for each row, the same length as the inputs. There is no window and no warm-up -- every row is
        defined from row ``0``, which falls back to ``high - low`` because no previous close exists. On well-formed OHLC
        data (``high >= low``) every value is non-negative.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Inputs**

        ``high``, ``low``, and ``close`` are taken as the canonical OHLC roles in that positional order and must share a
        length and alignment (the same row index is one bar).

        **Edge-case behavior**

        - **Null** — ``null`` handling follows ``pl.max_horizontal``, which **skips** ``null`` candidates rather than
          propagating them: a ``null`` in ``high`` or ``low`` (or a ``null`` previous ``close``) simply drops that
          candidate, so the row still resolves from whichever distances remain. The result is ``null`` only when all
          three candidates are ``null``: with a defined previous ``close`` that means ``high`` and ``low`` are both
          ``null`` at the row, but where the previous ``close`` is itself ``null`` (row ``0``, or any bar after a
          ``null`` close) the two gap distances are already ``null``, so a single ``null`` in ``high`` or ``low`` voids
          the row on its own.
        - **NaN** — a ``NaN`` price yields ``NaN`` for that row — it is not skipped like a ``null`` (it dominates the
          maximum), so a ``NaN`` ``close`` also contaminates the two gap terms of the next row.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`atr`: The Wilder-smoothed average of this per-bar range.
        - :func:`atr_normalized`: That average expressed as a percent of the current close.
        - :func:`vortex`: A directional indicator that normalizes its movement by this range.

    References:
        - Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
        - https://en.wikipedia.org/wiki/Average_true_range

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import true_range
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 12.0, 11.5, 13.0, 12.5],
        ...         "low": [9.0, 10.5, 10.0, 11.0, 11.5],
        ...         "close": [9.5, 11.0, 10.5, 12.5, 12.0],
        ...     }
        ... )
        >>> expr = true_range(pl.col("high"), pl.col("low"), pl.col("close"))
        >>> frame.select(true_range=expr.round(4))["true_range"].to_list()
        [1.0, 2.5, 1.5, 2.5, 1.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 4 + ["B"] * 4,
        ...         "high": [12.0, 13.0, 12.5, 14.0, 22.0, 24.0, 23.0, 25.0],
        ...         "low": [10.0, 11.0, 11.0, 12.0, 20.0, 21.0, 21.0, 23.0],
        ...         "close": [11.0, 12.5, 11.5, 13.5, 21.5, 21.5, 22.5, 24.0],
        ...     }
        ... )
        >>> expr = true_range(pl.col("high"), pl.col("low"), pl.col("close"))
        >>> frame.with_columns(true_range=expr.over("ticker").round(4))["true_range"].to_list()
        [2.0, 2.0, 1.5, 2.5, 2.0, 3.0, 2.0, 2.5]

        A ``null`` ``close`` (skipped, so the next bar falls back to ``high - low``) then a ``NaN`` ``close`` (which
        contaminates only the following bar's gap terms) make the exact handling visible at a glance:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        ...         "low": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        ...         "close": [11.5, 12.5, None, 14.5, 15.5, float("nan"), 17.5, 18.0],
        ...     }
        ... )
        >>> expr = true_range(pl.col("high"), pl.col("low"), pl.col("close"))
        >>> frame.select(true_range=expr.round(4))["true_range"].to_list()
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, nan, 2.0]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    # Pure native Polars expressions (shift / abs / max_horizontal): lazy/eager-uniform, no Rust kernel needed.
    # max_horizontal skips null candidates and propagates NaN; row 0 has no previous close, so TR0 = high - low.
    previous_close = close.shift(1)
    return (
        pl.max_horizontal(
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        )
    ).name.keep()
