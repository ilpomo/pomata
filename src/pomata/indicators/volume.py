"""
Volume indicators.
"""

import polars as pl

from pomata._expr import float64_expr, validate_window, validate_window_order
from pomata.indicators.moving_average import ema
from pomata.indicators.price_transform import price_typical

__all__ = (
    "accumulation_distribution",
    "accumulation_distribution_oscillator",
    "chaikin_money_flow",
    "money_flow_index",
    "obv",
    "vwap",
)


def accumulation_distribution(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    volume: pl.Expr,
) -> pl.Expr:
    r"""
    Accumulation/Distribution Line (AD), also known as the Accumulation/Distribution Index or the Chaikin A/D Line.

    A cumulative volume-flow indicator popularized by Marc Chaikin that gauges the cumulative flow of money into and
    out of an instrument. Each bar contributes a fraction of its ``volume`` weighted by where the ``close`` sits inside
    the bar's high-low range. The per-bar weight is the **Money Flow Multiplier** (``MFM``), bounded in :math:`[-1, +1]`
    (``+1`` at the high, ``-1`` at the low); multiplying it by ``volume`` gives the **Money Flow Volume** (``MFV``); the
    line is the running cumulative sum of ``MFV``:

    .. math::

        \mathrm{MFM}_t &= \frac{(\mathrm{close}_t - \mathrm{low}_t) - (\mathrm{high}_t - \mathrm{close}_t)}
                               {\mathrm{high}_t - \mathrm{low}_t}, \\
        \mathrm{MFV}_t &= \mathrm{MFM}_t \cdot \mathrm{volume}_t, \\
        \mathrm{AD}_t  &= \sum_{i=0}^{t} \mathrm{MFV}_i = \mathrm{AD}_{t-1} + \mathrm{MFV}_t.

    The line is unbounded and its absolute level is arbitrary (it depends on where the cumulative sum starts); only its
    slope and divergences from price are interpreted. There is no moving window: every bar from the start of the series
    contributes to the running total.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        volume: Traded-volume series (e.g. ``pl.col("volume")``).

    Returns:
        The Accumulation/Distribution Line for each row, the same length as the inputs. There is no warm-up: the first
        row already carries the first bar's Money Flow Volume, and the line is the running cumulative sum from there.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        **Zero-range bars:**

        On a doji bar (``high == low``) the Money Flow Multiplier is ``0`` by convention, so the denominator never hits
        ``0 / 0`` and ``close`` does not enter the bar's contribution.

        The zero-range convention applies only to a genuine equal-range bar (``high == low``), where the multiplier is
        ``0`` and ``close`` does not enter the contribution. A ``null`` or ``NaN`` in any input instead leaves the range
        ``null`` or ``NaN`` (never ``== 0``), so missing data propagates rather than being silently zeroed.

        **Edge-case behavior:**

        - **Null** — a row in which ``high``, ``low``, ``close``, or ``volume`` is ``null`` yields ``null`` at that
          position and leaves the running total untouched (the cumulative sum skips it and continues from the prior
          total). On a genuine doji bar (``high == low``, both finite) the multiplier is ``0`` and ``close`` is
          irrelevant, so a ``null`` in ``close`` on such a bar still yields ``0`` rather than ``null``.
        - **NaN** — a ``NaN`` in any operand that reaches the cumulative sum latches: once present, every later non-null
          row of the line is ``NaN``. A bar whose ``high`` and ``low`` are both ``NaN`` does **not** take the doji
          branch (``NaN - NaN`` is ``NaN``, never ``== 0``), so the ``NaN`` poisons the line rather than contributing
          ``0``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the cumulative sum restarts per
          series and never spans series boundaries, e.g.
          ``accumulation_distribution(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")).over("ticker")``

    See Also:
        - :func:`accumulation_distribution_oscillator`: The Chaikin oscillator — fast minus slow EMA of this line.
        - :func:`chaikin_money_flow`: The windowed money-flow ratio over the same multiplier.
        - :func:`obv`: Another cumulative volume-flow line.

    References:
        - Chaikin, Marc. "Accumulation/Distribution Line".
        - https://en.wikipedia.org/wiki/Accumulation/distribution_index
        - https://www.investopedia.com/terms/a/accumulationdistribution.asp

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import accumulation_distribution
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 13.0, 14.0],
        ...         "low": [8.0, 9.0, 10.0, 11.0, 12.0],
        ...         "close": [9.0, 10.5, 10.0, 13.0, 12.5],
        ...         "volume": [100.0, 200.0, 300.0, 400.0, 500.0],
        ...     }
        ... )
        >>> frame.select(
        ...     accumulation_distribution(
        ...         pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")
        ...     ).round(4).alias("ad")
        ... )["ad"].to_list()
        [0.0, 100.0, -200.0, 200.0, -50.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 4 + ["B"] * 4,
        ...         "high": [12.0, 13.0, 12.5, 14.0, 22.0, 24.0, 23.0, 25.0],
        ...         "low": [10.0, 11.0, 11.0, 12.0, 20.0, 21.0, 21.0, 23.0],
        ...         "close": [11.0, 12.5, 11.5, 13.5, 21.5, 21.5, 22.5, 24.0],
        ...         "volume": [100.0, 120.0, 90.0, 110.0, 100.0, 120.0, 90.0, 110.0],
        ...     }
        ... )
        >>> expr = accumulation_distribution(
        ...     pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")
        ... ).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("accumulation_distribution"))["accumulation_distribution"].to_list()
        [0.0, 60.0, 30.0, 85.0, 50.0, -30.0, 15.0, 15.0]

        A ``null`` (skipped, the running total carrying across it) and a ``NaN`` (which propagates) make the
        exact handling visible at a glance:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        ...         "low": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        ...         "close": [11.5, 12.5, 13.0, 14.5, None, 16.0, float("nan"), 18.0],
        ...         "volume": [100.0, 120.0, 90.0, 110.0, 130.0, 100.0, 95.0, 140.0],
        ...     }
        ... )
        >>> expr = accumulation_distribution(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")).round(4)
        >>> frame.select(expr.alias("accumulation_distribution"))["accumulation_distribution"].to_list()
        [50.0, 110.0, 110.0, 165.0, None, 165.0, nan, nan]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    volume = float64_expr(volume)
    # Doji guard: high - low == 0 pins MFM to 0 (avoids 0/0). A null/NaN input makes the range null/NaN (not == 0), so
    # missing data propagates to a null/NaN money-flow volume rather than being silently zeroed.
    high_low_range = high - low
    money_flow_multiplier = (
        pl.when(high_low_range == 0).then(pl.lit(0.0)).otherwise(((close - low) - (high - close)) / high_low_range)
    )
    money_flow_volume = money_flow_multiplier * volume
    return money_flow_volume.cum_sum().name.keep()


def accumulation_distribution_oscillator(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    volume: pl.Expr,
    *,
    window_fast: int,
    window_slow: int,
) -> pl.Expr:
    r"""
    Accumulation/Distribution Oscillator, also known as the Chaikin Oscillator.

    Marc Chaikin's volume-momentum oscillator: the gap between a fast and a slow :func:`ema` of the
    :func:`accumulation_distribution` line, so it measures the momentum of accumulation rather than its level and
    oscillates around zero. With :math:`\mathrm{AD}` the accumulation/distribution line and :math:`n_f`, :math:`n_s` the
    fast and slow spans:

    .. math::

        \mathrm{ADOSC}_t = \mathrm{EMA}(\mathrm{AD}, n_f)_t - \mathrm{EMA}(\mathrm{AD}, n_s)_t.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        volume: Traded-volume series (e.g. ``pl.col("volume")``).
        window_fast: Span of the fast EMA (canonically ``3``). Must be ``>= 1``.
        window_slow: Span of the slow EMA (canonically ``10``). Must be ``>= 1`` and ``>= window_fast``.

    Returns:
        The oscillator for each row, the same length as the inputs. The first ``window_slow - 1`` values are ``null``
        (warm-up), inherited from the slow :func:`ema` of the accumulation/distribution line, the later of the two to
        warm up.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window_fast < 1``, ``window_slow < 1``, or ``window_fast > window_slow``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        **Scaling:** homogeneous of degree ``1`` — the accumulation/distribution multiplier is scale-invariant in price
        while the line scales with ``volume``, so multiplying all four inputs by ``k`` scales the oscillator by ``k``.

        **Edge-case behavior:**

        - **Null** — a ``null`` is skipped and the accumulation/distribution line and its EMAs bridge the gap (only a
          ``NaN`` latches).
        - **NaN** — a ``NaN`` propagates through the line and the EMAs, yielding ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the running sum and the EMAs do
          not span series boundaries.

    See Also:
        - :func:`accumulation_distribution`: The line this oscillates.
        - :func:`ema`: The exponential moving average of the two smoothings.
        - :func:`macd`: The same fast-minus-slow-EMA oscillator, applied to price.

    References:
        - Chaikin, Marc. "Chaikin Oscillator".
        - https://www.investopedia.com/terms/c/chaikinoscillator.asp

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import accumulation_distribution_oscillator
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.2, 10.5, 10.7, 10.3, 10.8],
        ...         "low": [9.8, 10.0, 10.2, 9.9, 10.3],
        ...         "close": [10.0, 10.3, 10.5, 10.1, 10.6],
        ...         "volume": [100.0, 150.0, 120.0, 200.0, 180.0],
        ...     }
        ... )
        >>> expr = accumulation_distribution_oscillator(
        ...     pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), window_fast=2, window_slow=3
        ... ).round(4)
        >>> frame.select(expr.alias("adosc"))["adosc"].to_list()
        [None, None, 13.0, 8.6667, 11.0556]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 4 + ["B"] * 4,
        ...         "high": [12.0, 13.0, 12.5, 14.0, 22.0, 24.0, 23.0, 25.0],
        ...         "low": [10.0, 11.0, 11.0, 12.0, 20.0, 21.0, 21.0, 23.0],
        ...         "close": [11.0, 12.5, 11.5, 13.5, 21.5, 21.5, 22.5, 24.0],
        ...         "volume": [100.0, 120.0, 90.0, 110.0, 100.0, 120.0, 90.0, 110.0],
        ...     }
        ... )
        >>> expr = (
        ...     accumulation_distribution_oscillator(
        ...         pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), window_fast=2, window_slow=3
        ...     )
        ...     .over("ticker")
        ...     .round(4)
        ... )
        >>> frame.with_columns(expr.alias("adosc"))["adosc"].to_list()
        [None, None, 0.0, 9.1667, None, None, 1.6667, 1.1111]

        A ``null`` (which voids the line and its EMAs) and a ``NaN`` (which propagates) make the exact handling
        visible at a glance:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        ...         "low": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        ...         "close": [11.5, 12.5, 13.0, 14.5, None, 16.0, float("nan"), 18.0],
        ...         "volume": [100.0, 120.0, 90.0, 110.0, 130.0, 100.0, 95.0, 140.0],
        ...     }
        ... )
        >>> expr = accumulation_distribution_oscillator(
        ...     pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), window_fast=2, window_slow=3
        ... ).round(4)
        >>> frame.select(expr.alias("adosc"))["adosc"].to_list()
        [None, None, 10.0, 15.8333, None, 9.4048, nan, nan]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    volume = float64_expr(volume)
    validate_window(window_fast, name="window_fast")
    validate_window(window_slow, name="window_slow")
    validate_window_order(window_fast, window_slow)
    line = accumulation_distribution(high, low, close, volume)
    return (ema(line, window_fast) - ema(line, window_slow)).name.keep()


def chaikin_money_flow(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    volume: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Chaikin Money Flow (CMF).

    A volume-weighted breadth oscillator (Marc Chaikin) that gauges buying versus selling pressure over a rolling
    window. Each bar is first scored by where its ``close`` sits inside the bar's range via the Money Flow Multiplier
    :math:`\mathrm{MFM}`, scaled by ``volume`` into the Money Flow Volume :math:`\mathrm{MFV}`,
    and the CMF is the ratio of the windowed sums of money-flow volume to raw volume:

    .. math::

        \mathrm{MFM}_t &= \frac{(C_t - L_t) - (H_t - C_t)}{H_t - L_t}, \\
        \mathrm{MFV}_t &= \mathrm{MFM}_t \, V_t, \\
        \mathrm{CMF}_t &= \frac{\sum_{i=0}^{n-1} \mathrm{MFV}_{t-i}}{\sum_{i=0}^{n-1} V_{t-i}},
        \qquad n = \text{window},

    where :math:`H`, :math:`L`, :math:`C`, :math:`V` are ``high``, ``low``, ``close``, ``volume``. The multiplier lives
    in :math:`[-1, +1]`: it is :math:`+1` when the close prints at the high (maximum buying pressure), :math:`-1` at the
    low (maximum selling pressure), and :math:`0` at the midpoint. A zero-range bar (:math:`H_t = L_t`) has an undefined
    multiplier, so by convention :math:`\mathrm{MFM}_t = 0` there — that bar contributes nothing to the numerator while
    its volume still counts in the denominator. Because the CMF is a volume-weighted average of multipliers, it is
    itself bounded in :math:`[-1, +1]`.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        volume: Traded-volume series (e.g. ``pl.col("volume")``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        The CMF for each row, the same length as the inputs. The first ``window - 1`` values are ``null`` (warm-up): the
        value is defined only once a full window of bars is available.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        **Zero-range bars:**

        The zero-range convention applies only to a genuine equal-range bar (``high == low``), where the multiplier is
        ``0`` (it adds ``0`` to the numerator while its volume still counts in the denominator). A ``null`` or ``NaN``
        in any input instead leaves that bar's money-flow volume ``null`` or ``NaN``, so missing data propagates
        rather than being silently zeroed.

        **Edge-case behavior:**

        - **Null** — a window in which any of ``high`` / ``low`` / ``close`` / ``volume`` contains a ``null`` yields
          ``null``; ``null`` takes precedence over ``NaN``.
        - **NaN** — a window containing a ``NaN`` (and no ``null``) yields ``NaN``.
        - **Zero volume** — a window whose volume is all zero is the ``0 / 0`` degenerate; the window is detected
          exactly (the rolling maximum of the absolute volume is zero) and the result is ``NaN``, not the rounding noise
          a sub-ULP residual in the rolling-sum denominator would otherwise produce. With non-negative volume this is
          the only reachable division-by-zero case, since an all-zero volume window also zeroes the numerator.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so neither rolling sum spans
          series boundaries, e.g.
          ``chaikin_money_flow(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), 20).over("ticker")``.

    See Also:
        - :func:`accumulation_distribution`: The cumulative (window-less) money-flow line.
        - :func:`money_flow_index`: A bounded windowed money-flow oscillator.
        - :func:`accumulation_distribution_oscillator`: Chaikin's momentum oscillator over the same line.

    References:
        - Chaikin, Marc. "Chaikin Money Flow".
        - https://en.wikipedia.org/wiki/Chaikin_Analytics#Chaikin_Money_Flow
        - https://www.investopedia.com/terms/c/chaikinmoneyflow.asp

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import chaikin_money_flow
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 12.0, 11.0, 13.0, 14.0],
        ...         "low": [8.0, 9.0, 9.0, 10.0, 11.0],
        ...         "close": [9.0, 11.0, 10.0, 12.0, 13.0],
        ...         "volume": [100.0, 200.0, 150.0, 300.0, 250.0],
        ...     }
        ... )
        >>> frame.select(
        ...     chaikin_money_flow(
        ...         pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), window=3
        ...     ).round(4).alias("cmf_3")
        ... )["cmf_3"].to_list()
        [None, None, 0.1481, 0.2564, 0.2619]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 4 + ["B"] * 4,
        ...         "high": [12.0, 13.0, 12.5, 14.0, 22.0, 24.0, 23.0, 25.0],
        ...         "low": [10.0, 11.0, 11.0, 12.0, 20.0, 21.0, 21.0, 23.0],
        ...         "close": [11.0, 12.5, 11.5, 13.5, 21.5, 21.5, 22.5, 24.0],
        ...         "volume": [100.0, 120.0, 90.0, 110.0, 100.0, 120.0, 90.0, 110.0],
        ...     }
        ... )
        >>> expr = chaikin_money_flow(
        ...     pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), 2
        ... ).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("chaikin_money_flow"))["chaikin_money_flow"].to_list()
        [None, 0.2727, 0.1429, 0.125, None, -0.1364, -0.1667, 0.225]

        A ``null`` (skipped, and any window it touches yields ``null``) and a ``NaN`` (which propagates) make the
        exact handling visible at a glance:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        ...         "low": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        ...         "close": [11.5, 12.5, 13.0, 14.5, None, 16.0, float("nan"), 18.0],
        ...         "volume": [100.0, 120.0, 90.0, 110.0, 130.0, 100.0, 95.0, 140.0],
        ...     }
        ... )
        >>> expr = chaikin_money_flow(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), 2).round(4)
        >>> frame.select(expr.alias("chaikin_money_flow"))["chaikin_money_flow"].to_list()
        [None, 0.5, 0.2857, 0.275, None, None, nan, nan]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    volume = float64_expr(volume)
    validate_window(window)
    # Doji guard: high - low == 0 pins MFM to 0 (avoids 0/0). A null/NaN input makes the range null/NaN (not == 0),
    # so missing data propagates to a null/NaN money-flow volume rather than being silently zeroed.
    high_low_range = high - low
    money_flow_multiplier = (
        pl.when(high_low_range == 0).then(pl.lit(0.0)).otherwise(((close - low) - (high - close)) / high_low_range)
    )
    money_flow_volume = money_flow_multiplier * volume
    weighted_sum = money_flow_volume.rolling_sum(window_size=window)
    raw = weighted_sum / volume.rolling_sum(window_size=window)
    # A window whose volume is all zero is the 0/0 degenerate: detect it exactly via the rolling maximum of the absolute
    # volume (which is exactly 0 only when every volume in the window is 0), so a sub-ULP residual left in the rolling
    # sum cannot fake a finite or infinite reading, and return NaN as documented. The ``& weighted_sum.is_not_null()``
    # gate preserves null precedence: a ``null`` in a price input nulls the weighted sum even though the all-zero
    # volume's rolling maximum is still ``0``, so such a window stays ``null`` rather than taking the guard's ``NaN``.
    # The clip pins the bound: the ratio is mathematically in [-1, 1] (so is the multiplier), so past a sane dynamic
    # range a residual-dominated near-zero-volume window degrades but stays in range (see CORRECTNESS.md).
    is_zero_volume = (volume.abs().rolling_max(window_size=window) == 0) & weighted_sum.is_not_null()
    return pl.when(is_zero_volume).then(float("nan")).otherwise(raw.clip(-1.0, 1.0)).name.keep()


def money_flow_index(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    volume: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Money Flow Index (MFI), also known as the volume-weighted Relative Strength Index.

    A bounded ``[0, 100]`` momentum oscillator (Quong & Soudack, 1989) that grades buying versus selling pressure by
    weighting each bar's typical price by its traded volume. It is the volume-aware analogue of the RSI: where the RSI
    accumulates price gains and losses, the MFI accumulates the *raw money flow* (typical price times volume) on up-days
    versus down-days. With :math:`n = \text{window}`, typical price :math:`\mathrm{TP}_t = (H_t + L_t + C_t) / 3` and
    raw money flow :math:`\mathrm{RMF}_t = \mathrm{TP}_t \cdot V_t`:

    .. math::

        \mathrm{PF}_t &= \mathrm{RMF}_t \,\mathbf{1}\!\left[\mathrm{TP}_t > \mathrm{TP}_{t-1}\right], \qquad
        \mathrm{NF}_t = \mathrm{RMF}_t \,\mathbf{1}\!\left[\mathrm{TP}_t < \mathrm{TP}_{t-1}\right], \\[4pt]
        \mathrm{MR}_t &= \frac{\sum_{i=0}^{n-1} \mathrm{PF}_{t-i}}{\sum_{i=0}^{n-1} \mathrm{NF}_{t-i}}, \qquad
        \mathrm{MFI}_t = 100 - \frac{100}{1 + \mathrm{MR}_t}.

    A bar whose typical price is unchanged (:math:`\mathrm{TP}_t = \mathrm{TP}_{t-1}`) contributes to neither the
    positive nor the negative money flow. The first row has no predecessor and so contributes no money flow, which is
    why a full ``window`` of *changes* (and therefore ``window + 1`` price bars) is needed before the first value is
    defined. As a volume-weighted RSI it is bounded in :math:`[0, 100]`: a window with no negative money flow saturates
    the oscillator at ``100`` and one with no positive money flow at ``0``.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        volume: Traded-volume series (e.g. ``pl.col("volume")``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        The MFI for each row, bounded in ``[0, 100]`` and the same length as the inputs. The first ``window`` values are
        ``null`` (warm-up): the value is defined only once ``window`` price *changes* have accumulated, so the first
        defined row is at index ``window`` rather than ``window - 1``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        **Classification:**

        The up / down classification compares each typical price with the previous one, so a missing or undefined
        typical price taints two consecutive change positions (its own and the following one).

        **Edge-case behavior:**

        - **Null** — a ``null`` in ``high``, ``low``, or ``close`` voids the typical price at that row and at the next
          change, so any window reaching either yields ``null``; a ``null`` in ``volume`` voids only that row's money
          flow. ``null`` takes precedence over ``NaN``.
        - **NaN** — a ``NaN`` in any input contaminates the affected money flow and yields ``NaN`` for every window that
          contains it. A ``NaN`` typical price makes both its own change and the next one undefined in sign, so each is
          poisoned into the positive *and* the negative money flow as ``NaN``, voiding every window that reaches either
          change (the same two-position taint as the ``null`` analogue, but surfaced as ``NaN`` rather than ``null``).
        - **Division by zero** — a window with no negative money flow but non-zero positive flow has money ratio
          ``+inf`` and the MFI saturates at ``100``; symmetrically an all-down window gives ``0``. A window in which
          both flows are zero (the typical price never moves) leaves the money ratio at ``0 / 0`` and yields ``NaN`` --
          the oscillator is genuinely undefined there.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so neither the difference nor the
          rolling sums span series boundaries, e.g.
          ``money_flow_index(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), 14).over("ticker")``.

    See Also:
        - :func:`rsi`: The price-only analogue (no volume weighting).
        - :func:`chaikin_money_flow`: Another volume-weighted money-flow oscillator.
        - :func:`price_typical`: The per-bar typical price this weights by volume.

    References:
        - Quong, Gene & Soudack, Avrum (1989). "Volume-Weighted RSI: Money Flow".
          *Technical Analysis of Stocks & Commodities*, 7(3).
        - https://en.wikipedia.org/wiki/Money_flow_index
        - https://www.investopedia.com/terms/m/mfi.asp

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import money_flow_index
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0],
        ...         "low": [8.0, 9.0, 10.0, 9.0, 11.0, 12.0, 11.0, 13.0],
        ...         "close": [9.0, 10.0, 11.0, 10.0, 12.0, 13.0, 12.0, 14.0],
        ...         "volume": [100.0, 150.0, 120.0, 130.0, 110.0, 160.0, 140.0, 170.0],
        ...     }
        ... )
        >>> frame.select(
        ...     money_flow_index(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), window=3)
        ...     .round(4)
        ...     .alias("mfi_3")
        ... )["mfi_3"].to_list()
        [None, None, None, 68.4466, 67.0051, 72.3404, 66.9291, 72.6384]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 4 + ["B"] * 4,
        ...         "high": [12.0, 13.0, 12.5, 14.0, 22.0, 24.0, 23.0, 25.0],
        ...         "low": [10.0, 11.0, 11.0, 12.0, 20.0, 21.0, 21.0, 23.0],
        ...         "close": [11.0, 12.5, 11.5, 13.5, 21.5, 21.5, 22.5, 24.0],
        ...         "volume": [100.0, 120.0, 90.0, 110.0, 100.0, 120.0, 90.0, 110.0],
        ...     }
        ... )
        >>> expr = (
        ...     money_flow_index(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), 2)
        ...     .over("ticker")
        ...     .round(4)
        ... )
        >>> frame.with_columns(expr.alias("money_flow_index"))["money_flow_index"].to_list()
        [None, None, 58.1673, 57.972, None, None, 100.0, 100.0]

        A ``null`` (skipped, and any window it touches yields ``null``) and a ``NaN`` (which propagates) make the
        exact handling visible at a glance:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
        ...         "low": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        ...         "close": [11.5, 12.5, 13.0, 14.5, None, 16.0, float("nan"), 18.0],
        ...         "volume": [100.0, 120.0, 90.0, 110.0, 130.0, 100.0, 95.0, 140.0],
        ...     }
        ... )
        >>> expr = money_flow_index(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), 2).round(4)
        >>> frame.select(expr.alias("money_flow_index"))["money_flow_index"].to_list()
        [None, None, 100.0, 100.0, None, None, None, nan]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    volume = float64_expr(volume)
    validate_window(window)
    typical_price = price_typical(high, low, close)
    raw_money_flow = typical_price * volume
    typical_change = typical_price.diff()
    # A NaN typical change is undefined in sign: route it to NaN in *both* flows so it poisons every window reaching the
    # successor change, matching the null analogue (which voids its own and the following change). Without this guard
    # Polars' total order (``NaN > 0`` is ``True``) would route the NaN change into the positive flow at its finite raw
    # money flow, faking a fully-positive bar after a NaN typical price.
    change_is_nan = typical_change.is_nan()
    positive_flow = (
        pl.when(change_is_nan)
        .then(pl.lit(float("nan")))
        .when(typical_change > 0.0)
        .then(raw_money_flow)
        .when(typical_change <= 0.0)
        .then(pl.lit(0.0))
        .otherwise(None)
    )
    negative_flow = (
        pl.when(change_is_nan)
        .then(pl.lit(float("nan")))
        .when(typical_change < 0.0)
        .then(raw_money_flow)
        .when(typical_change >= 0.0)
        .then(pl.lit(0.0))
        .otherwise(None)
    )
    money_ratio = positive_flow.rolling_sum(window_size=window) / negative_flow.rolling_sum(window_size=window)
    index = 100.0 - 100.0 / (1.0 + money_ratio)
    # Flat window: when every typical-price change is exactly zero both flows are zero, the documented 0/0 -> NaN.
    # Detect it exactly via the rolling maximum of the absolute change (zero only when every change is exactly 0), so a
    # sub-ULP residual left in the rolling sums after large flows slide out cannot fake a saturated reading. A null/NaN
    # change makes the rolling maximum null/NaN (never == 0), so the guard does not fire and the null-precedence and
    # NaN-poisoning above are unchanged. The clip pins the [0, 100] bound: beyond a sane dynamic range a residual-
    # dominated near-flat window degrades but stays in range rather than escaping (see CORRECTNESS.md).
    is_flat = typical_change.abs().rolling_max(window_size=window) == 0
    return pl.when(is_flat).then(float("nan")).otherwise(index.clip(0.0, 100.0)).name.keep()


def obv(
    expr: pl.Expr,
    volume: pl.Expr,
) -> pl.Expr:
    r"""
    On-Balance Volume (OBV), also known as Granville's cumulative volume.

    A momentum indicator that ties volume to price direction (Joseph Granville, 1963). Each bar adds the whole bar
    volume to a running total when the close rises, subtracts it when the close falls, and leaves the total unchanged
    when the close is flat; the cumulative line is read for divergences against price rather than for its absolute
    level:

    .. math::

        \mathrm{OBV}_t = \sum_{i=1}^{t} \operatorname{sign}(x_i - x_{i-1})\, V_i,
        \qquad
        \operatorname{sign}(d) =
        \begin{cases}
            +1, & d > 0, \\
            \phantom{+}0, & d = 0, \\
            -1, & d < 0,
        \end{cases}

    where :math:`x` is ``expr`` and :math:`V` is ``volume``. The first bar has no predecessor, so its direction is
    undefined; it contributes ``0`` and the series therefore starts at :math:`\mathrm{OBV}_0 = 0`.

    OBV is an unbounded, level-arbitrary cumulative series; only its slope and divergences against price are meaningful,
    not its absolute magnitude.

    Because the direction is the sign of the bar-to-bar close change, it is invariant to any additive shift of the price
    level and homogeneous of degree one in ``volume`` (scaling all volumes by a constant scales OBV by that constant).

    Args:
        expr: Input series, conventionally the close (any series is accepted; e.g. ``pl.col("close")``).
        volume: Traded-volume series (e.g. ``pl.col("volume")``).

    Returns:
        The OBV for each row, the same length as the inputs. There is no window and no warm-up: every row is defined,
        starting at ``0`` on the first row.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        **Edge-case behavior:**

        - **Null** — a ``null`` close zeroes the direction both at its own row and at the following row (each ``diff``
          touching the ``null`` is itself ``null`` and is filled to ``0``), so those bars contribute nothing while the
          running total carries on. A ``null`` volume makes that bar's contribution ``null`` (``0 * null`` is ``null``,
          so this holds even on the first or a flat bar where the direction is ``0``): the output is ``null`` at exactly
          that row while the cumulative sum skips it and continues from the prior total.
        - **NaN** — a ``NaN`` close (via ``diff``) or a ``NaN`` volume poisons the contribution at its row and, once
          summed, latches the running total to ``NaN`` for every subsequent row; because ``0 * NaN`` is ``NaN`` under
          IEEE-754, a ``NaN`` volume contaminates the total even on a flat or first bar where the direction is ``0``. A
          ``null``-contribution row still emits ``null`` at its own position even after the latch.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so neither the ``diff`` nor the
          cumulative sum spans series boundaries, e.g. ``obv(pl.col("close"), pl.col("volume")).over("ticker")``.

    See Also:
        - :func:`accumulation_distribution`: Another cumulative volume line.
        - :func:`money_flow_index`: A bounded volume-weighted oscillator.
        - :func:`chaikin_money_flow`: A windowed volume-weighted money-flow ratio.

    References:
        - Granville, Joseph E. (1963). *Granville's New Key to Stock Market Profits*. Prentice-Hall.
        - https://en.wikipedia.org/wiki/On-balance_volume
        - https://www.investopedia.com/terms/o/onbalancevolume.asp

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import obv
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "close": [10.0, 12.0, 11.0, 11.0, 13.0, 9.0],
        ...         "volume": [100.0, 200.0, 150.0, 80.0, 300.0, 250.0],
        ...     }
        ... )
        >>> frame.select(obv(pl.col("close"), pl.col("volume")).round(4).alias("obv"))["obv"].to_list()
        [0.0, 200.0, 50.0, 50.0, 350.0, 100.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 4 + ["B"] * 4,
        ...         "close": [11.0, 12.5, 11.5, 13.5, 21.5, 21.5, 22.5, 24.0],
        ...         "volume": [100.0, 120.0, 90.0, 110.0, 100.0, 120.0, 90.0, 110.0],
        ...     }
        ... )
        >>> expr = obv(pl.col("close"), pl.col("volume")).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("obv"))["obv"].to_list()
        [0.0, 120.0, 30.0, 140.0, 0.0, 0.0, 90.0, 200.0]

        A ``null`` (skipped, the running total carrying across it) and a ``NaN`` (which propagates) make the
        exact handling visible at a glance:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "close": [10.0, 11.0, 12.0, 13.0, None, 15.0, float("nan"), 17.0, 18.0, 19.0],
        ...         "volume": [100.0, 120.0, 90.0, 110.0, 130.0, 100.0, 95.0, 140.0, 105.0, 115.0],
        ...     }
        ... )
        >>> expr = obv(pl.col("close"), pl.col("volume")).round(4)
        >>> frame.select(expr.alias("obv"))["obv"].to_list()
        [0.0, 120.0, 210.0, 320.0, 320.0, 320.0, nan, nan, nan, nan]
    """
    expr = float64_expr(expr)
    volume = float64_expr(volume)
    # Bar-local direction + one cumulative sum — no path-dependent recursion, so no Rust kernel is needed.
    direction = expr.diff().sign().fill_null(0)
    return (direction * volume).cum_sum()


def vwap(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    volume: pl.Expr,
) -> pl.Expr:
    r"""
    Volume-Weighted Average Price (VWAP) — the running volume-weighted mean of the typical price.

    The benchmark intraday price: each bar's typical price (:func:`price_typical`) weighted by its volume, accumulated
    from the start of the partition. It anchors to that start, so the canonical use is one session per ``.over(...)``
    group -- an un-anchored VWAP over years of data is rarely meaningful:

    .. math::

        \mathrm{VWAP}_t = \frac{\sum_{i \le t} \mathrm{typical}_i \, V_i}{\sum_{i \le t} V_i},
            \quad V_i = \mathrm{volume}_i,
            \quad \mathrm{typical}_i = \frac{\mathrm{high}_i + \mathrm{low}_i + \mathrm{close}_i}{3}.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        volume: Traded-volume series (e.g. ``pl.col("volume")``).

    Returns:
        The running VWAP for each row, the same length as the inputs. There is no warm-up: row ``0`` is defined as soon
        as its cumulative volume is positive (a leading zero-volume run reads ``NaN`` until volume accrues).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the conditioning limit of
        the long cumulative sums beyond it.

        **Anchoring:** VWAP accumulates from the start of the partition, so wrap the call in ``.over(session_key)`` to
        reset it per session (e.g. one trading day): ``vwap(...).over("session")``. Without an anchor it accumulates
        across the whole series, the classic VWAP misuse.

        **Inputs:**

        ``high`` / ``low`` / ``close`` / ``volume`` must share a length and alignment; ``volume`` is expected
        non-negative (a negative volume is summed as-is, with no guard -- garbage in, garbage out).

        **Edge-case behavior:**

        - **Zero volume** — at the head a zero cumulative volume gives ``0 / 0 == NaN`` until volume accrues; an
          interior zero-volume bar adds nothing (the prefix sums carry forward, with no subtract-on-exit residual).
        - **Null** — a ``null`` in any input nulls that bar's contribution at its own row; both cumulative sums skip the
          bar together (a ``null`` price input drops its volume from the denominator too), so the bar is a clean
          missing observation, not a denominator-only contribution.
        - **NaN** — a ``NaN`` in any input poisons the cumulative sum from its row onward (it cannot be subtracted out).
        - **Partitioning** — see Anchoring above; ``.over(...)`` is the intended use, not an afterthought.

    See Also:
        - :func:`vwma`: The windowed volume-weighted moving average, for a rolling rather than anchored weight.
        - :func:`price_typical`: The per-bar price this weights.
        - :func:`sma`: The equal-weighted moving average, the volume-blind analogue.

    References:
        - https://en.wikipedia.org/wiki/Volume-weighted_average_price
        - https://www.investopedia.com/terms/v/vwap.asp

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import vwap
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [2.0, 4.0, 6.0],
        ...         "low": [0.0, 2.0, 4.0],
        ...         "close": [1.0, 3.0, 5.0],
        ...         "volume": [10.0, 20.0, 30.0],
        ...     }
        ... )
        >>> expr = vwap(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"))
        >>> frame.select(expr.round(4).alias("vwap"))["vwap"].to_list()
        [1.0, 2.3333, 3.6667]

        Anchor per session with ``.over`` so each day's VWAP restarts:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "session": ["a", "a", "b", "b"],
        ...         "high": [2.0, 4.0, 12.0, 14.0],
        ...         "low": [0.0, 2.0, 10.0, 12.0],
        ...         "close": [1.0, 3.0, 11.0, 13.0],
        ...         "volume": [10.0, 20.0, 10.0, 20.0],
        ...     }
        ... )
        >>> expr = vwap(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")).over("session")
        >>> frame.select(expr.round(4).alias("vwap"))["vwap"].to_list()
        [1.0, 2.3333, 11.0, 12.3333]

        A ``null`` (yields ``null`` at that row) and a ``NaN`` (which latches in the running totals) make it visible:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        ...         "low": [8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
        ...         "close": [9.0, 10.0, None, 12.0, float("nan"), 14.0],
        ...         "volume": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        ...     }
        ... )
        >>> expr = vwap(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"))
        >>> frame.select(expr.round(4).alias("vwap"))["vwap"].to_list()
        [9.0, 9.6667, None, 11.0, nan, nan]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    volume = float64_expr(volume)
    # Cumulative typical-price-times-volume over cumulative volume; anchors to the partition start (use .over). A null
    # in a price input nulls the bar's weighted term, so the denominator must drop that bar's volume too -- both sums
    # skip the bar together, so a null is a clean missing observation rather than a denominator-only contribution.
    typical = price_typical(high, low, close)
    weighted = typical * volume
    volume_masked = pl.when(weighted.is_null()).then(None).otherwise(volume)
    return weighted.cum_sum() / volume_masked.cum_sum()
