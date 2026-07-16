"""
Trend indicators — the Parabolic SAR (stop and reverse) and the SuperTrend (ATR-band ratchet).
"""

import math
from functools import partial
from typing import Final

import polars as pl

from pomata._expr import float64_expr, validate_positive, validate_unit_fraction, validate_window
from pomata.indicators.price_transform import price_median
from pomata.indicators.volatility import atr

__all__ = ("parabolic_sar", "supertrend")

_MINIMUM_SEED_BARS: Final = 2

_SUPERTREND_DTYPE: Final = pl.Struct({"line": pl.Float64, "direction": pl.Float64})


def _sar_kernel(
    series: pl.Series,
    acceleration: float,
    maximum: float,
) -> pl.Series:
    """
    Sequential Parabolic SAR recurrence over one Series of ``{high, low}`` structs (the pure-Python kernel).

    Seeds the trend from the first two valid bars, then trails the stop with ``sar += af * (ep - sar)``, clamped to the
    two prior valid extremes, accelerating ``af`` on each new extreme and flipping the trend when price crosses the
    stop. A bar with a ``null`` / ``NaN`` high or low yields ``null`` / ``NaN`` and is skipped (the running state
    bridges it).
    """
    highs: list[float | None] = series.struct.field("high").to_list()
    lows: list[float | None] = series.struct.field("low").to_list()
    result: list[float | None] = [None] * len(highs)
    recent_high: list[float] = []
    recent_low: list[float] = []
    is_long = True
    sar = 0.0
    extreme = 0.0
    factor = acceleration
    seeded = False
    for index in range(len(highs)):
        high_value = highs[index]
        low_value = lows[index]
        if high_value is None or low_value is None:
            continue
        if math.isnan(high_value) or math.isnan(low_value):
            result[index] = math.nan
            continue
        if not seeded:
            recent_high.append(high_value)
            recent_low.append(low_value)
            if len(recent_high) < _MINIMUM_SEED_BARS:
                continue
            is_long = (recent_high[1] - recent_high[0]) >= (recent_low[0] - recent_low[1])
            sar = recent_low[0] if is_long else recent_high[0]
            extreme = recent_high[1] if is_long else recent_low[1]
            result[index] = sar
            seeded = True
            continue
        sar = sar + factor * (extreme - sar)
        if is_long:
            sar = min(sar, recent_low[-1], recent_low[-2])
            if high_value > extreme:
                extreme = high_value
                factor = min(factor + acceleration, maximum)
            if low_value <= sar:
                is_long = False
                sar = extreme
                extreme = low_value
                factor = acceleration
        else:
            sar = max(sar, recent_high[-1], recent_high[-2])
            if low_value < extreme:
                extreme = low_value
                factor = min(factor + acceleration, maximum)
            if high_value >= sar:
                is_long = True
                sar = extreme
                extreme = high_value
                factor = acceleration
        result[index] = sar
        recent_high = [*recent_high[-1:], high_value]
        recent_low = [*recent_low[-1:], low_value]
    return pl.Series(result, dtype=pl.Float64)


def parabolic_sar(
    high: pl.Expr,
    low: pl.Expr,
    *,
    acceleration: float = 0.02,
    maximum: float = 0.20,
) -> pl.Expr:
    r"""
    Parabolic SAR (Stop And Reverse) — a trailing stop that accelerates toward the extreme and reverses on a cross.

    Introduced by J. Welles Wilder (1978): a trend-following stop that trails price, tightening as a trend extends and
    flipping to the other side when price crosses it. Each bar the stop steps a fraction — the acceleration factor --
    of the way toward the trend's extreme point:

    .. math::

        \mathrm{SAR}_t = \mathrm{SAR}_{t-1} + \mathrm{AF}_{t-1} \, (\mathrm{EP}_{t-1} - \mathrm{SAR}_{t-1}),

    where ``EP`` is the highest high of the current up-trend (or lowest low of a down-trend) and ``AF`` starts at
    ``acceleration``, rising by ``acceleration`` on each new extreme up to ``maximum``. In an up-trend the stop is held
    at or below the prior two lows; when a low crosses it the trend reverses, the stop jumps to the prior extreme,
    ``EP`` resets to the new low, and ``AF`` resets (symmetrically for a down-trend).

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        acceleration: Starting acceleration factor, and its per-extreme increment (canonical default ``0.02``,
            Wilder's step). Must be in the half-open interval ``(0, 1]``, and never
            above ``maximum`` (so the factor is capped from the seed onward, not only on the increment path).
        maximum: Cap on the acceleration factor. Must be in the half-open interval ``(0, 1]``, and at least
            ``acceleration``.

    Returns:
        The Parabolic SAR for each row, the same length as the inputs. Row ``0`` is ``null`` (the trend is seeded from
        the first two bars); the value at row ``1`` is the seed stop, and the recurrence runs from there.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``acceleration`` or ``maximum`` is not in the half-open interval ``(0, 1]``, or if
            ``acceleration > maximum``.

    Note:
        **Precision:**
        The parabolic SAR is a path-dependent stop-and-reverse recurrence, so its reference oracle necessarily mirrors
        the implementation's state machine and confirms internal consistency, not independence; the independent witness
        is the set of golden masters hand-computed from Wilder's published rules. Agreement holds to ten significant
        figures (a ``1e-10`` band) on any finite input within a sane dynamic range; the documentation's *Correctness*
        page gives the method and the float-conditioning limit beyond it.

        It is homogeneous of degree ``1`` under a positive common rescaling of ``high`` and ``low`` (the stop is a price
        level and the recurrence and crossings are linear in price).

        **Seeding:**

        Wilder's original leaves the initial trend unspecified; here it is taken long when the first bar-to-bar up-move
        is at least the down-move, else short, and the first stop is the prior low (long) or high (short).

        **Edge-case behavior:**

        - **Null** — a ``null`` price makes that row ``null`` (``null`` takes precedence over ``NaN``) — the running
          trend state bridges the gap and resumes on the next complete bar, later rows reconverging as the stop
          recurrence contracts.
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there — the raw high/low feed the kernel
          directly with no recurrence to latch onto, so the running trend state bridges the gap and resumes on the next
          complete bar.
        - **Partitioning** — wrap the call in ``.over(...)`` so the window never spans series boundaries, e.g.
          ``parabolic_sar(pl.col("high"), pl.col("low")).over("ticker")``.

    See Also:
        - :func:`supertrend`: The other trailing-stop trend tool, ATR-scaled rather than accelerating.
        - :func:`adx`: Wilder's directional-movement trend-strength index.
        - :func:`atr`: Wilder's volatility average.

    References:
        - Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
        - https://en.wikipedia.org/wiki/Parabolic_SAR

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import parabolic_sar
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0, 10.0, 11.0],
        ...         "low": [9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0, 9.0, 10.0],
        ...     }
        ... )
        >>> frame.select(parabolic_sar(pl.col("high"), pl.col("low")).round(4).alias("sar"))["sar"].to_list()
        [None, 9.0, 9.0, 9.12, 9.3528, 9.7246, 10.0666, 14.0, 13.92, 13.7232]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker seeds independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "high": [10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 22.0, 21.0, 20.0],
        ...         "low": [9.0, 10.0, 11.0, 12.0, 13.0, 19.0, 20.0, 21.0, 20.0, 19.0],
        ...     }
        ... )
        >>> expr = parabolic_sar(pl.col("high"), pl.col("low")).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("sar"))["sar"].to_list()
        [None, 9.0, 9.0, 9.12, 9.3528, None, 19.0, 19.0, 19.12, 22.0]

        A ``null`` then a ``NaN`` in ``high`` each yield ``null`` / ``NaN`` at that row and are skipped, the
        running trend state bridging the gap:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, None, 14.0, float("nan"), 12.0, 11.0],
        ...         "low": [9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0],
        ...     }
        ... )
        >>> frame.select(parabolic_sar(pl.col("high"), pl.col("low")).round(4).alias("sar"))["sar"].to_list()
        [None, 9.0, 9.0, None, 9.12, nan, 9.4128, 9.688]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    validate_unit_fraction(acceleration, "acceleration")
    validate_unit_fraction(maximum, "maximum")
    if acceleration > maximum:
        raise ValueError(f"acceleration must be <= maximum, got acceleration={acceleration}, maximum={maximum}")
    return (
        pl.struct(high=high, low=low).map_batches(
            partial(_sar_kernel, acceleration=acceleration, maximum=maximum), return_dtype=pl.Float64
        )
    ).name.keep()


def _supertrend_kernel(
    series: pl.Series,
) -> pl.Series:
    """
    Sequential SuperTrend recurrence over one Series of ``{basic_upper, basic_lower, close}`` structs (the kernel).

    Ratchets the two final bands -- the upper can only fall while price holds below it, the lower can only rise while
    price holds above it -- and flips the line between them when ``close`` strictly crosses it, carrying the
    direction (``+1.0`` up, ``-1.0`` down). The trend seeds short when the first valid close sits at or below the lower
    band, else long. A bar with a ``null`` / ``NaN`` input yields ``null`` / ``NaN`` on both fields and is skipped (the
    running state, and the last valid close the ratchet reads, bridge it).
    """
    uppers: list[float | None] = series.struct.field("basic_upper").to_list()
    lowers: list[float | None] = series.struct.field("basic_lower").to_list()
    closes: list[float | None] = series.struct.field("close").to_list()
    line: list[float | None] = [None] * len(closes)
    direction: list[float | None] = [None] * len(closes)
    final_upper = 0.0
    final_lower = 0.0
    trend = 1.0
    previous_close = 0.0
    seeded = False
    for index in range(len(closes)):
        upper_value = uppers[index]
        lower_value = lowers[index]
        close_value = closes[index]
        if upper_value is None or lower_value is None or close_value is None:
            continue
        if math.isnan(upper_value) or math.isnan(lower_value) or math.isnan(close_value):
            line[index] = math.nan
            direction[index] = math.nan
            continue
        if not seeded:
            final_upper = upper_value
            final_lower = lower_value
            trend = -1.0 if close_value <= final_lower else 1.0
            line[index] = final_upper if trend < 0.0 else final_lower
            direction[index] = trend
            previous_close = close_value
            seeded = True
            continue
        if upper_value < final_upper or previous_close > final_upper:
            final_upper = upper_value
        if lower_value > final_lower or previous_close < final_lower:
            final_lower = lower_value
        if trend > 0.0:
            if close_value < final_lower:
                trend = -1.0
                line[index] = final_upper
            else:
                line[index] = final_lower
        elif close_value > final_upper:
            trend = 1.0
            line[index] = final_lower
        else:
            line[index] = final_upper
        direction[index] = trend
        previous_close = close_value
    return pl.DataFrame(
        {"line": pl.Series(line, dtype=pl.Float64), "direction": pl.Series(direction, dtype=pl.Float64)}
    ).to_struct()


def supertrend(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    window: int,
    *,
    multiplier: float = 3.0,
) -> pl.Expr:
    r"""
    SuperTrend — an ATR-scaled band around the median price, ratcheted into a stop that flips on a close cross.

    Introduced by Olivier Seban: an ATR-scaled trailing band that follows price on one side and flips to the other when
    a close crosses it, so the line reads as a dynamic stop and the sign of its move as the prevailing trend. Each bar
    sets a basic band around the median price by a multiple of the Average True Range,

    .. math::

        \mathrm{upper}_t = \frac{\mathrm{high}_t + \mathrm{low}_t}{2} + m \, \mathrm{ATR}_t, \qquad
        \mathrm{lower}_t = \frac{\mathrm{high}_t + \mathrm{low}_t}{2} - m \, \mathrm{ATR}_t,

    then ratchets a *final* band from it: the final upper only falls while the prior close stays below it, the final
    lower only rises while the prior close stays above it. The line tracks the final lower in an up-trend and the final
    upper in a down-trend, flipping -- and switching ``direction`` between ``+1`` and ``-1`` -- when a close strictly
    crosses the active band.

    Args:
        high: High-price series (e.g. ``pl.col("high")``).
        low: Low-price series (e.g. ``pl.col("low")``).
        close: Close-price series (e.g. ``pl.col("close")``).
        window: Number of observations in the ATR moving window (canonically ``10``). Must be ``>= 1``.
        multiplier: Band half-width as a multiple of the ATR (canonically ``3.0``). Must be a finite number ``> 0`` (a
            non-positive multiplier would collapse or invert the bands).

    Returns:
        A struct ``pl.Expr`` with two ``Float64`` fields, the same length as the inputs:

        - ``line`` — the trailing stop.
        - ``direction`` — ``+1.0`` in an up-trend (the line below price); ``-1.0`` in a down-trend (the line above
          price).

        The first ``window - 1`` rows are ``null`` (the ATR's warm-up). Read a field with ``.struct.field("line")``
        or split both with ``.struct.unnest()``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1`` or ``multiplier`` is not a finite number ``> 0`` (i.e. ``<= 0``, ``NaN``, or
            ``±inf``).

    Note:
        **Precision:**
        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        The ``line`` is homogeneous of degree ``1`` under a positive common rescaling of ``high`` / ``low`` / ``close``
        (a price level), while ``direction`` is scale-invariant (the crossings compare like-scaled quantities).

        **Tie-break and seeding:**

        A flip needs a *strict* cross, so a close exactly on the active band holds the current trend; over a flat series
        the bands collapse onto the midpoint and the line tracks it. The trend seeds short when the first valid close is
        at or below the lower band, else long -- chosen so the line sits on the correct side of price from row one.

        **Edge-case behavior:**

        - **Null** — a leading ``null`` run stays ``null`` until the first non-null seed; an interior ``null`` yields
          ``null`` at that position while the recursion continues across the gap (on both struct fields, the running
          state and the last valid close the ratchet reads bridging it).
        - **NaN** — a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null position
          (on both struct fields; a later ``null`` row shows ``null`` there only — nothing flushes the poisoned state).
        - **window == 1** — the ATR has no memory term, so a ``NaN`` self-heals once the true range is finite again.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history, e.g. ``supertrend(pl.col("high"), pl.col("low"), pl.col("close")).over("ticker")``.

    See Also:
        - :func:`parabolic_sar`: The other trailing-stop trend tool, accelerating rather than ATR-scaled.
        - :func:`atr`: The volatility average that sets the band half-width.
        - :func:`keltner_channels`: The other ATR-scaled band envelope, centered on an EMA rather than ratcheting.

    References:
        - Seban, O. (2009). *Tout le monde mérite d'être riche*.

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import supertrend
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0],
        ...         "low": [9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0],
        ...         "close": [9.5, 10.8, 11.8, 10.2, 12.8, 11.2, 13.8],
        ...     }
        ... )
        >>> expr = supertrend(pl.col("high"), pl.col("low"), pl.col("close"), 2, multiplier=2.0)
        >>> out = frame.select(expr.alias("st")).unnest("st")
        >>> out.select(pl.col("line").round(4))["line"].to_list()
        [None, 8.0, 9.05, 9.05, 9.05, 9.05, 9.05]
        >>> out["direction"].to_list()
        [None, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker's ratchet warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "high": [10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 21.0, 22.0, 21.0, 23.0],
        ...         "low": [9.0, 10.0, 11.0, 10.0, 12.0, 19.0, 20.0, 21.0, 20.0, 22.0],
        ...         "close": [9.5, 10.8, 11.8, 10.2, 12.8, 19.5, 20.8, 21.8, 20.2, 22.8],
        ...     }
        ... )
        >>> expr = supertrend(pl.col("high"), pl.col("low"), pl.col("close"), 2, multiplier=2.0)
        >>> frame.with_columns(expr.over("ticker").struct.field("line").round(4).alias("l"))["l"].to_list()
        [None, 8.0, 9.05, 9.05, 9.05, None, 18.0, 19.05, 19.05, 19.05]

        A ``null`` in ``close`` is skipped and bridged by the running state, while a ``NaN`` poisons the ATR
        recursion and latches ``NaN`` thereafter:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "high": [10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0],
        ...         "low": [9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0],
        ...         "close": [9.5, 10.8, 11.8, None, 12.8, float("nan"), 13.8],
        ...     }
        ... )
        >>> expr = supertrend(pl.col("high"), pl.col("low"), pl.col("close"), 2, multiplier=2.0)
        >>> frame.select(expr.struct.field("line").round(4).alias("l"))["l"].to_list()
        [None, 8.0, 9.05, None, 9.9875, nan, nan]
    """
    high = float64_expr(high)
    low = float64_expr(low)
    close = float64_expr(close)
    validate_window(window)
    validate_positive(multiplier, "multiplier")
    half_width = multiplier * atr(high, low, close, window)
    middle = price_median(high, low)
    bands = pl.struct(basic_upper=middle + half_width, basic_lower=middle - half_width, close=close)
    return bands.map_batches(_supertrend_kernel, return_dtype=_SUPERTREND_DTYPE).name.keep()
