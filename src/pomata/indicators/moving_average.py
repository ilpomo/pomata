"""
Moving-average indicators (overlap studies).
"""

import math
from functools import partial

import polars as pl

from pomata._expr import float64_expr, validate_finite, validate_window, validate_window_order

__all__ = ("dema", "ema", "hma", "kama", "rma", "sma", "t3", "tema", "trima", "vwma", "wma")


def _seeded_recursive_mean_kernel(
    series: pl.Series,
    alpha: float,
    window: int,
) -> pl.Series:
    """
    Pure-Python kernel for one ``Float64`` batch of the SMA-seeded recurrence; the body of
    :func:`_seeded_recursive_mean`, run once per ``map_batches`` batch (per group under ``.over``).

    A single sequential pass: it counts non-null observations, seeds the running mean with the simple average of the
    first ``window`` of them on the ``window``-th non-null row, then applies the unadjusted recurrence
    ``y_t = (1 - alpha) * y_{t-1} + alpha * x_t``. An interior null yields null at its row while the running weight
    decays by ``1 - alpha`` across the gap (the ``ignore_nulls=False`` convention); a ``NaN`` latches and propagates.
    """
    decay = 1.0 - alpha
    result: list[float | None] = []
    average: float | None = None  # running mean; None until the seed is emitted
    weight = 1.0  # weight of the running mean since its last update, decaying across null gaps
    seed_total = 0.0  # sum of the first ``window`` non-null observations
    observed = 0  # non-null observations counted so far
    for value in series.to_list():
        if average is not None:
            weight *= decay
        if value is None:
            result.append(None)
            continue
        if average is None:
            observed += 1
            seed_total += value
            if observed < window:
                result.append(None)
            else:
                average = seed_total / window
                weight = 1.0
                result.append(average)
            continue
        average = (weight * average + alpha * value) / (weight + alpha)
        weight = 1.0
        result.append(average)
    return pl.Series(series.name, result, dtype=pl.Float64)


def _seeded_recursive_mean(
    expr: pl.Expr,
    alpha: float,
    window: int,
) -> pl.Expr:
    """
    Recursive exponential mean seeded with the simple average of the first ``window`` observations.

    The shared engine of :func:`ema` (``alpha = 2 / (window + 1)``) and :func:`rma` (``alpha = 1 / window``). The
    recurrence is the unadjusted ``y_t = (1 - alpha) * y_{t-1} + alpha * x_t``; the seed is the canonical SMA of the
    first ``window`` observations (Wilder's initialization, and the classical EMA initialization), so the warm-up
    matches the industry reference rather than the lighter first-observation seed. The seed lands on the ``window``-th
    non-null observation -- leading nulls skip the warm-up instead of consuming it.

    A path-dependent recurrence with a custom seed cannot be a single native expression without re-reading its input,
    which makes chained smoothers such as :func:`t3` blow up super-linearly; it therefore runs as a ``map_batches``
    kernel that consumes the series once. The pure-Python kernel is the portable form of this sequential recurrence; a
    compiled kernel is a future performance optimization.
    """
    return expr.map_batches(
        partial(_seeded_recursive_mean_kernel, alpha=alpha, window=window),
        return_dtype=pl.Float64,
    )


def dema(
    expr: pl.Expr,
    window: int,
    *,
    adjust: bool = False,
) -> pl.Expr:
    r"""
    Double Exponential Moving Average (DEMA), also known as Mulloy's DEMA.

    A lag-reduced moving average introduced by Patrick Mulloy (1994). Despite its name it is not an EMA applied twice
    but a linear combination of a single :func:`ema` and the :func:`ema` of that result, engineered to cancel most of
    the first-order lag while staying smoother than the raw series:

    .. math::

        \mathrm{DEMA}_t = 2\,\mathrm{EMA}(x)_t - \mathrm{EMA}\!\bigl(\mathrm{EMA}(x)\bigr)_t,
        \qquad \alpha = \frac{2}{\text{window} + 1}.

    Both exponential passes use the same ``window``. The inner term :math:`\mathrm{EMA}(\mathrm{EMA}(x))` is the EMA of
    the already-smoothed series.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Span of the exponential weighting, mapped to ``alpha = 2 / (window + 1)``. Must be ``>= 1``.
        adjust: When ``False`` (default) use the recursive technical-analysis EMA form; when ``True`` use the
            bias-corrected (adjusted) exponential weighting. The flag is forwarded unchanged to both :func:`ema` passes;
            the canonical DEMA uses ``False``.

    Returns:
        The DEMA for each row, the same length as ``expr``. The first ``2 * (window - 1)`` values are ``null``
        (warm-up), clamped to the series length: the value is composed from two chained :func:`ema` passes of the same
        ``window`` (each carrying a ``window - 1`` warm-up), so the warm-up is twice that of a plain EMA. Under the
        default ``adjust=False``, each pass is seeded with the SMA of the first ``window`` observations.

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
        - **Insufficient sample** — a series no longer than the warm-up, so the result is ``null``.
        - **window == 1** — each EMA reduces to the identity, so the expression reproduces the input.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`ema`: The single exponential pass this is built from.
        - :func:`tema`: The triple-EMA sibling.
        - :func:`t3`: The six-pass Tillson member of the lag-reduced family.

    References:
        - Mulloy, P. G. (1994). "Smoothing Data with Faster Moving Averages." *Technical Analysis of Stocks &
          Commodities*, 12(1).
        - https://en.wikipedia.org/wiki/Double_exponential_moving_average

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import dema
        >>>
        >>> frame = pl.DataFrame({"close": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]})
        >>> frame.select(dema=dema(pl.col("close"), window=2).round(4))["dema"].to_list()
        [None, None, 6.0, 8.0, 10.0, 12.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "close": [10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0, 22.0],
        ...     }
        ... )
        >>> frame.with_columns(dema=dema(pl.col("close"), 2).over("ticker").round(4))["dema"].to_list()
        [None, None, 12.0, 11.2222, 12.8148, None, None, 21.0, 22.7778, 22.1852]

        A ``null`` (skipped: it voids its own row while the recursion bridges the gap) and a ``NaN`` (which latches)
        make the exact handling visible at a glance:

        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, None, 15.0, float("nan"), 17.0, 18.0, 19.0]})
        >>> frame.select(dema=dema(pl.col("close"), 2).round(4))["dema"].to_list()
        [None, None, 12.0, 13.0, None, 15.0204, nan, nan, nan, nan]

        **Insufficient sample** — a one-row series has no history beyond the seed itself, but with ``window=1`` both
        EMA passes are the identity, so the lone value passes through unchanged:

        >>> frame = pl.DataFrame({"close": [42.0]})
        >>> frame.select(dema=dema(pl.col("close"), window=1))["dema"].to_list()
        [42.0]
    """
    expr = float64_expr(expr)
    validate_window(window)
    ema_once = ema(expr, window, adjust=adjust)
    ema_twice = ema(ema_once, window, adjust=adjust)
    return (2 * ema_once - ema_twice).name.keep()


def ema(
    expr: pl.Expr,
    window: int,
    *,
    adjust: bool = False,
) -> pl.Expr:
    r"""
    Exponential Moving Average (EMA), also known as the Exponentially Weighted Moving Average (EWMA).

    The TA-standard recursive form, with smoothing factor :math:`\alpha = 2 / (n + 1)`:

    .. math::

        \mathrm{EMA}_{n-1} = \frac{1}{n} \sum_{i=0}^{n-1} x_i, \qquad
        \mathrm{EMA}_t = \alpha\, x_t + (1 - \alpha)\, \mathrm{EMA}_{t-1},
        \qquad \alpha = \frac{2}{n + 1}, \quad n = \text{window}.

    The recursive form is the TA standard, seeded with the simple average of the first ``window`` observations -- the
    classical EMA initialization, so the warm-up matches the industry reference.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Span of the exponential weighting, mapped to ``alpha = 2 / (window + 1)``. Must be ``>= 1``.
        adjust: When ``False`` (default) use the recursive form above. When ``True`` use the finite-window
            bias-corrected (adjusted) weighting that divides by the decaying sum of weights at each step. The two forms
            differ at every row in general (coinciding only for ``window == 1`` or a constant series), the gap largest
            near the start of the series and decaying geometrically as the history grows.

    Returns:
        The EMA for each row, the same length as ``expr``. The first ``window - 1`` values are ``null`` (warm-up),
        matching the uniform warm-up of the moving-average family: the value is defined only once ``window`` non-null
        observations have been seen.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Seeding**

        The unadjusted recursion (the default) is seeded with the simple average of the first ``window`` observations,
        the canonical EMA initialization; the adjusted form is exact from the first observation.

        **Edge-case behavior**

        - **Null** — a leading ``null`` run stays ``null`` until the first non-null seed; an interior ``null`` yields
          ``null`` at that position while the recursion continues across the gap (a leading run consumes no warm-up
          budget, and an interior gap decays the carried weight by ``(1 - alpha) ** k`` per Polars'
          ``ignore_nulls=False`` convention).
        - **NaN** — a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null
          position (a ``NaN`` still inside the warm-up shows as that warm-up's ``null`` on its own row, then latches
          from the first emitted row).
        - **window == 1** — the smoothing factor is ``1``, so the EMA reproduces the input.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`rma`: Wilder's variant, with smoothing factor ``1 / window``.
        - :func:`dema`: A lag-reduced average built from two chained EMAs.
        - :func:`sma`: The equal-weight simple average this is the exponential analog of.

    References:
        - Roberts, S. W. (1959). "Control Chart Tests Based on Geometric Moving Averages." *Technometrics*, 1(3),
          239-250.
        - https://doi.org/10.1080/00401706.1959.10489860
        - https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import ema
        >>>
        >>> frame = pl.DataFrame({"close": [2.0, 4.0, 6.0, 8.0, 10.0]})
        >>> frame.select(ema=ema(pl.col("close"), window=3).round(4))["ema"].to_list()
        [None, None, 4.0, 6.0, 8.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "close": [10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0, 22.0],
        ...     }
        ... )
        >>> frame.with_columns(ema=ema(pl.col("close"), 2).over("ticker").round(4))["ema"].to_list()
        [None, 10.5, 11.5, 11.1667, 12.3889, None, 21.0, 21.0, 22.3333, 22.1111]

        A ``null`` (skipped: it voids its own row while the recursion bridges the gap) and a ``NaN`` (which latches)
        make the exact handling visible at a glance:

        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, None, 15.0, float("nan"), 17.0, 18.0, 19.0]})
        >>> frame.select(ema=ema(pl.col("close"), 2).round(4))["ema"].to_list()
        [None, 10.5, 11.5, 12.5, None, 14.6429, nan, nan, nan, nan]

        **window == 1** — the smoothing factor ``alpha=1`` reproduces the input exactly, with no warm-up:

        >>> frame = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
        >>> frame.select(ema=ema(pl.col("close"), window=1))["ema"].to_list()
        [1.0, 2.0, 3.0]
    """
    expr = float64_expr(expr)
    validate_window(window)
    if window == 1:
        # span=1 => alpha=1 => identity; avoids ewm catastrophic cancellation on tiny values (``expr`` is already
        # ``Float64`` from ``float64_expr`` above).
        return (expr).name.keep()
    if adjust:
        # The bias-corrected finite-window weighting is exact from the first row; it has no seed to choose.
        return (expr.ewm_mean(span=window, adjust=True, min_samples=window, ignore_nulls=False)).name.keep()
    return (_seeded_recursive_mean(expr, 2.0 / (window + 1), window)).name.keep()


def hma(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Hull Moving Average (HMA), also known as the Hull MA.

    A low-lag, smooth moving average (Alan Hull, 2005) built from three weighted moving averages. With
    :math:`n = \text{window}`, the half-period :math:`h = \lfloor n / 2 + \tfrac{1}{2} \rfloor` and the smoothing period
    :math:`s = \lfloor \sqrt{n} + \tfrac{1}{2} \rfloor` (both rounded half **up**):

    .. math::

        \mathrm{raw}_t &= 2 \, \mathrm{WMA}(x, h)_t - \mathrm{WMA}(x, n)_t \\
        \mathrm{HMA}_t &= \mathrm{WMA}(\mathrm{raw}, s)_t.

    The inner difference :math:`2\,\mathrm{WMA}(x, h) - \mathrm{WMA}(x, n)` cancels most of the lag of a plain weighted
    average, and the final smoothing over :math:`s` observations restores smoothness. Because the lag correction can
    over- and under-shoot, the HMA may exceed the range of the input window.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of observations in the moving window. Must be ``>= 2``.

    Returns:
        The HMA for each row, the same length as ``expr``. The first ``window + s - 2`` values are ``null`` (warm-up),
        where :math:`s = \lfloor \sqrt{n} + \tfrac{1}{2} \rfloor`: the inner ``WMA(x, window)`` needs ``window``
        observations, after which the final ``WMA(., s)`` needs ``s - 1`` more.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``. The half-period :math:`\lfloor n / 2 + \tfrac{1}{2} \rfloor` collapses to ``1``
            at ``window == 1`` and the HMA degenerates there, so the smallest meaningful window is ``2``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Period rounding**

        The two period reductions use round-half-**up** (``floor(window / 2 + 0.5)`` and ``floor(sqrt(window) + 0.5)``),
        not Python's built-in ``round`` (which rounds half to even). The two disagree on the half-period only for an
        odd ``window`` whose half ``floor(window / 2)`` is even -- ``window`` congruent to ``1`` modulo ``4`` (``5``,
        ``9``, ``13``, ...) -- where round-half-up takes the ``.5`` up while round-half-to-even takes it down to the
        even floor. For ``window`` congruent to ``3`` modulo ``4`` (``3``, ``7``, ``11``, ...) the half still lands on
        a ``.5`` boundary but both round alike.

        **Edge-case behavior**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)
          — propagated through every composing :func:`wma`.
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`wma`: The weighted mean this composes.
        - :func:`sma`: The unweighted baseline.
        - :func:`dema`: A lag-reduced average built by the same doubling correction.

    References:
        - Hull, A. (2005). "Hull Moving Average."
        - https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/hull-moving-average-hma

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import hma
        >>>
        >>> frame = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
        >>> frame.select(hma=hma(pl.col("close"), window=4).round(4))["hma"].to_list()
        [None, None, None, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "close": [10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0, 22.0],
        ...     }
        ... )
        >>> frame.with_columns(hma=hma(pl.col("close"), 2).over("ticker").round(4))["hma"].to_list()
        [None, 11.3333, 12.3333, 10.6667, 13.6667, None, 22.6667, 20.6667, 23.6667, 21.6667]

        A ``null`` (skipped, and any window it touches yields ``null``) and a ``NaN`` (which propagates) make the
        exact handling visible at a glance:

        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, None, 15.0, float("nan"), 17.0, 18.0, 19.0]})
        >>> frame.select(hma=hma(pl.col("close"), 2).round(4))["hma"].to_list()
        [None, 11.3333, 12.3333, 13.3333, None, None, nan, nan, 18.3333, 19.3333]
    """
    expr = float64_expr(expr)
    validate_window(window, minimum=2)
    # Period reductions round half UP (floor(x + 0.5)), not Python's round (half-to-even); they differ on the
    # half-period for odd windows. Composed from ``wma``, not a weighted ``rolling_mean``, whose kernel panics on null
    # inputs across the supported polars range ("weights not yet supported...") and would break null propagation.
    half_window = math.floor(window / 2 + 0.5)
    smoothing_window = math.floor(math.sqrt(window) + 0.5)
    raw_expr = 2 * wma(expr, half_window) - wma(expr, window)
    return wma(raw_expr, smoothing_window).name.keep()


def _kama_kernel(
    series: pl.Series,
    window: int,
) -> pl.Series:
    """
    Sequential KAMA recurrence over one Series of ``{value, sc}`` structs (the pure-Python kernel).

    Seeds at ``value[window - 1]`` (the first computable bar) and then iterates ``kama += sc * (value - kama)``. A
    ``null`` ``value`` or ``sc`` leaves ``null`` at that row while preserving the running state (a bridge); a ``NaN``
    flows into the recurrence and latches.
    """
    values: list[float | None] = series.struct.field("value").to_list()
    smoothing: list[float | None] = series.struct.field("sc").to_list()
    result: list[float | None] = [None] * len(values)
    kama_previous = 0.0  # placeholder until the first seeded bar; the `seeded` flag guards it from earlier use
    seeded = False
    for index in range(window - 1, len(values)):
        value = values[index]
        constant = smoothing[index]
        if not seeded:
            if value is None:
                continue
            result[index] = value
            kama_previous = value
            seeded = True
            continue
        if value is None or constant is None:
            continue
        kama_previous = kama_previous + constant * (value - kama_previous)
        result[index] = kama_previous
    return pl.Series(result, dtype=pl.Float64)


def kama(
    expr: pl.Expr,
    *,
    window: int,
    window_fast: int,
    window_slow: int,
) -> pl.Expr:
    r"""
    Kaufman's Adaptive Moving Average (KAMA) — an EMA whose smoothing constant is driven by an efficiency ratio.

    Introduced by Perry Kaufman (1995): a moving average whose smoothing constant adapts to how *efficiently* price is
    moving. Over the ``window``, the efficiency ratio compares the net move to the summed bar-to-bar travel; a high
    ratio (a clean trend) drives the smoothing toward a fast average, a low ratio (chop) toward a slow one. KAMA then
    follows price closely in trends and flattens in noise:

    .. math::

        \mathrm{ER}_t &= \frac{\lvert \mathrm{close}_t - \mathrm{close}_{t-n} \rvert}{\sum_{i=1}^{n}
            \lvert \mathrm{close}_{t-i+1} - \mathrm{close}_{t-i} \rvert}, \\
        \mathrm{SC}_t &= \bigl( \mathrm{ER}_t \, (f - s) + s \bigr)^2,
            \qquad f = \frac{2}{n_f + 1}, \quad s = \frac{2}{n_s + 1}, \\
        \mathrm{KAMA}_t &= \mathrm{KAMA}_{t-1} + \mathrm{SC}_t \, (\mathrm{close}_t - \mathrm{KAMA}_{t-1}),

    where :math:`n` is ``window``, :math:`n_f` is ``window_fast``, and :math:`n_s` is ``window_slow``. The recurrence is
    seeded at ``close`` on the bar one step before the efficiency ratio is first defined (row ``window - 1``); the first
    adaptive update then runs at row ``window``, where the efficiency ratio becomes available.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of observations in the efficiency-ratio look-back. Must be ``>= 1``.
        window_fast: Period of the fast smoothing-constant bound (canonically ``2``), ``2 / (window_fast + 1)``. Must be
            ``>= 1`` (the fast bound is the more responsive end of the adaptive range).
        window_slow: Period of the slow smoothing-constant bound (canonically ``30``), ``2 / (window_slow + 1)``.
            Must be ``>= 1`` and ``>= window_fast``.

    Returns:
        The KAMA for each row, the same length as ``expr``. The first ``window - 1`` values are ``null`` (warm-up); the
        value at row ``window - 1`` is ``close`` itself (the seed), and the adaptive recurrence runs from there.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``, ``window_fast < 1``, ``window_slow < 1``, or ``window_fast > window_slow``.

    Note:
        **Precision**

        The efficiency ratio and adaptive smoothing constant are checked against an independent reference, but the
        seeded recurrence they drive is one-shape with the implementation, so the oracle confirms its internal
        consistency, not its independence; the independent witnesses are the TA-Lib differential and frozen hand-derived
        golden masters. Agreement holds to ten significant figures (a ``1e-10`` band) on any finite input within a sane
        dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning limit beyond
        it.

        It is homogeneous of degree ``1`` (the efficiency ratio is scale-invariant — a ratio of absolute moves — and
        the recurrence is linear in the input, so ``kama(k * x) == k * kama(x)``).

        **Edge-case behavior**

        - **Null** — a leading ``null`` run stays ``null`` until the first non-null seed; an interior ``null`` yields
          ``null`` at that position while the recursion continues across the gap — whether the ``null`` reaches the
          recurrence directly through ``close`` or via the efficiency-ratio window touching one.
        - **NaN** — a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null
          position.
        - **Degenerate denominator** — when there is no bar-to-bar travel the efficiency ratio is taken as ``0``
          (avoiding the ``0 / 0`` degenerate), so the smoothing constant sits at the slow bound and KAMA barely moves.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`ema`: The fixed-smoothing exponential average KAMA adapts between.
        - :func:`rma`: Wilder's fixed-smoothing average.
        - :func:`mama`: The MESA adaptive average, steered by cycle phase rather than efficiency.

    References:
        - Kaufman, P. J. (1995). *Smarter Trading: Improving Performance in Changing Markets*. McGraw-Hill.
        - https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/kaufmans-adaptive-moving-average-kama

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import kama
        >>>
        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 11.0, 13.0, 12.5]})
        >>> frame.select(
        ...     kama=kama(pl.col("close"), window=2, window_fast=2, window_slow=30).round(4)
        ... )["kama"].to_list()
        [None, 11.0, 11.4444, 11.4426, 11.5522, 11.724]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "close": [10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0, 22.0],
        ...     }
        ... )
        >>> frame.with_columns(
        ...     kama=kama(pl.col("close"), window=2, window_fast=2, window_slow=30).over("ticker").round(4)
        ... )["kama"].to_list()
        [None, 11.0, 11.4444, 11.4426, 11.5522, None, 22.0, 21.9297, 22.0049, 22.0046]

        A ``null`` (bridged) and a ``NaN`` (latched) make the handling visible:

        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, None, 13.0, float("nan"), 15.0, 16.0]})
        >>> frame.select(
        ...     kama=kama(pl.col("close"), window=2, window_fast=2, window_slow=30).round(4)
        ... )["kama"].to_list()
        [None, 11.0, 11.4444, None, None, None, nan, nan]

        **Degenerate denominator** — a flat series has zero bar-to-bar travel, so the efficiency ratio is taken as
        ``0`` (avoiding the ``0 / 0`` degenerate) and KAMA holds at the constant:

        >>> frame = pl.DataFrame({"close": [5.0, 5.0, 5.0, 5.0]})
        >>> frame.select(kama=kama(pl.col("close"), window=2, window_fast=2, window_slow=30))["kama"].to_list()
        [None, 5.0, 5.0, 5.0]
    """
    expr = float64_expr(expr)
    validate_window(window)
    validate_window(window_fast, name="window_fast")
    validate_window(window_slow, name="window_slow")
    validate_window_order(window_fast, window_slow)
    change = (expr - expr.shift(window)).abs()
    volatility = expr.diff().abs().rolling_sum(window)
    efficiency_ratio = pl.when(volatility == 0).then(0.0).otherwise(change / volatility)
    fast = 2.0 / (window_fast + 1)
    slow = 2.0 / (window_slow + 1)
    smoothing_constant = (efficiency_ratio * (fast - slow) + slow) ** 2
    return (
        pl.struct(value=expr, sc=smoothing_constant)
        .map_batches(partial(_kama_kernel, window=window), return_dtype=pl.Float64)
        .name.keep()
    )


def rma(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Wilder Moving Average (RMA), also known as SMMA / Wilder smoothing / Modified MA.

    An exponential moving average whose smoothing factor is pinned to :math:`\alpha = 1 / n`, evaluated in its recursive
    (unadjusted) form:

    .. math::

        \mathrm{RMA}_t =
        \begin{cases}
            \dfrac{1}{n} \sum_{i=0}^{n-1} x_i, & t = n - 1, \\[4pt]
            \mathrm{RMA}_{t-1} + \dfrac{1}{n}\,\bigl(x_t - \mathrm{RMA}_{t-1}\bigr)
              = \Bigl(1 - \tfrac{1}{n}\Bigr)\,\mathrm{RMA}_{t-1} + \tfrac{1}{n}\,x_t, & t \ge n,
        \end{cases}
        \qquad n = \text{window}.

    It is the smoothing Wilder used throughout *New Concepts in Technical Trading Systems* (RSI, ATR, ADX, DMI).
    Equivalently it is an EMA with ``alpha = 1 / window``, seeded with the simple average of the first ``window``
    observations -- Wilder's initialization, so the warm-up matches the industry reference from the first emitted value.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        The RMA for each row, the same length as ``expr``. The first ``window - 1`` values are ``null`` (warm-up) -- the
        recursion emits only once ``window`` non-null observations have been counted -- seeded there with their simple
        average -- after which every later row is defined wherever its own input is (an interior ``null`` still
        voids its own row, as the Note details).

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
          ``null`` at that position while the recursion continues across the gap (a leading run consumes no warm-up
          budget, and an interior gap decays the carried weight by ``(1 - alpha) ** k``, emulating
          ``ewm_mean(adjust=False, ignore_nulls=False)`` semantics).
        - **NaN** — a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null
          position.
        - **window == 1** — the smoothing factor is ``1``, the warm-up vanishes, and the result reproduces the input.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`ema`: The same recursion with smoothing factor ``2 / (window + 1)``.
        - :func:`atr`: The volatility average that smooths the true range with this Wilder mean.
        - :func:`sma`: The equal-weight baseline.

    References:
        - Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import rma
        >>>
        >>> frame = pl.DataFrame({"close": [2.0, 4.0, 6.0, 8.0, 10.0]})
        >>> frame.select(rma=rma(pl.col("close"), window=3).round(4))["rma"].to_list()
        [None, None, 4.0, 5.3333, 6.8889]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "close": [10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0, 22.0],
        ...     }
        ... )
        >>> frame.with_columns(rma=rma(pl.col("close"), 2).over("ticker").round(4))["rma"].to_list()
        [None, 10.5, 11.25, 11.125, 12.0625, None, 21.0, 21.0, 22.0, 22.0]

        A ``null`` (skipped: it voids its own row while the recursion bridges the gap) and a ``NaN`` (which latches)
        make the exact handling visible at a glance:

        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, None, 15.0, float("nan"), 17.0, 18.0, 19.0]})
        >>> frame.select(rma=rma(pl.col("close"), 2).round(4))["rma"].to_list()
        [None, 10.5, 11.25, 12.125, None, 14.0417, nan, nan, nan, nan]

        **window == 1** — the smoothing factor ``alpha=1`` reproduces the input exactly, with zero warm-up:

        >>> frame = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
        >>> frame.select(rma=rma(pl.col("close"), window=1))["rma"].to_list()
        [1.0, 2.0, 3.0]
    """
    expr = float64_expr(expr)
    validate_window(window)
    if window == 1:
        # alpha=1 => identity; short-circuits the seeded map_batches kernel for the trivial case (``expr`` is already
        # ``Float64`` from ``float64_expr`` above).
        return (expr).name.keep()
    return (_seeded_recursive_mean(expr, 1.0 / window, window)).name.keep()


def sma(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Simple Moving Average (SMA) — the unweighted arithmetic mean over a trailing window.

    The unweighted arithmetic mean of the last ``window`` observations, assigning equal weight to every point in the
    window:

    .. math:: \mathrm{SMA}_t = \frac{1}{n} \sum_{i=0}^{n-1} x_{t-i}, \qquad n = \text{window}.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        The SMA for each row, the same length as ``expr``. The first ``window - 1`` values are ``null`` (warm-up) -- the
        value is defined only once ``window`` observations have been seen.

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
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there (``null`` takes precedence over
          ``NaN``).
        - **Insufficient sample** — a series shorter than ``window`` observations, so the result is ``null``.
        - **window == 1** — the one-point mean is the input itself, so the SMA reproduces the input.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`ema`: The exponentially-weighted analog, more responsive to recent values.
        - :func:`wma`: The linearly-weighted analog.
        - :func:`trima`: The triangular average, a simple average of a simple average.

    References:
        - https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import sma
        >>>
        >>> frame = pl.DataFrame({"close": [2.0, 4.0, 6.0, 8.0, 10.0]})
        >>> frame.select(sma=sma(pl.col("close"), window=3).round(4))["sma"].to_list()
        [None, None, 4.0, 6.0, 8.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "close": [10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0, 22.0],
        ...     }
        ... )
        >>> frame.with_columns(sma=sma(pl.col("close"), 2).over("ticker").round(4))["sma"].to_list()
        [None, 10.5, 11.5, 11.5, 12.0, None, 21.0, 21.5, 22.0, 22.5]

        A ``null`` (skipped, and any window it touches yields ``null``) and a ``NaN`` (which propagates) make the
        exact handling visible at a glance:

        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, None, 15.0, float("nan"), 17.0, 18.0, 19.0]})
        >>> frame.select(sma=sma(pl.col("close"), 2).round(4))["sma"].to_list()
        [None, 10.5, 11.5, 12.5, None, None, nan, nan, 17.5, 18.5]

        **Insufficient sample** — a one-element series holds only a single observation, and at ``window=1`` that is
        exactly enough, so the mean returns the value itself:

        >>> frame = pl.DataFrame({"close": [42.0]})
        >>> frame.select(sma=sma(pl.col("close"), window=1))["sma"].to_list()
        [42.0]
    """
    expr = float64_expr(expr)
    validate_window(window)
    return (expr.rolling_mean(window)).name.keep()


def t3(
    expr: pl.Expr,
    window: int,
    *,
    volume_factor: float = 0.7,
    adjust: bool = False,
) -> pl.Expr:
    r"""
    Tillson T3 Moving Average (T3), also known as the Tillson Moving Average.

    A heavily smoothed yet low-lag moving average built by applying a Generalized DEMA (``GD``) three times. With
    ``v = volume_factor`` and ``EMA`` the recursive exponential moving average of length ``window``:

    .. math::

        \mathrm{GD}(x) = (1 + v)\,\mathrm{EMA}(x) - v\,\mathrm{EMA}(\mathrm{EMA}(x)),
        \qquad
        \mathrm{T3} = \mathrm{GD}(\mathrm{GD}(\mathrm{GD}(x))).

    Expanded over the six chained EMAs :math:`e_1 = \mathrm{EMA}(x),\, e_2 = \mathrm{EMA}(e_1), \dots,\,
    e_6 = \mathrm{EMA}(e_5)`, this equals — for the ideal EMA operator — the closed coefficient form computed here
    (the two agree exactly once warmed up; during the warm-up region the masked-EMA composition and the coefficient
    form differ in the transient and converge only asymptotically):

    .. math::

        \mathrm{T3} = c_1 e_6 + c_2 e_5 + c_3 e_4 + c_4 e_3,

    .. math::

        c_1 = -v^3,\quad
        c_2 = 3v^2 + 3v^3,\quad
        c_3 = -6v^2 - 3v - 3v^3,\quad
        c_4 = 1 + 3v + 3v^2 + v^3.

    The coefficients sum to exactly ``1`` (:math:`c_1 + c_2 + c_3 + c_4 = 1`), so T3 of a constant series is that
    constant.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Span of the exponential weighting, mapped to ``alpha = 2 / (window + 1)``. Must be ``>= 1``.
        volume_factor: The Tillson volume factor ``v`` controlling smoothing versus responsiveness; the canonical
            default is ``0.7``. Must be a finite number.
        adjust: Whether to use the bias-corrected expanding-weights EMA (``True``), which differs from the recursive
            form at every emitted row — the gap largest near the start and decaying as history grows — or the recursive
            Technical-Analysis EMA seeded with the SMA of the first ``window`` observations (``False``, the default).

    Returns:
        The T3 for each row, the same length as ``expr``. Because the value is composed from six chained :func:`ema`
        passes of the same ``window`` (each carrying a ``window - 1`` warm-up), the first ``6 * (window - 1)`` values
        are ``null`` (warm-up), clamped to the series length.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``, or if ``volume_factor`` is not a finite number.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Seeding**

        The recursive EMA is seeded with the SMA of the first ``window`` observations.

        **Edge-case behavior**

        - **Null** — a leading ``null`` run stays ``null`` until the first non-null seed; an interior ``null`` yields
          ``null`` at that position while the recursion continues across the gap.
        - **NaN** — a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null
          position.
        - **Insufficient sample** — a series no longer than the warm-up, so the result is ``null``.
        - **window == 1** — each EMA reduces to the identity, so the expression reproduces the input up to a
          floating-point rounding (unlike ``dema`` / ``tema``, the coefficient form does not cancel exactly).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`dema`: The double-EMA lag-reduced average.
        - :func:`tema`: The triple-EMA lag-reduced average.
        - :func:`ema`: The exponential pass T3 chains six times.

    References:
        - Tillson, T. (1998). "Better Moving Averages." *Technical Analysis of Stocks & Commodities*, 16(1).

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import t3
        >>>
        >>> frame = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
        >>> frame.select(t3=t3(pl.col("close"), window=2).round(4))["t3"].to_list()
        [None, None, None, None, None, None, 6.55, 7.55]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 8 + ["B"] * 8,
        ...         "close": [
        ...             10.0,
        ...             11.0,
        ...             12.0,
        ...             11.0,
        ...             13.0,
        ...             14.0,
        ...             13.0,
        ...             15.0,
        ...             20.0,
        ...             22.0,
        ...             21.0,
        ...             23.0,
        ...             22.0,
        ...             24.0,
        ...             25.0,
        ...             24.0,
        ...         ],
        ...     }
        ... )
        >>> frame.with_columns(t3=t3(pl.col("close"), 2).over("ticker").round(4))["t3"].to_list()
        [None, None, None, None, None, None, 13.3568, 14.2815, None, None, None, None, None, None, 24.4079, 24.3942]

        A ``null`` (skipped: it voids its own row while the recursion bridges the gap) and a ``NaN`` (which latches)
        make the exact handling visible at a glance:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "close": [
        ...             10.0,
        ...             11.0,
        ...             12.0,
        ...             13.0,
        ...             14.0,
        ...             15.0,
        ...             16.0,
        ...             17.0,
        ...             None,
        ...             19.0,
        ...             float("nan"),
        ...             21.0,
        ...             22.0,
        ...         ],
        ...     }
        ... )
        >>> frame.select(t3=t3(pl.col("close"), 2).round(4))["t3"].to_list()
        [None, None, None, None, None, None, 15.55, 16.55, None, 18.7118, nan, nan, nan]

        **Insufficient sample** — a one-row series has no history beyond the seed, but at ``window=1`` every chained
        EMA is the identity, so the value passes through:

        >>> frame = pl.DataFrame({"close": [42.0]})
        >>> frame.select(t3=t3(pl.col("close"), window=1))["t3"].to_list()
        [42.0]

        **window == 1** — every chained EMA reduces to the identity and the four coefficients sum to exactly ``1``,
        so the T3 reproduces the input:

        >>> frame = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
        >>> frame.select(t3=t3(pl.col("close"), window=1))["t3"].to_list()
        [1.0, 2.0, 3.0]
    """
    expr = float64_expr(expr)
    validate_window(window)
    validate_finite(volume_factor, "volume_factor")

    volume_factor_squared = volume_factor * volume_factor
    volume_factor_cubed = volume_factor_squared * volume_factor
    coefficient_e6 = -volume_factor_cubed
    coefficient_e5 = 3.0 * volume_factor_squared + 3.0 * volume_factor_cubed
    coefficient_e4 = -6.0 * volume_factor_squared - 3.0 * volume_factor - 3.0 * volume_factor_cubed
    coefficient_e3 = 1.0 + 3.0 * volume_factor + 3.0 * volume_factor_squared + volume_factor_cubed

    ema_1 = ema(expr, window, adjust=adjust)
    ema_2 = ema(ema_1, window, adjust=adjust)
    ema_3 = ema(ema_2, window, adjust=adjust)
    ema_4 = ema(ema_3, window, adjust=adjust)
    ema_5 = ema(ema_4, window, adjust=adjust)
    ema_6 = ema(ema_5, window, adjust=adjust)

    combined = coefficient_e6 * ema_6 + coefficient_e5 * ema_5 + coefficient_e4 * ema_4 + coefficient_e3 * ema_3
    return combined.name.keep()


def tema(
    expr: pl.Expr,
    window: int,
    *,
    adjust: bool = False,
) -> pl.Expr:
    r"""
    Triple Exponential Moving Average (TEMA), also known as the triple EMA.

    A low-lag smoother (Mulloy, 1994) built from three nested exponential moving averages of the same window. With
    :math:`\mathrm{EMA}^{(1)} = \mathrm{EMA}(x)`, :math:`\mathrm{EMA}^{(2)} = \mathrm{EMA}(\mathrm{EMA}^{(1)})` and
    :math:`\mathrm{EMA}^{(3)} = \mathrm{EMA}(\mathrm{EMA}^{(2)})`, the triple-EMA correction cancels the lag of the
    cascade:

    .. math:: \mathrm{TEMA}_t = 3\,\mathrm{EMA}^{(1)}_t - 3\,\mathrm{EMA}^{(2)}_t + \mathrm{EMA}^{(3)}_t.

    Each :func:`ema` uses the recursive technical-analysis form with smoothing factor
    :math:`\alpha = 2 / (\text{window} + 1)`, seeded with the SMA of the first ``window`` observations.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Span of the exponential weighting, mapped to ``alpha = 2 / (window + 1)``. Must be ``>= 1``.
        adjust: Whether to use the bias-corrected expanding-weights EMA. ``False`` (the default) selects the recursive
            technical-analysis EMA.

    Returns:
        The TEMA for each row, the same length as ``expr``. The first ``3 * (window - 1)`` values are ``null``
        (warm-up), clamped to the series length: the value is composed from three chained :func:`ema` passes of the same
        ``window`` (each carrying a ``window - 1`` warm-up), so the warm-up is three times that of a plain EMA. Under
        the default ``adjust=False``, each pass is seeded with the SMA of the first ``window`` observations.

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
        - **window == 1** — each EMA reduces to the identity, so the expression reproduces the input.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`dema`: The double-EMA sibling.
        - :func:`t3`: The six-pass Tillson sibling.
        - :func:`ema`: The exponential pass this chains three times.

    References:
        - Mulloy, P. G. (1994). "Smoothing Data with Faster Moving Averages." *Technical Analysis of Stocks &
          Commodities*, 12(1).
        - https://en.wikipedia.org/wiki/Triple_exponential_moving_average

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import tema
        >>>
        >>> frame = pl.DataFrame({"close": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]})
        >>> frame.select(tema=tema(pl.col("close"), window=2).round(4))["tema"].to_list()
        [None, None, None, 8.0, 10.0, 12.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "close": [10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0, 22.0],
        ...     }
        ... )
        >>> frame.with_columns(tema=tema(pl.col("close"), 2).over("ticker").round(4))["tema"].to_list()
        [None, None, None, 11.2222, 12.9383, None, None, None, 22.7778, 22.0617]

        A ``null`` (skipped: it voids its own row while the recursion bridges the gap) and a ``NaN`` (which latches)
        make the exact handling visible at a glance:

        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, None, 15.0, float("nan"), 17.0, 18.0, 19.0]})
        >>> frame.select(tema=tema(pl.col("close"), 2).round(4))["tema"].to_list()
        [None, None, None, 13.0, None, 15.0029, nan, nan, nan, nan]

        **window == 1** — each of the three nested EMAs collapses to the identity, so the TEMA reproduces the input:

        >>> frame = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]})
        >>> frame.select(tema=tema(pl.col("close"), window=1))["tema"].to_list()
        [1.0, 2.0, 3.0, 4.0]
    """
    expr = float64_expr(expr)
    validate_window(window)
    ema_first = ema(expr, window, adjust=adjust)
    ema_second = ema(ema_first, window, adjust=adjust)
    ema_third = ema(ema_second, window, adjust=adjust)
    return (3 * ema_first - 3 * ema_second + ema_third).name.keep()


def trima(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Triangular Moving Average — a double-smoothed mean that weights the middle of the window most.

    A moving average run twice, so the weights form a triangle (or trapezoid) peaking at the center of the window
    rather than being uniform. It is equivalent to an :func:`sma` of an :func:`sma`, with the two sub-windows chosen so
    the combined span is ``window``:

    .. math::

        \mathrm{TRIMA}_t = \mathrm{SMA}\bigl(\mathrm{SMA}(x, m_1), m_2\bigr)_t, \qquad m_1 + m_2 = n + 1,

    where for an odd window ``m_1 = m_2 = (n + 1) / 2`` and for an even window ``m_1 = n / 2``, ``m_2 = n / 2 + 1``
    (the order does not matter — the double average is commutative).

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        The triangular moving average for each row, the same length as the input. The first ``window - 1`` values are
        ``null`` (warm-up), matching the uniform warm-up of the moving-average family.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Edge-case behavior**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)
          — built from two :func:`sma` passes, each holding to that same ``min_samples=window`` contract.
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there — through both :func:`sma` passes
          it composes.
        - **Insufficient sample** — a series shorter than ``window`` observations, so the result is ``null``.
        - **window == 1** — both sub-windows are ``1``, so the TRIMA reproduces the input.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`sma`: The single-pass simple moving average this double-smooths.
        - :func:`wma`: A single-pass linearly-weighted average, also tilting the window's weights off uniform.
        - :func:`hma`: Another average built by composing simpler moving averages.

    References:
        - https://en.wikipedia.org/wiki/Moving_average

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import trima
        >>>
        >>> frame = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        >>> frame.select(trima=trima(pl.col("x"), 4).round(4))["trima"].to_list()
        [None, None, None, 2.5, 3.5, 4.5]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 3 + ["B"] * 3,
        ...         "x": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        ...     }
        ... )
        >>> expr = trima(pl.col("x"), 2).over("ticker").round(4)
        >>> frame.with_columns(trima=expr)["trima"].to_list()
        [None, 1.5, 2.5, None, 15.0, 25.0]

        A ``null`` (which both passes propagate) and a ``NaN`` make the missing-data handling visible:

        >>> frame = pl.DataFrame({"x": [1.0, None, 3.0, float("nan"), 5.0, 6.0]})
        >>> frame.select(trima=trima(pl.col("x"), 3).round(4))["trima"].to_list()
        [None, None, None, None, nan, nan]

        **Insufficient sample** — a one-row series has nothing to average beyond itself, so at ``window=1`` the
        triangular smoothing is the identity and the value passes through unchanged:

        >>> frame = pl.DataFrame({"x": [42.0]})
        >>> frame.select(trima=trima(pl.col("x"), window=1))["trima"].to_list()
        [42.0]
    """
    expr = float64_expr(expr)
    validate_window(window)
    # Triangular weights = an SMA of an SMA; the two spans sum to window + 1 (commutative, so order is irrelevant).
    if window % 2:
        half = (window + 1) // 2
        return (sma(sma(expr, half), half)).name.keep()
    return (sma(sma(expr, window // 2), window // 2 + 1)).name.keep()


def vwma(
    expr: pl.Expr,
    volume: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Volume-Weighted Moving Average (VWMA), also known as the Volume-Weighted MA.

    The rolling mean of ``expr`` weighted by ``volume`` over the last ``window`` observations:

    .. math::

        \mathrm{VWMA}_t = \frac{\sum_{i=0}^{n-1} P_{t-i}\, V_{t-i}}{\sum_{i=0}^{n-1} V_{t-i}},
        \qquad n = \text{window},

    where :math:`P` is ``expr`` and :math:`V` is ``volume``. When every volume in the window is equal it reduces to the
    SMA of ``expr``; with ``window == 1`` (and non-zero volume) it reproduces ``expr`` itself.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        volume: Traded-volume series (e.g. ``pl.col("volume")``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        The VWMA for each row, the same length as ``expr``. The first ``window - 1`` values are ``null`` (warm-up) --
        the value is defined only once ``window`` observations have been seen.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Precision**

        Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite input
        within a sane dynamic range; the documentation's *Correctness* page gives the method and the float-conditioning
        limit beyond it.

        **Edge-case behavior**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)
          — whether the ``null`` is in ``expr`` or in ``volume``.
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there (``null`` takes precedence over
          ``NaN``).
        - **Insufficient sample** — a series shorter than ``window`` observations, so the result is ``null``.
        - **Degenerate denominator** — every volume in the window is zero, so the result is a ``0 / 0``, i.e. ``NaN``
          — the window is detected exactly (via the rolling maximum of ``|volume|``), so a sub-ULP rolling-sum
          residual cannot leak a spurious ``+/-inf`` instead.
        - **window == 1** — with non-zero volume the single ``(price, volume)`` pair reduces to ``expr`` itself, so
          the VWMA reproduces the price to within a rounding ULP (``(p * v) / v`` is one float multiply-divide, not
          an identity copy — its siblings' bit-exact ``window == 1`` identity does not apply here).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`sma`: The equal-weight mean it reduces to when volume is constant.
        - :func:`vwap`: The cumulative volume-weighted price, the session-anchored cousin.
        - :func:`wma`: The linearly-weighted mean.

    References:
        - https://en.wikipedia.org/wiki/Moving_average

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import vwma
        >>>
        >>> frame = pl.DataFrame(
        ...     {
        ...         "close": [10.0, 11.0, 12.0, 13.0, 14.0],
        ...         "volume": [100.0, 200.0, 300.0, 400.0, 500.0],
        ...     }
        ... )
        >>> frame.select(vwma=vwma(pl.col("close"), pl.col("volume"), window=3).round(4))["vwma"].to_list()
        [None, None, 11.3333, 12.2222, 13.1667]

        On a multi-series panel, wrap the call in ``.over`` so each group warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "price": [10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0, 22.0],
        ...         "volume": [100.0, 120.0, 90.0, 110.0, 130.0, 100.0, 120.0, 90.0, 110.0, 130.0],
        ...     }
        ... )
        >>> expr = vwma(pl.col("price"), pl.col("volume"), 2).over("ticker").round(4)
        >>> frame.with_columns(vwma=expr)["vwma"].to_list()
        [None, 10.5455, 11.4286, 11.45, 12.0833, None, 21.0909, 21.5714, 22.1, 22.4583]

        A ``null`` (skipped, and any window it touches yields ``null``) and a ``NaN`` (which propagates) make the
        exact handling visible at a glance:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "price": [10.0, 11.0, 12.0, 13.0, None, 15.0, float("nan"), 17.0, 18.0, 19.0],
        ...         "volume": [100.0, 120.0, 90.0, 110.0, 130.0, 100.0, 95.0, 140.0, 105.0, 115.0],
        ...     }
        ... )
        >>> expr = vwma(pl.col("price"), pl.col("volume"), 2).round(4)
        >>> frame.select(vwma=expr)["vwma"].to_list()
        [None, 10.5455, 11.4286, 12.55, None, None, nan, nan, 17.4286, 18.5227]

        **Insufficient sample** — a one-row input has only a single (price, volume) pair, and at ``window=1`` that
        pair alone determines the weighted mean, so the price passes through:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "close": [42.0],
        ...         "volume": [10.0],
        ...     }
        ... )
        >>> frame.select(vwma=vwma(pl.col("close"), pl.col("volume"), window=1))["vwma"].to_list()
        [42.0]

        **Degenerate denominator** — an all-zero-volume window is the IEEE-754 ``0 / 0`` degenerate, so once the
        window fills the ratio is ``NaN``:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "close": [10.0, 11.0, 12.0],
        ...         "volume": [0.0, 0.0, 0.0],
        ...     }
        ... )
        >>> frame.select(vwma=vwma(pl.col("close"), pl.col("volume"), window=2))["vwma"].to_list()
        [None, nan, nan]
    """
    expr = float64_expr(expr)
    volume = float64_expr(volume)
    validate_window(window)
    weighted_sum = (expr * volume).rolling_sum(window)
    raw = weighted_sum / volume.rolling_sum(window)
    # An all-zero-volume window is the 0/0 degenerate: detect it exactly via the rolling maximum of the absolute
    # volume (which is exactly 0 only when every volume in the window is 0), so a sub-ULP residual in the rolling-sum
    # numerator cannot fake a ±inf reading, and return NaN as documented. Gate on the weighted-sum being non-null so a
    # null in expr (which voids that sum) keeps null precedence: a window holding a null still propagates null through
    # the division.
    is_zero_volume = (volume.abs().rolling_max(window) == 0) & weighted_sum.is_not_null()
    return pl.when(is_zero_volume).then(float("nan")).otherwise(raw).name.keep()


def wma(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Weighted Moving Average (WMA), also known as the Linear Weighted Moving Average (LWMA).

    A moving average whose weights rise linearly from ``1`` on the oldest observation in the window to ``window`` on the
    most recent, normalized by the sum of the weights. The most recent price therefore carries the highest weight,
    making the WMA more responsive to recent price action than the equally-weighted SMA:

    .. math::

        \mathrm{WMA}_t = \frac{\sum_{i=0}^{n-1} (n - i)\, x_{t-i}}{\sum_{i=1}^{n} i}
                       = \frac{\sum_{i=0}^{n-1} (n - i)\, x_{t-i}}{\tfrac{n(n+1)}{2}},
        \qquad n = \text{window}.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        The WMA for each row, the same length as ``expr``. The first ``window - 1`` values are ``null`` (warm-up) -- the
        value is defined only once ``window`` observations have been seen.

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
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there (``null`` takes precedence over
          ``NaN``).
        - **Insufficient sample** — a series shorter than ``window`` observations, so the result is ``null``.
        - **window == 1** — the single weight normalizes to one, so the WMA reproduces the input.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`sma`: The unweighted analog.
        - :func:`hma`: A low-lag average built by composing weighted means.
        - :func:`ema`: The exponentially-weighted analog.

    References:
        - https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import wma
        >>>
        >>> frame = pl.DataFrame({"close": [2.0, 4.0, 6.0, 8.0, 10.0]})
        >>> frame.select(wma=wma(pl.col("close"), window=3).round(4))["wma"].to_list()
        [None, None, 4.6667, 6.6667, 8.6667]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "close": [10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0, 22.0],
        ...     }
        ... )
        >>> frame.with_columns(wma=wma(pl.col("close"), 2).over("ticker").round(4))["wma"].to_list()
        [None, 10.6667, 11.6667, 11.3333, 12.3333, None, 21.3333, 21.3333, 22.3333, 22.3333]

        A ``null`` (skipped, and any window it touches yields ``null``) and a ``NaN`` (which propagates) make the
        exact handling visible at a glance:

        >>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, None, 15.0, float("nan"), 17.0, 18.0, 19.0]})
        >>> frame.select(wma=wma(pl.col("close"), 2).round(4))["wma"].to_list()
        [None, 10.6667, 11.6667, 12.6667, None, None, nan, nan, 17.6667, 18.6667]

        **Insufficient sample** — a one-element series holds only a single observation, and at ``window=1`` that
        observation alone determines the weighted mean, so it returns the value itself:

        >>> frame = pl.DataFrame({"close": [42.0]})
        >>> frame.select(wma=wma(pl.col("close"), window=1))["wma"].to_list()
        [42.0]

        **window == 1** — the single weight in the window normalizes to ``1``, so the WMA reproduces the input
        exactly:

        >>> frame = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
        >>> frame.select(wma=wma(pl.col("close"), window=1))["wma"].to_list()
        [1.0, 2.0, 3.0]
    """
    expr = float64_expr(expr)
    validate_window(window)
    # Shifted, linearly-weighted terms over the constant weight sum n(n+1)/2 — not `rolling_mean(weights=)`, whose
    # kernel panics on null inputs across the supported polars range ("weights not yet supported...").
    # The shift form propagates null/NaN like the SMA and is numerically identical on clean data.
    weight_total = window * (window + 1) / 2
    weighted_terms = expr * window
    for offset in range(1, window):
        weighted_terms = weighted_terms + expr.shift(offset) * (window - offset)
    return (weighted_terms / weight_total).name.keep()
