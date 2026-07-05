"""
Statistic indicators.
"""

import polars as pl

from pomata._expr import float64_expr, validate_ddof, validate_window

__all__ = (
    "linear_regression",
    "linear_regression_angle",
    "linear_regression_intercept",
    "linear_regression_slope",
    "standard_deviation_ewma",
    "standard_deviation_rolling",
    "time_series_forecast",
    "variance_ewma",
    "variance_rolling",
)


def linear_regression(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Linear Regression (the endpoint of the rolling least-squares line).

    The value at the most recent bar of the ordinary-least-squares line fitted to the last ``window`` observations
    against their position in the window: a smoothed, lag-reduced estimate of where the fitted trend "is now".

    .. math::

        \mathrm{LINEARREG}_t = \bar{x}_t + \mathrm{slope}_t \cdot \frac{n - 1}{2}, \qquad n = \text{window},

    with :math:`\bar{x}_t` the window mean and :math:`\mathrm{slope}_t` the rolling :func:`linear_regression_slope`.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of observations in the regression window. Must be ``>= 2`` (a line needs at least two points).

    Returns:
        The fitted endpoint for each row, the same length as the input. The first ``window - 1`` values are ``null``
        (warm-up): the window must hold ``window`` non-null values before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        It is homogeneous of degree ``1`` in ``expr`` (a fitted price scales with the price). For a perfectly linear
        input the endpoint reproduces the series exactly.

        **Edge-case behavior:**

        - **Null** — a window containing a ``null`` yields ``null``.
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the window never spans series
          boundaries, e.g. ``linear_regression(pl.col("close"), 14).over("ticker")``.

    See Also:
        - :func:`linear_regression_slope`: The slope of the same fitted line.
        - :func:`linear_regression_intercept`: The line's value at the oldest bar of the window.
        - :func:`time_series_forecast`: The line extrapolated one bar into the future.

    References:
        - https://en.wikipedia.org/wiki/Simple_linear_regression
        - https://www.investopedia.com/terms/r/regression.asp

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import linear_regression
        >>>
        >>> frame = pl.DataFrame({"x": [10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0]})
        >>> frame.select(linear_regression(pl.col("x"), 3).round(4).alias("linreg"))["linreg"].to_list()
        [None, None, 12.8333, 12.5, 13.5, 13.5, 14.5]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {"ticker": ["A"] * 4 + ["B"] * 4, "x": [10.0, 11.0, 13.0, 12.0, 20.0, 22.0, 21.0, 24.0]}
        ... )
        >>> expr = linear_regression(pl.col("x"), 3).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("linreg"))["linreg"].to_list()
        [None, None, 12.8333, 12.5, None, None, 21.5, 23.3333]

        A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make the handling visible:

        >>> frame = pl.DataFrame({"x": [10.0, 11.0, 13.0, None, 14.0, float("nan"), 16.0]})
        >>> frame.select(linear_regression(pl.col("x"), 3).round(4).alias("linreg"))["linreg"].to_list()
        [None, None, 12.8333, None, None, None, nan]
    """
    expr = float64_expr(expr)
    validate_window(window, minimum=2)
    return expr.rolling_mean(window) + linear_regression_slope(expr, window) * (window - 1) / 2.0


def linear_regression_angle(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Linear Regression Angle (the slope of the rolling least-squares line, in degrees).

    The arctangent of the rolling :func:`linear_regression_slope`, converted to degrees, so the trend's steepness reads
    on a bounded :math:`(-90, 90)` scale:

    .. math::

        \mathrm{ANGLE}_t = \frac{180}{\pi} \arctan\bigl(\mathrm{slope}_t\bigr).

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of observations in the regression window. Must be ``>= 2`` (a line needs at least two points).

    Returns:
        The angle in degrees for each row, the same length as the input, in :math:`(-90, 90)`. The first ``window - 1``
        values are ``null`` (warm-up).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        Unlike the other regression outputs, the angle is **not** homogeneous in ``expr``: the arctangent is non-linear,
        so scaling the input does not scale the angle: amplifying it steepens the angle toward :math:`\pm 90`,
        attenuating it flattens the angle toward :math:`0`. The angle depends on the
        numeric scale of ``expr`` versus its bar spacing, so it is most meaningful on a chart's own price/time units.

        **Edge-case behavior:**

        - **Null** — a window containing a ``null`` yields ``null``.
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the window never spans series
          boundaries, e.g. ``linear_regression_angle(pl.col("close"), 14).over("ticker")``.

    See Also:
        - :func:`linear_regression_slope`: The slope this takes the arctangent of.
        - :func:`linear_regression`: The fitted line's endpoint whose steepness this reports.
        - :func:`time_series_forecast`: The same line projected one bar ahead.

    References:
        - https://en.wikipedia.org/wiki/Simple_linear_regression
        - https://www.investopedia.com/terms/r/regression.asp

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import linear_regression_angle
        >>>
        >>> frame = pl.DataFrame({"x": [10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0]})
        >>> frame.select(linear_regression_angle(pl.col("x"), 3).round(4).alias("angle"))["angle"].to_list()
        [None, None, 56.3099, 26.5651, 26.5651, 26.5651, 26.5651]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {"ticker": ["A"] * 4 + ["B"] * 4, "x": [10.0, 11.0, 13.0, 12.0, 20.0, 22.0, 21.0, 24.0]}
        ... )
        >>> expr = linear_regression_angle(pl.col("x"), 3).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("angle"))["angle"].to_list()
        [None, None, 56.3099, 26.5651, None, None, 26.5651, 45.0]

        A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make the handling visible:

        >>> frame = pl.DataFrame({"x": [10.0, 11.0, 13.0, None, 14.0, float("nan"), 16.0]})
        >>> frame.select(linear_regression_angle(pl.col("x"), 3).round(4).alias("angle"))["angle"].to_list()
        [None, None, 56.3099, None, None, None, nan]
    """
    expr = float64_expr(expr)
    validate_window(window, minimum=2)
    return linear_regression_slope(expr, window).arctan().degrees()


def linear_regression_intercept(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Linear Regression Intercept (the rolling least-squares line at the oldest bar of the window).

    The value of the fitted line at the first observation of the window (position ``0``), i.e. the ``y``-intercept of
    the regression of the last ``window`` observations against their in-window position:

    .. math::

        \mathrm{INTERCEPT}_t = \bar{x}_t - \mathrm{slope}_t \cdot \frac{n - 1}{2}, \qquad n = \text{window},

    with :math:`\bar{x}_t` the window mean and :math:`\mathrm{slope}_t` the rolling :func:`linear_regression_slope`.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of observations in the regression window. Must be ``>= 2`` (a line needs at least two points).

    Returns:
        The fitted intercept for each row, the same length as the input. The first ``window - 1`` values are ``null``
        (warm-up).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        It is homogeneous of degree ``1`` in ``expr`` (a fitted price scales with the price).

        **Edge-case behavior:**

        - **Null** — a window containing a ``null`` yields ``null``.
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the window never spans series
          boundaries, e.g. ``linear_regression_intercept(pl.col("close"), 14).over("ticker")``.

    See Also:
        - :func:`linear_regression`: The same line evaluated at the most recent bar instead of the oldest.
        - :func:`linear_regression_slope`: The slope of the same fitted line.
        - :func:`time_series_forecast`: The same line projected one bar past the most recent.

    References:
        - https://en.wikipedia.org/wiki/Simple_linear_regression
        - https://www.investopedia.com/terms/r/regression.asp

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import linear_regression_intercept
        >>>
        >>> frame = pl.DataFrame({"x": [10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0]})
        >>> frame.select(linear_regression_intercept(pl.col("x"), 3).round(4).alias("intercept"))["intercept"].to_list()
        [None, None, 9.8333, 11.5, 12.5, 12.5, 13.5]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {"ticker": ["A"] * 4 + ["B"] * 4, "x": [10.0, 11.0, 13.0, 12.0, 20.0, 22.0, 21.0, 24.0]}
        ... )
        >>> expr = linear_regression_intercept(pl.col("x"), 3).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("intercept"))["intercept"].to_list()
        [None, None, 9.8333, 11.5, None, None, 20.5, 21.3333]

        A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make the handling visible:

        >>> frame = pl.DataFrame({"x": [10.0, 11.0, 13.0, None, 14.0, float("nan"), 16.0]})
        >>> frame.select(linear_regression_intercept(pl.col("x"), 3).round(4).alias("intercept"))["intercept"].to_list()
        [None, None, 9.8333, None, None, None, nan]
    """
    expr = float64_expr(expr)
    validate_window(window, minimum=2)
    return expr.rolling_mean(window) - linear_regression_slope(expr, window) * (window - 1) / 2.0


def linear_regression_slope(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Linear Regression Slope (the slope of the rolling least-squares line).

    The ordinary-least-squares slope of the last ``window`` observations regressed against their position in the window
    — the per-bar rate of change of the fitted trend. With the in-window position :math:`i` running over the window and
    :math:`n = \text{window}`:

    .. math::

        \mathrm{slope}_t = \frac{n \sum i\,x - \sum i \sum x}{n \sum i^2 - \bigl(\sum i\bigr)^2}.

    The implementation evaluates this closed form as a fixed-weight rolling sum over the window — the weight of the bar
    ``k`` steps back is its position's deviation from the window center, divided by the position variance — which is
    numerically stable (it never forms a growing running index that would lose precision through cancellation).

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of observations in the regression window. Must be ``>= 2`` (a line needs at least two points).

    Returns:
        The fitted slope for each row, the same length as the input. The first ``window - 1`` values are ``null``
        (warm-up).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        It is homogeneous of degree ``1`` in ``expr`` (the rise scales with the price while the run is fixed). For a
        perfectly linear input it returns the exact constant slope.

        **Edge-case behavior:**

        - **Null** — a window containing a ``null`` yields ``null``.
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the window never spans series
          boundaries, e.g. ``linear_regression_slope(pl.col("close"), 14).over("ticker")``.

    See Also:
        - :func:`linear_regression`: The fitted line's value at the most recent bar.
        - :func:`linear_regression_angle`: This slope expressed as an angle in degrees.
        - :func:`time_series_forecast`: The line projected one bar ahead using this slope.

    References:
        - https://en.wikipedia.org/wiki/Simple_linear_regression
        - https://www.investopedia.com/terms/r/regression.asp

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import linear_regression_slope
        >>>
        >>> frame = pl.DataFrame({"x": [10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0]})
        >>> frame.select(linear_regression_slope(pl.col("x"), 3).round(4).alias("slope"))["slope"].to_list()
        [None, None, 1.5, 0.5, 0.5, 0.5, 0.5]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {"ticker": ["A"] * 4 + ["B"] * 4, "x": [10.0, 11.0, 13.0, 12.0, 20.0, 22.0, 21.0, 24.0]}
        ... )
        >>> expr = linear_regression_slope(pl.col("x"), 3).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("slope"))["slope"].to_list()
        [None, None, 1.5, 0.5, None, None, 0.5, 1.0]

        A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make the handling visible:

        >>> frame = pl.DataFrame({"x": [10.0, 11.0, 13.0, None, 14.0, float("nan"), 16.0]})
        >>> frame.select(linear_regression_slope(pl.col("x"), 3).round(4).alias("slope"))["slope"].to_list()
        [None, None, 1.5, None, None, None, nan]
    """
    expr = float64_expr(expr)
    validate_window(window, minimum=2)
    # OLS slope as a fixed-weight rolling sum: the bar k steps back sits at in-window position window - 1 - k, whose
    # deviation from the position mean (window - 1) / 2 is the center - k below; dividing by the position variance sum
    # window * (window ** 2 - 1) / 12 gives the slope. Keeping the weights small avoids the cancellation a running index
    # would suffer at large row offsets.
    denominator = window * (window * window - 1) / 12.0
    center = (window - 1) / 2.0
    weighted = center * expr
    for offset in range(1, window):
        weighted = weighted + (center - offset) * expr.shift(offset)
    return weighted / denominator


def standard_deviation_ewma(
    expr: pl.Expr,
    window: int,
    *,
    adjust: bool = False,
    bias: bool = True,
) -> pl.Expr:
    r"""
    Exponentially-Weighted Standard Deviation over a window.

    The square root of the exponentially-weighted :func:`variance_ewma` — the spread of the input around its
    exponentially-weighted mean, in the same units as the input, with recent observations weighted more heavily
    (smoothing factor :math:`\alpha = 2 / (\text{window} + 1)`, the same span convention as :func:`ema`):

    .. math::

        \sigma^{\mathrm{ewm}}_t = \sqrt{\mathrm{Var}^{\mathrm{ewm}}_t}.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Span of the exponential weighting, mapped to ``alpha = 2 / (window + 1)``. Must be ``>= 2``.
        adjust: When ``False`` (default) use the recursive form; when ``True`` use the finite-window bias-corrected
            weighting (the same flag as :func:`ema`).
        bias: When ``True`` (default) the population standard deviation; when ``False`` the unbiased sample one.
            ``True`` mirrors the ``ddof = 0`` default of :func:`standard_deviation_rolling`. See :func:`variance_ewma`.

    Returns:
        The exponentially-weighted standard deviation for each row, the same length as the input. The first
        ``window - 1`` values are ``null`` (warm-up): the recursion emits only once ``window`` non-null observations
        have been seen.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        It is homogeneous of degree ``1`` in ``expr`` (a spread in the input's own units).

        **Edge-case behavior:**

        - **Null** — a leading ``null`` run stays ``null`` and does not consume warm-up; an interior ``null`` yields
          ``null`` at that row while the weights decay across the gap (``ignore_nulls=False``).
        - **NaN** — a ``NaN`` poisons the recursion and yields ``NaN`` for every subsequent non-null row.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the recursion re-seeds per
          series, e.g. ``standard_deviation_ewma(pl.col("close"), 20).over("ticker")``.

    See Also:
        - :func:`variance_ewma`: The square of this, of which it is the root.
        - :func:`standard_deviation_rolling`: The equal-weighted (rolling-window) counterpart.
        - :func:`ema`: The exponential mean these deviations are measured from.

    References:
        - https://en.wikipedia.org/wiki/Exponentially_weighted_moving_average

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import standard_deviation_ewma
        >>>
        >>> frame = pl.DataFrame({"x": [10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0]})
        >>> frame.select(standard_deviation_ewma(pl.col("x"), 3).round(4).alias("std"))["std"].to_list()
        [None, None, 1.299, 0.927, 1.2484, 0.8833, 1.1923]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {"ticker": ["A"] * 4 + ["B"] * 4, "x": [10.0, 11.0, 13.0, 12.0, 20.0, 22.0, 21.0, 24.0]}
        ... )
        >>> expr = standard_deviation_ewma(pl.col("x"), 3).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("std"))["std"].to_list()
        [None, None, 1.299, 0.927, None, None, 0.7071, 1.5811]

        A ``null`` (decays across the gap) and a ``NaN`` (which propagates) make the handling visible:

        >>> frame = pl.DataFrame({"x": [10.0, 11.0, 13.0, None, 14.0, float("nan"), 16.0, 17.0]})
        >>> frame.select(standard_deviation_ewma(pl.col("x"), 3).round(4).alias("std"))["std"].to_list()
        [None, None, 1.299, None, 1.299, nan, nan, nan]
    """
    expr = float64_expr(expr)
    validate_window(window, minimum=2)
    return expr.ewm_std(span=window, adjust=adjust, bias=bias, min_samples=window)


def standard_deviation_rolling(
    expr: pl.Expr,
    window: int,
    *,
    ddof: int = 0,
) -> pl.Expr:
    r"""
    Rolling Standard Deviation over a window.

    The square root of the rolling :func:`variance_rolling` — a measure of how widely the values in each window
    spread around their mean, in the same units as the input:

    .. math::

        \sigma_t = \sqrt{\mathrm{Var}_t}
            = \sqrt{\frac{1}{n - \mathrm{ddof}} \sum_{i=t-n+1}^{t} \bigl(x_i - \bar{x}_t\bigr)^2},
            \qquad n = \text{window}.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of observations in the moving window. Must be ``>= 1``.
        ddof: Delta degrees of freedom — the divisor is ``window - ddof``. ``0`` (default) is the **population**
            standard deviation; ``1`` is the **sample** standard deviation. Must be ``< window``. See
            :func:`variance_rolling`.

    Returns:
        The rolling standard deviation for each row, the same length as the input. The first ``window - 1`` values are
        ``null`` (warm-up): the window must hold ``window`` non-null values before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``, ``ddof < 0``, or if ``ddof >= window`` (the divisor ``window - ddof`` would be
            non-positive).

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        **Degrees of freedom:**

        ``ddof`` carries the same meaning as in :func:`variance_rolling` (population vs sample); the standard
        deviation is just its square root. It must be strictly below ``window`` so the divisor stays positive.

        **Edge-case behavior:**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **window == 1** — a single value has no spread, so the result is ``0`` with the default ``ddof = 0``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the window never spans series
          boundaries, e.g. ``standard_deviation_rolling(pl.col("close"), 20).over("ticker")``.

    See Also:
        - :func:`variance_rolling`: The square of this, of which it is the root.
        - :func:`sma`: The moving mean the deviations are measured from.
        - :func:`bollinger_bands`: Volatility bands placed a multiple of this standard deviation around the mean.

    References:
        - https://en.wikipedia.org/wiki/Standard_deviation

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import standard_deviation_rolling
        >>>
        >>> frame = pl.DataFrame({"x": [10.0, 11.0, 12.0, 11.0, 13.0]})
        >>> frame.select(standard_deviation_rolling(pl.col("x"), 3).round(4).alias("std"))["std"].to_list()
        [None, None, 0.8165, 0.4714, 0.8165]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame({"ticker": ["A"] * 3 + ["B"] * 3, "x": [10.0, 11.0, 12.0, 20.0, 22.0, 21.0]})
        >>> expr = standard_deviation_rolling(pl.col("x"), 2).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("std"))["std"].to_list()
        [None, 0.5, 0.5, None, 1.0, 0.5]

        A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make the handling visible:

        >>> frame = pl.DataFrame({"x": [10.0, None, 12.0, float("nan"), 14.0, 15.0]})
        >>> frame.select(standard_deviation_rolling(pl.col("x"), 2).round(4).alias("std"))["std"].to_list()
        [None, None, None, nan, nan, 0.5]
    """
    expr = float64_expr(expr)
    validate_window(window)
    validate_ddof(ddof, window)
    # Native Polars rolling std; ddof=0 (population) is the charting default, not Polars' own sample ddof=1.
    return expr.rolling_std(window, ddof=ddof)


def time_series_forecast(
    expr: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Time Series Forecast (the rolling least-squares line extrapolated one bar ahead).

    The ordinary-least-squares line fitted to the last ``window`` observations, evaluated one bar beyond the window --
    a one-step-ahead projection of the fitted trend:

    .. math::

        \mathrm{TSF}_t = \bar{x}_t + \mathrm{slope}_t \cdot \frac{n + 1}{2}, \qquad n = \text{window},

    with :math:`\bar{x}_t` the window mean and :math:`\mathrm{slope}_t` the rolling :func:`linear_regression_slope`. It
    is exactly one slope step beyond :func:`linear_regression` (the line at the current bar).

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of observations in the regression window. Must be ``>= 2`` (a line needs at least two points).

    Returns:
        The one-step-ahead forecast for each row, the same length as the input. The first ``window - 1`` values are
        ``null`` (warm-up).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        It is homogeneous of degree ``1`` in ``expr`` (a projected price scales with the price). For a perfectly linear
        input the forecast equals the next value of the line exactly.

        **Edge-case behavior:**

        - **Null** — a window containing a ``null`` yields ``null``.
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the window never spans series
          boundaries, e.g. ``time_series_forecast(pl.col("close"), 14).over("ticker")``.

    See Also:
        - :func:`linear_regression`: The same line evaluated at the current bar rather than one ahead.
        - :func:`linear_regression_slope`: The slope used for the projection.
        - :func:`linear_regression_intercept`: The same line's value at the oldest bar of the window.

    References:
        - https://en.wikipedia.org/wiki/Simple_linear_regression
        - https://www.investopedia.com/terms/r/regression.asp

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import time_series_forecast
        >>>
        >>> frame = pl.DataFrame({"x": [10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0]})
        >>> frame.select(time_series_forecast(pl.col("x"), 3).round(4).alias("tsf"))["tsf"].to_list()
        [None, None, 14.3333, 13.0, 14.0, 14.0, 15.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {"ticker": ["A"] * 4 + ["B"] * 4, "x": [10.0, 11.0, 13.0, 12.0, 20.0, 22.0, 21.0, 24.0]}
        ... )
        >>> expr = time_series_forecast(pl.col("x"), 3).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("tsf"))["tsf"].to_list()
        [None, None, 14.3333, 13.0, None, None, 22.0, 24.3333]

        A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make the handling visible:

        >>> frame = pl.DataFrame({"x": [10.0, 11.0, 13.0, None, 14.0, float("nan"), 16.0]})
        >>> frame.select(time_series_forecast(pl.col("x"), 3).round(4).alias("tsf"))["tsf"].to_list()
        [None, None, 14.3333, None, None, None, nan]
    """
    expr = float64_expr(expr)
    validate_window(window, minimum=2)
    return expr.rolling_mean(window) + linear_regression_slope(expr, window) * (window + 1) / 2.0


def variance_ewma(
    expr: pl.Expr,
    window: int,
    *,
    adjust: bool = False,
    bias: bool = True,
) -> pl.Expr:
    r"""
    Exponentially-Weighted Variance over a window.

    The exponentially-weighted variance of the input around its exponentially-weighted mean — recent observations
    weighted more heavily, with the smoothing factor :math:`\alpha = 2 / (\text{window} + 1)` (the same span convention
    as :func:`ema`). The exponential counterpart of :func:`variance_rolling`:

    .. math::

        \mathrm{Var}^{\mathrm{ewm}}_t = \frac{\sum_i w_i \,(x_{t-i} - \bar{x}_t)^2}{\sum_i w_i},
            \qquad w_i = (1 - \alpha)^i,

    with :math:`\bar{x}_t` the exponentially-weighted mean (the weights decay by :math:`1 - \alpha` per step). The
    displayed weights :math:`w_i = (1 - \alpha)^i` are the ``adjust=True`` form; the default ``adjust=False`` instead
    uses the recursive weighting :math:`\alpha (1 - \alpha)^i`, with the oldest observation carrying :math:`(1 -
    \alpha)^t` — a different weighting that yields different numbers (the values in the Examples below are the default).

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Span of the exponential weighting, mapped to ``alpha = 2 / (window + 1)``. Must be ``>= 2``.
        adjust: When ``False`` (default) use the recursive form; when ``True`` use the finite-window bias-corrected
            weighting (the same flag as :func:`ema`).
        bias: When ``True`` (default) the population variance (divides by the weight total); when ``False`` the unbiased
            sample variance (the reliability correction ``1 - sum(w ** 2) / (sum w) ** 2``). ``True`` mirrors the
            ``ddof = 0`` default of :func:`variance_rolling`.

    Returns:
        The exponentially-weighted variance for each row, the same length as the input. The first ``window - 1`` values
        are ``null`` (warm-up): the recursion emits only once ``window`` non-null observations have been seen.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``.

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        ``window`` must be ``>= 2``: a single observation yields a well-defined ``0`` under the default ``bias=True``,
        but divides by zero under the unbiased ``bias=False`` correction, so a minimum of ``2`` is enforced uniformly
        across both paths. It is homogeneous of degree ``2`` in ``expr`` (a variance scales with the square of the
        input).

        **Edge-case behavior:**

        - **Null** — a leading ``null`` run stays ``null`` and does not consume warm-up; an interior ``null`` yields
          ``null`` at that row while the weights decay across the gap (``ignore_nulls=False``).
        - **NaN** — a ``NaN`` poisons the recursion and yields ``NaN`` for every subsequent non-null row.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the recursion re-seeds per
          series, e.g. ``variance_ewma(pl.col("close"), 20).over("ticker")``.

    See Also:
        - :func:`standard_deviation_ewma`: Its square root, in the input's own units.
        - :func:`variance_rolling`: The equal-weighted (rolling-window) counterpart.
        - :func:`ema`: The exponential mean these deviations are measured from.

    References:
        - https://en.wikipedia.org/wiki/Exponentially_weighted_moving_average

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import variance_ewma
        >>>
        >>> frame = pl.DataFrame({"x": [10.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0]})
        >>> frame.select(variance_ewma(pl.col("x"), 3).round(4).alias("var"))["var"].to_list()
        [None, None, 1.6875, 0.8594, 1.5586, 0.7803, 1.4216]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {"ticker": ["A"] * 4 + ["B"] * 4, "x": [10.0, 11.0, 13.0, 12.0, 20.0, 22.0, 21.0, 24.0]}
        ... )
        >>> expr = variance_ewma(pl.col("x"), 3).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("var"))["var"].to_list()
        [None, None, 1.6875, 0.8594, None, None, 0.5, 2.5]

        A ``null`` (decays across the gap) and a ``NaN`` (which propagates) make the handling visible:

        >>> frame = pl.DataFrame({"x": [10.0, 11.0, 13.0, None, 14.0, float("nan"), 16.0, 17.0]})
        >>> frame.select(variance_ewma(pl.col("x"), 3).round(4).alias("var"))["var"].to_list()
        [None, None, 1.6875, None, 1.6875, nan, nan, nan]
    """
    expr = float64_expr(expr)
    validate_window(window, minimum=2)
    return expr.ewm_var(span=window, adjust=adjust, bias=bias, min_samples=window)


def variance_rolling(
    expr: pl.Expr,
    window: int,
    *,
    ddof: int = 0,
) -> pl.Expr:
    r"""
    Rolling Variance over a window.

    The mean squared deviation of the values in each window from their window mean — a measure of dispersion in
    squared units of the input:

    .. math::

        \mathrm{Var}_t = \frac{1}{n - \mathrm{ddof}} \sum_{i=t-n+1}^{t} \bigl(x_i - \bar{x}_t\bigr)^2,
            \qquad \bar{x}_t = \frac{1}{n} \sum_{i=t-n+1}^{t} x_i, \qquad n = \text{window}.

    Args:
        expr: Input series, typically a price column (e.g. ``pl.col("close")``).
        window: Number of observations in the moving window. Must be ``>= 1``.
        ddof: Delta degrees of freedom — the divisor is ``window - ddof``. ``0`` (default) divides by ``window`` (the
            **population** variance); ``1`` divides by ``window - 1`` (the **sample** variance, the
            unbiased estimator used when the window is a sample of a larger population). Must be ``< window`` (the
            divisor ``window - ddof`` must be positive).

    Returns:
        The rolling variance for each row, the same length as the input. The first ``window - 1`` values are ``null``
        (warm-up): the window must hold ``window`` non-null values before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``, ``ddof < 0``, or if ``ddof >= window`` (the divisor ``window - ddof`` would be
            non-positive).

    Note:
        **Precision** -- agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on
        any finite input within a sane dynamic range; ``CORRECTNESS.md`` gives the method and the float-conditioning
        limit beyond it.

        **Degrees of freedom:**

        ``ddof`` selects the divisor ``window - ddof``: ``ddof = 0`` is the population variance (÷ ``window``), the
        charting convention; ``ddof = 1`` is the sample variance (÷ ``window - 1``), Bessel's unbiased
        estimator. The two differ by the factor ``window / (window - ddof)`` — e.g. on ``[10, 11, 12]`` the population
        variance is ``0.6667`` and the sample variance is ``1.0``. ``ddof`` must be strictly below ``window`` so the
        divisor stays positive.

        **Edge-case behavior:**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **window == 1** — a single value has no spread, so the result is ``0`` with the default ``ddof = 0``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the window never spans series
          boundaries, e.g. ``variance_rolling(pl.col("close"), 20).over("ticker")``.

    See Also:
        - :func:`standard_deviation_rolling`: Its square root, in the input's own units.
        - :func:`variance_ewma`: The exponentially-weighted counterpart, weighting recent observations more.
        - :func:`sma`: The moving mean the deviations are measured from.

    References:
        - https://en.wikipedia.org/wiki/Variance

    Examples:
        >>> import polars as pl
        >>> from pomata.indicators import variance_rolling
        >>>
        >>> frame = pl.DataFrame({"x": [10.0, 11.0, 12.0, 11.0, 13.0]})
        >>> frame.select(variance_rolling(pl.col("x"), 3).round(4).alias("variance"))["variance"].to_list()
        [None, None, 0.6667, 0.2222, 0.6667]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame({"ticker": ["A"] * 3 + ["B"] * 3, "x": [10.0, 11.0, 12.0, 20.0, 22.0, 21.0]})
        >>> expr = variance_rolling(pl.col("x"), 2).over("ticker").round(4)
        >>> frame.with_columns(expr.alias("variance"))["variance"].to_list()
        [None, 0.25, 0.25, None, 1.0, 0.25]

        A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make the handling visible:

        >>> frame = pl.DataFrame({"x": [10.0, None, 12.0, float("nan"), 14.0, 15.0]})
        >>> frame.select(variance_rolling(pl.col("x"), 2).round(4).alias("variance"))["variance"].to_list()
        [None, None, None, nan, nan, 0.25]
    """
    expr = float64_expr(expr)
    validate_window(window)
    validate_ddof(ddof, window)
    # Native Polars rolling var; ddof=0 (population) is the charting default, not Polars' own sample ddof=1.
    return expr.rolling_var(window, ddof=ddof)
