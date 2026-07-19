"""
Drawdown metrics — the running drawdown series and its peak, depth, duration, and tail summaries, read from an equity
curve.
"""

import polars as pl

from pomata._expr import float64_expr, validate_confidence, validate_window

__all__ = (
    "conditional_drawdown_at_risk",
    "drawdown",
    "drawdown_rolling",
    "max_drawdown",
    "max_drawdown_duration",
    "pain_index",
    "ulcer_index",
)


def conditional_drawdown_at_risk(
    equity_curve: pl.Expr,
    *,
    confidence: float = 0.95,
) -> pl.Expr:
    r"""
    Conditional Drawdown at Risk (CDaR), the mean of the worst drawdowns beyond a confidence level.

    The Rockafellar-Uryasev average of the worst ``1 - confidence`` of drawdowns -- the expected depth of the deepest
    drawdowns, the drawdown analog of conditional value-at-risk. With the drawdowns sorted ascending and
    :math:`k = (1 - c)\,n`, the worst :math:`\lfloor k \rfloor` are averaged in full and the next carries the fractional
    weight :math:`k - \lfloor k \rfloor`:

    .. math::

        \mathrm{CDaR}_{c} = \frac{1}{k} \left( \sum_{i=1}^{\lfloor k \rfloor} D_{(i)}
        + (k - \lfloor k \rfloor)\, D_{(\lfloor k \rfloor + 1)} \right), \qquad
        D_i = \frac{E_i}{\max_{j \le i} E_j} - 1, \quad k = (1 - c)\, n,

    where :math:`D_{(1)} \le D_{(2)} \le \dots` are the order statistics of the drawdown series and :math:`c` is
    ``confidence``. The fractional boundary weight makes the estimator continuous in the data. The value is non-positive
    (a drawdown).

    Args:
        equity_curve: Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.
        confidence: The tail confidence level (canonically ``0.95``); the mean is taken over the worst
            ``1 - confidence`` of drawdowns. Must be in the open interval ``(0, 1)``.

    Returns:
        A single ``Float64`` value: the conditional drawdown at risk (one value in ``select``, one per group under
        ``.over``). ``null`` when there are no observations.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``confidence`` is not in the open interval ``(0, 1)``.

    Note:
        **Correctness**

        The result is checked against an independent reference oracle on every input, and every edge case (missing data
        and boundaries) is given a defined behavior.

        **Edge-case behavior**

        - **Null** — a ``null`` equity is skipped; an all-null (or empty) series yields ``null`` (the running peak
          carries across it).
        - **NaN** — a ``NaN`` equity propagates, yielding ``NaN``.
        - **Insufficient sample** — a single observation is trivially at its own peak, so the result is exactly ``0``,
          not ``null``.
        - **Degenerate denominator** — a monotonically non-decreasing curve has an all-zero drawdown series, so the
          result is ``0`` (not a ``0 / 0``).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`max_drawdown`: The single worst drawdown.
        - :func:`conditional_value_at_risk`: The return-space analog (expected shortfall).
        - :func:`pain_index`: The full-sample mean drawdown, against this worst-tail mean.

    References:
        - Chekhlov, A., Uryasev, S. & Zabarankin, M. (2005). "Drawdown Measure in Portfolio Optimization."
          *International Journal of Theoretical and Applied Finance*, 8(1), 13-58.
        - https://doi.org/10.1142/S0219024905002767
        - https://en.wikipedia.org/wiki/Drawdown_%28economics%29

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import conditional_drawdown_at_risk
        >>>
        >>> frame = pl.DataFrame({"equity": [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4]})
        >>> frame.select(conditional_drawdown_at_risk(pl.col("equity")).round(4)).item()
        -0.0455

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 7 + ["NVDA"] * 7,
        ...         "equity": [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4] + [1.0, 0.9, 0.95, 1.1, 1.0, 1.2, 1.15],
        ...     }
        ... )
        >>> expr = conditional_drawdown_at_risk(pl.col("equity"))
        >>> frame.select(conditional_drawdown_at_risk=expr.over("ticker").round(4))[
        ...     "conditional_drawdown_at_risk"
        ... ].unique().sort().to_list()
        [-0.1, -0.0455]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"equity": [1.1, 1.05, None, 1.2, float("nan"), 1.15, 1.3]})
        >>> frame.select(conditional_drawdown_at_risk(pl.col("equity")).round(4)).item()
        nan

        **Insufficient sample** — a one-element series is at its own peak, so CDaR is exactly ``0``:

        >>> frame = pl.DataFrame({"equity": [1.0]})
        >>> frame.select(conditional_drawdown_at_risk(pl.col("equity"))).item()
        0.0

        **Degenerate denominator** — a monotonically rising curve has an all-zero drawdown series, so CDaR is ``0``:

        >>> frame = pl.DataFrame({"equity": [1.0, 1.1, 1.21]})
        >>> frame.select(conditional_drawdown_at_risk(pl.col("equity"))).item()
        0.0
    """
    equity_curve = float64_expr(equity_curve)
    validate_confidence(confidence)
    declines = drawdown(equity_curve)
    # Rockafellar-Uryasev empirical tail average of the drawdowns (as in :func:`conditional_value_at_risk`): the worst
    # ``1 - confidence`` of drawdowns, weighting the boundary order statistic by the fractional part of
    # ``k = (1 - confidence) * n`` -- ``weight = clip(k - rank, 0, 1)`` -- so the estimator is continuous in the data.
    rank = declines.rank(method="ordinal") - 1
    count = declines.count()
    weight = ((1.0 - confidence) * count - rank).clip(lower_bound=0.0, upper_bound=1.0)
    tail_mean = (weight * declines).sum() / weight.sum()
    # An empty (or all-null) series has ``weight.sum() == 0``; report ``null`` rather than the ``0 / 0`` NaN.
    tail_mean = pl.when(count == 0).then(pl.lit(None, dtype=pl.Float64)).otherwise(tail_mean)
    return pl.when(equity_curve.is_nan().any()).then(float("nan")).otherwise(tail_mean).name.keep()


def drawdown(
    equity_curve: pl.Expr,
) -> pl.Expr:
    r"""
    Drawdown, the running fractional decline of an equity curve from its prior peak.

    For each row, the equity relative to the highest equity seen up to that row, minus one — the fraction by which the
    curve is currently below its running peak (``0`` at a new high, negative while underwater):

    .. math::

        D_t = \frac{E_t}{\max_{i \le t} E_i} - 1 \le 0,

    where :math:`E` is the equity curve. Unlike the other metrics this one is **series-valued** (one drawdown per row),
    so it is the natural input to a custom drawdown analysis; :func:`max_drawdown` and :func:`ulcer_index` summarize it.

    Args:
        equity_curve: Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.

    Returns:
        The drawdown for each row, the same length as ``equity_curve`` -- ``0`` at a running peak and negative while
        below it. A leading input ``null`` stays ``null``.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness**

        The result is checked against an independent reference oracle on every input, and every edge case (missing data
        and boundaries) is given a defined behavior.

        **Inception**

        The running peak starts at the FIRST observation: a curve fed from :func:`~pomata.pnl.equity_curve` begins at
        its first post-return value, so a drawdown from the starting capital itself (an opening losing streak) is
        invisible by construction. Prepend a literal ``1.0`` row to count declines from inception; the convention
        matches quantstats (empyrical instead prepends the start).

        **Edge-case behavior**

        - **Null** — a ``null`` equity makes that row ``null`` (``null`` takes precedence over ``NaN``); the running
          peak carries across it unchanged.
        - **NaN** — a ``NaN`` equity yields ``NaN`` for that row; the running peak ignores it (Polars' ``cum_max``
          semantics), so later rows are unaffected.
        - **Insufficient sample** — a single-row series is trivially at its own peak, so its (only) drawdown is exactly
          ``0``, not ``null``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`max_drawdown`: The deepest point of this series.
        - :func:`ulcer_index`: The root-mean-square of this series.
        - :func:`drawdown_rolling`: The trailing-window form, healed once an old peak rolls out.

    References:
        - https://en.wikipedia.org/wiki/Drawdown_%28economics%29

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import drawdown
        >>>
        >>> frame = pl.DataFrame({"equity": [1.0, 1.1, 1.05, 1.2, 0.9, 1.0]})
        >>> frame.select(drawdown=drawdown(pl.col("equity")).round(4))["drawdown"].to_list()
        [0.0, 0.0, -0.0455, 0.0, -0.25, -0.1667]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker's running peak restarts independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 7 + ["NVDA"] * 7,
        ...         "equity": [1.0, 1.1, 1.05, 1.2, 0.9, 1.0, 1.1] + [1.0, 0.95, 1.05, 1.0, 1.15, 1.1, 1.2],
        ...     }
        ... )
        >>> frame.with_columns(drawdown=drawdown(pl.col("equity")).over("ticker").round(4))["drawdown"].to_list()
        [0.0, 0.0, -0.0455, 0.0, -0.25, -0.1667, -0.0833, 0.0, -0.05, 0.0, -0.0476, 0.0, -0.0435, 0.0]

        A ``null`` (skipped) and a ``NaN`` (which propagates at its row) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"equity": [1.0, 1.1, None, 1.2, float("nan"), 1.0]})
        >>> frame.select(drawdown=drawdown(pl.col("equity")).round(4))["drawdown"].to_list()
        [0.0, 0.0, None, 0.0, nan, -0.1667]

        **Insufficient sample** — a one-element series is at its own peak, so the drawdown is ``0``:

        >>> frame = pl.DataFrame({"equity": [1.0]})
        >>> frame.select(drawdown=drawdown(pl.col("equity")))["drawdown"].to_list()
        [0.0]
    """
    equity_curve = float64_expr(equity_curve)
    return (equity_curve / equity_curve.cum_max() - 1).name.keep()


def drawdown_rolling(
    equity_curve: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Rolling Drawdown over a window — the decline from each trailing window's peak.

    The current equity relative to the highest equity in the trailing window (a non-positive fraction):

    .. math::

        D_t = \frac{E_t}{\max_{t-n+1 \le i \le t} E_i} - 1, \qquad n = \text{window},

    where :math:`E` is the equity curve. This is a DISTINCT quantity from the running :func:`drawdown`, whose peak is
    the all-time high to date; here the peak is only the trailing ``window``, so a decline "heals" once the old high
    rolls out of the window.

    Args:
        equity_curve: Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        The rolling drawdown for each row, the same length as the input. The first ``window - 1`` rows are ``null``
        (warm-up): the window must hold ``window`` non-null values before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Correctness**

        Each window matches an independent reference oracle (the current equity over the window peak, less one).

        **Edge-case behavior**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`drawdown`: The running form, measured against the all-time high to date.
        - :func:`max_drawdown`: The deepest all-time decline.
        - :func:`max_drawdown_duration`: The time dimension (longest underwater stretch).

    References:
        - https://en.wikipedia.org/wiki/Drawdown_%28economics%29

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import drawdown_rolling
        >>>
        >>> frame = pl.DataFrame({"equity": [1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25]})
        >>> expr = drawdown_rolling(pl.col("equity"), window=3)
        >>> frame.select(drawdown_rolling=expr.round(4))["drawdown_rolling"].to_list()
        [None, None, -0.0455, 0.0, -0.0417, 0.0, -0.0385]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker's window restarts independently and never
        spans the boundary:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 7 + ["NVDA"] * 7,
        ...         "equity": [1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25] + [1.0, 0.95, 1.05, 1.0, 1.15, 1.1, 1.2],
        ...     }
        ... )
        >>> expr = drawdown_rolling(pl.col("equity"), window=3)
        >>> frame.with_columns(drawdown_rolling=expr.over("ticker").round(4))["drawdown_rolling"].to_list()
        [None, None, -0.0455, 0.0, -0.0417, 0.0, -0.0385, None, None, 0.0, -0.0476, 0.0, -0.0435, 0.0]

        A leading ``null`` and a later ``NaN`` make the windowed handling visible: a window covering the ``null`` is
        ``null``, and the ``NaN`` poisons every window it enters:

        >>> frame = pl.DataFrame({"equity": [None, 1.1, 1.05, 1.2, float("nan"), 1.15, 1.3]})
        >>> expr = drawdown_rolling(pl.col("equity"), window=3)
        >>> frame.select(drawdown_rolling=expr.round(4))["drawdown_rolling"].to_list()
        [None, None, None, 0.0, nan, nan, nan]
    """
    equity_curve = float64_expr(equity_curve)
    validate_window(window)
    return (equity_curve / equity_curve.rolling_max(window, min_samples=window) - 1.0).name.keep()


def max_drawdown(
    equity_curve: pl.Expr,
) -> pl.Expr:
    r"""
    Maximum Drawdown, the deepest peak-to-trough decline of an equity curve.

    The most negative value of the running :func:`drawdown` — the worst fractional loss from a prior peak over the whole
    series:

    .. math::

        \mathrm{MDD} = \min_t \left( \frac{E_t}{\max_{i \le t} E_i} - 1 \right) \le 0.

    Args:
        equity_curve: Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.

    Returns:
        A single ``Float64`` value: the maximum drawdown (one value in ``select``, one per group under ``.over``). It is
        ``<= 0`` (``0`` for a never-declining curve); ``null`` when there are no observations.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness**

        The result is checked against an independent reference oracle on every input, and every edge case (missing data
        and boundaries) is given a defined behavior.

        **Edge-case behavior**

        - **Null** — a ``null`` equity is skipped; an all-null (or empty) series yields ``null`` (a missing bar does not
          start a drawdown).
        - **NaN** — a ``NaN`` equity propagates, yielding ``NaN`` (an undefined equity makes the worst-drawdown summary
          undefined).
        - **Insufficient sample** — a single observation is trivially at its own peak, so the result is exactly ``0``,
          not ``null``.
        - **Degenerate denominator** — a never-declining curve has zero drawdown throughout, so the result is ``0`` (not
          a ``0 / 0``).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`drawdown`: The running series this reduces.
        - :func:`calmar_ratio`: The return-over-drawdown ratio built on this.
        - :func:`max_drawdown_duration`: The duration dimension (longest underwater stretch).

    References:
        - https://en.wikipedia.org/wiki/Drawdown_%28economics%29

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import max_drawdown
        >>>
        >>> frame = pl.DataFrame({"equity": [1.0, 1.1, 1.05, 1.2, 0.9, 1.0]})
        >>> frame.select(max_drawdown(pl.col("equity")).round(4)).item()
        -0.25

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 7 + ["NVDA"] * 7,
        ...         "equity": [1.0, 1.1, 1.05, 1.2, 0.9, 1.0, 1.1] + [1.0, 0.95, 1.05, 1.0, 1.15, 1.1, 1.2],
        ...     }
        ... )
        >>> expr = max_drawdown(pl.col("equity"))
        >>> frame.select(max_drawdown=expr.over("ticker").round(4))["max_drawdown"].unique().sort().to_list()
        [-0.25, -0.05]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"equity": [1.0, 1.1, None, 1.2, float("nan"), 1.0]})
        >>> frame.select(max_drawdown(pl.col("equity")).round(4)).item()
        nan

        **Insufficient sample** — a one-element series is at its own peak, so the maximum drawdown is ``0``:

        >>> frame = pl.DataFrame({"equity": [1.0]})
        >>> frame.select(max_drawdown(pl.col("equity"))).item()
        0.0

        **Degenerate denominator** — a never-declining curve has zero drawdown:

        >>> frame = pl.DataFrame({"equity": [1.0, 1.1, 1.2, 1.3]})
        >>> frame.select(max_drawdown(pl.col("equity"))).item()
        0.0
    """
    equity_curve = float64_expr(equity_curve)
    declines = drawdown(equity_curve)
    return pl.when(equity_curve.is_nan().any()).then(float("nan")).otherwise(declines.min()).name.keep()


def max_drawdown_duration(
    equity_curve: pl.Expr,
) -> pl.Expr:
    r"""
    Maximum Drawdown Duration, the length of the longest underwater stretch (in bars).

    The greatest number of consecutive observations the equity spends below a prior peak -- the longest run of strictly
    negative drawdown, the time dimension of drawdown risk (returned as a ``Float64`` count of bars).

    Args:
        equity_curve: Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.

    Returns:
        A single ``Float64`` value: the longest underwater run length in bars (one value in ``select``, one per group
        under ``.over``). ``0`` when the curve never goes below a prior peak; ``null`` when there are no observations.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness**

        The result is checked against an independent reference oracle on every input, and every edge case (missing data
        and boundaries) is given a defined behavior.

        The duration is a count of observations, not a calendar span; with irregular spacing scale it by the bar period
        externally.

        **Edge-case behavior**

        - **Null** — a ``null`` equity is skipped; an all-null (or empty) series yields ``null`` (the run is measured
          over the retained observations, so a gap neither breaks nor extends the underwater stretch).
        - **NaN** — a ``NaN`` equity propagates, yielding ``NaN``.
        - **Insufficient sample** — a single observation is never underwater, so the result is exactly ``0``, not
          ``null``.
        - **Degenerate denominator** — a monotonically non-decreasing curve is never underwater, so the result is ``0``
          (not a ``0 / 0``).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`max_drawdown`: The depth dimension (worst decline).
        - :func:`drawdown`: The running series whose underwater runs this counts.
        - :func:`ulcer_index`: Penalizes prolonged declines, blending depth and duration.

    References:
        - https://en.wikipedia.org/wiki/Drawdown_%28economics%29

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import max_drawdown_duration
        >>>
        >>> frame = pl.DataFrame({"equity": [1.0, 0.9, 0.8, 0.85, 1.1, 1.05]})
        >>> frame.select(max_drawdown_duration(pl.col("equity"))).item()
        3.0

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 7 + ["NVDA"] * 7,
        ...         "equity": [1.0, 0.9, 0.8, 0.85, 1.1, 1.05, 1.2] + [1.0, 1.05, 0.95, 0.9, 1.1, 1.0, 1.2],
        ...     }
        ... )
        >>> expr = max_drawdown_duration(pl.col("equity"))
        >>> frame.select(max_drawdown_duration=expr.over("ticker").round(4))[
        ...     "max_drawdown_duration"
        ... ].unique().sort().to_list()
        [2.0, 3.0]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"equity": [1.0, 0.9, None, 0.85, float("nan"), 1.05]})
        >>> frame.select(max_drawdown_duration(pl.col("equity")).round(4)).item()
        nan

        **Insufficient sample** — a one-element series is never underwater, so the duration is ``0``:

        >>> frame = pl.DataFrame({"equity": [1.0]})
        >>> frame.select(max_drawdown_duration(pl.col("equity"))).item()
        0.0

        **Degenerate denominator** — a monotonically rising curve is never underwater, so the duration is ``0``:

        >>> frame = pl.DataFrame({"equity": [1.0, 1.1, 1.21]})
        >>> frame.select(max_drawdown_duration(pl.col("equity"))).item()
        0.0
    """
    equity_curve = float64_expr(equity_curve)
    defined = equity_curve.drop_nulls()
    underwater = defined / defined.cum_max() - 1 < 0
    running = underwater.cum_sum()
    reset = pl.when(~underwater).then(running).otherwise(None).forward_fill()
    longest = (running - reset).max().cast(pl.Float64)
    return pl.when(equity_curve.is_nan().any()).then(float("nan")).otherwise(longest).name.keep()


def pain_index(
    equity_curve: pl.Expr,
) -> pl.Expr:
    r"""
    Pain Index, the average depth of drawdown over the whole curve.

    The mean of the absolute drawdown across every observation -- the average distance below the running peak (the
    arithmetic-mean counterpart of the root-mean-square :func:`ulcer_index`):

    .. math::

        \mathrm{PI} = \frac{1}{n} \sum_{i=1}^{n} \lvert D_i \rvert, \qquad D_i = \frac{E_i}{\max_{j \le i} E_j} - 1.

    Args:
        equity_curve: Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.

    Returns:
        A single ``Float64`` value: the pain index (one value in ``select``, one per group under ``.over``). It is
        ``>= 0``; ``null`` when there are no observations.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness**

        The result is checked against an independent reference oracle on every input, and every edge case (missing data
        and boundaries) is given a defined behavior.

        **Edge-case behavior**

        - **Null** — a ``null`` equity is skipped; an all-null (or empty) series yields ``null`` (the running peak
          carries across it).
        - **NaN** — a ``NaN`` equity propagates, yielding ``NaN``.
        - **Insufficient sample** — a single observation is trivially at its own peak, so the result is exactly ``0``,
          not ``null``.
        - **Degenerate denominator** — a monotonically non-decreasing curve is never below its peak, so the result is
          ``0`` (not a ``0 / 0``).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`ulcer_index`: The root-mean-square counterpart.
        - :func:`pain_ratio`: The return-to-pain ratio built on this.
        - :func:`max_drawdown`: The single worst drawdown, against this average depth.

    References:
        - Becker, T. (2006). "The Pain Index and Pain Ratio." *Zephyr Associates*.

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import pain_index
        >>>
        >>> frame = pl.DataFrame({"equity": [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4]})
        >>> frame.select(pain_index(pl.col("equity")).round(4)).item()
        0.0179

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 7 + ["NVDA"] * 7,
        ...         "equity": [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4] + [1.0, 0.9, 0.95, 1.1, 1.0, 1.2, 1.15],
        ...     }
        ... )
        >>> expr = pain_index(pl.col("equity"))
        >>> frame.select(pain_index=expr.over("ticker").round(4))["pain_index"].unique().sort().to_list()
        [0.0179, 0.0404]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"equity": [1.1, 1.05, None, 1.2, float("nan"), 1.15, 1.3]})
        >>> frame.select(pain_index(pl.col("equity")).round(4)).item()
        nan

        **Insufficient sample** — a one-element series is at its own peak, so the pain index is exactly ``0``:

        >>> frame = pl.DataFrame({"equity": [1.0]})
        >>> frame.select(pain_index(pl.col("equity"))).item()
        0.0

        **Degenerate denominator** — a monotonically rising curve is never below its running peak, so the mean drawdown
        is ``0``:

        >>> frame = pl.DataFrame({"equity": [1.0, 1.1, 1.21]})
        >>> frame.select(pain_index(pl.col("equity"))).item()
        0.0
    """
    return (drawdown(equity_curve).abs().mean()).name.keep()


def ulcer_index(
    equity_curve: pl.Expr,
) -> pl.Expr:
    r"""
    Ulcer Index, the root-mean-square depth of an equity curve's drawdowns.

    Peter Martin's measure of downside risk (1987): the quadratic mean of the running :func:`drawdown`, which penalizes
    deep and prolonged declines more than shallow ones (unlike the single worst point of :func:`max_drawdown`):

    .. math::

        \mathrm{UI} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} D_t^2}, \qquad
        D_t = \frac{E_t}{\max_{i \le t} E_i} - 1,

    expressed as a fraction (not a percentage). Lower is better; it is ``0`` only for a never-declining curve.

    Args:
        equity_curve: Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.

    Returns:
        A single ``Float64`` value: the Ulcer Index (one value in ``select``, one per group under ``.over``). It is
        ``>= 0``; ``null`` when there are no observations.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness**

        The result is checked against an independent reference oracle on every input, and every edge case (missing data
        and boundaries) is given a defined behavior.

        **Edge-case behavior**

        - **Null** — a ``null`` equity is skipped; an all-null (or empty) series yields ``null`` (excluded from the
          mean).
        - **NaN** — a ``NaN`` equity propagates, yielding ``NaN``.
        - **Insufficient sample** — a single observation has no drawdown, so the result is exactly ``0``, not ``null``.
        - **Degenerate denominator** — a never-declining curve has all-zero drawdowns, so the result is ``0`` (not a
          ``0 / 0``).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its
          own history.

    See Also:
        - :func:`max_drawdown`: The single worst drawdown, which the Ulcer Index complements with a continuous measure.
        - :func:`ulcer_performance_ratio`: The return-over-Ulcer ratio built on this.
        - :func:`pain_index`: The arithmetic-mean counterpart of this root-mean-square.

    References:
        - Martin, P. G. & McCann, B. B. (1989). *The Investor's Guide to Fidelity Funds*. Wiley.
        - https://en.wikipedia.org/wiki/Ulcer_index

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import ulcer_index
        >>>
        >>> frame = pl.DataFrame({"equity": [1.0, 1.1, 1.05, 1.2, 0.9, 1.0]})
        >>> frame.select(ulcer_index(pl.col("equity")).round(4)).item()
        0.1241

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["AAPL"] * 7 + ["NVDA"] * 7,
        ...         "equity": [1.0, 1.1, 1.05, 1.2, 0.9, 1.0, 1.1] + [1.0, 0.95, 1.05, 1.0, 1.15, 1.1, 1.2],
        ...     }
        ... )
        >>> expr = ulcer_index(pl.col("equity"))
        >>> frame.select(ulcer_index=expr.over("ticker").round(4))["ulcer_index"].unique().sort().to_list()
        [0.0308, 0.1191]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"equity": [1.0, 1.1, None, 1.2, float("nan"), 1.0]})
        >>> frame.select(ulcer_index(pl.col("equity")).round(4)).item()
        nan

        **Insufficient sample** — a one-element series has no drawdown, so the Ulcer Index is ``0``:

        >>> frame = pl.DataFrame({"equity": [1.0]})
        >>> frame.select(ulcer_index(pl.col("equity"))).item()
        0.0

        **Degenerate denominator** — a never-declining curve has all-zero drawdowns, so the Ulcer Index is exactly
        ``0``:

        >>> frame = pl.DataFrame({"equity": [1.0, 1.1, 1.2, 1.3]})
        >>> frame.select(ulcer_index(pl.col("equity"))).item()
        0.0
    """
    declines = drawdown(equity_curve)
    return (declines**2).mean().sqrt().name.keep()
