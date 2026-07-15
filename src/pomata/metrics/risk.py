"""
Risk and dispersion metrics — volatility, downside deviation, distribution shape, and tail risk.
"""

import math
from collections import deque
from statistics import NormalDist
from typing import Final, Literal

import polars as pl

from pomata._expr import (
    float64_expr,
    rolling_has_nan,
    rolling_is_constant,
    validate_confidence,
    validate_finite,
    validate_periods_per_year,
    validate_window,
)

__all__ = (
    "conditional_value_at_risk",
    "downside_deviation",
    "downside_deviation_rolling",
    "kelly_criterion",
    "kurtosis",
    "kurtosis_rolling",
    "payoff_ratio",
    "profit_factor",
    "risk_of_ruin",
    "skewness",
    "skewness_rolling",
    "tail_ratio",
    "tail_ratio_rolling",
    "value_at_risk",
    "value_at_risk_modified",
    "value_at_risk_parametric",
    "value_at_risk_rolling",
    "volatility",
    "volatility_rolling",
    "win_rate",
)

# The smallest number of observations a sample (ddof = 1) standard deviation is defined over.
_MINIMUM_SAMPLE_OBSERVATIONS: Final = 2


def _rolling_moment(
    expr: pl.Expr,
    window: int,
    *,
    kind: Literal["skew", "kurtosis"],
) -> pl.Expr:
    """
    The native rolling skewness or excess kurtosis over a window, guarded against empty, NaN, and constant windows.

    Routes to Polars' native ``rolling_skew`` / ``rolling_kurtosis`` -- both mean-centered, so they stay numerically
    stable on a *near*-constant (non-bit-identical) window far from zero, where the one-pass raw-moment form
    (``E[x^k]`` less the mean terms) cancels catastrophically. Those two native kernels panic on a length-0 series
    (unlike ``rolling_var`` / ``rolling_std``), and a ``pl.when`` length guard cannot prevent it (both branches
    evaluate), so the empty case is short-circuited inside a thin ``map_batches`` wrapper.

    The native kernels are *incremental*: their running sums carry every value that ever passed through the window,
    so once a value much larger than the current window's scale exits, the sums keep a stale residue and every later
    window's moment is silently wrong -- permanently, not just on the exit bar (one bad tick in a returns column is
    enough). The wrapper therefore tracks the largest magnitude seen so far against each window's own scale (a
    monotonic-deque sliding maximum, O(n)) and recomputes the windows at risk exactly, with a fresh two-pass
    mean-centered moment; clean series never leave the native fast path. A *bit*-constant window has zero variance,
    so the standardized moment is ``0 / 0`` = ``NaN``; that and the NaN-bearing window are still forced explicitly,
    mirroring :func:`volatility_rolling`. The population (biased) convention matches the reducing :func:`skewness` /
    :func:`kurtosis`, and warm-up, ``null``-in-window, and ``NaN`` propagation follow the native rolling policy.
    """

    def _exact(values: list[float | None]) -> float | None:
        if any(value is None for value in values):
            return None
        observations = [value for value in values if value is not None]
        if any(not math.isfinite(value) for value in observations):
            return math.nan
        mean = math.fsum(observations) / len(observations)
        centered = [value - mean for value in observations]
        second = math.fsum(delta**2 for delta in centered) / len(observations)
        if second == 0.0:
            return math.nan
        if kind == "skew":
            third = math.fsum(delta**3 for delta in centered) / len(observations)
            return third / math.pow(second, 1.5)
        fourth = math.fsum(delta**4 for delta in centered) / len(observations)
        return fourth / (second * second) - 3.0

    def _kernel(series: pl.Series) -> pl.Series:
        if series.len() == 0:
            return pl.Series(series.name, [], dtype=pl.Float64)
        if kind == "skew":
            native = series.rolling_skew(window, bias=True)
        else:
            native = series.rolling_kurtosis(window, fisher=True, bias=True)
        values = series.to_list()
        result = native.to_list()
        # Sliding maximum of |x| over the trailing window via a monotonic deque (O(n)); prefix_scale tracks the
        # largest magnitude the incremental sums have ever absorbed. The residue the sums keep after that value
        # exits scales like ulp(prefix^k) against a window moment of order window_scale^k, i.e. a relative error of
        # ~2^-52 * ratio^k for the k-th moment: at the kurtosis' k = 4, a ratio of 1e2 already leaves ~2e-8 — a
        # visible bite out of the 1e-7 absolute band the suite holds these standardized moments to — so the trigger
        # sits at 1e1; the worst boundary residue is ~2e-9, well inside that band. Flagged windows are recomputed
        # exactly; clean series never
        # leave the native fast path. The residue class is the native kernels' own stale-accumulator defect
        # (pola-rs/polars#28290, fixed upstream in #28309): once the polars floor includes that fix, this recompute
        # pass can be dropped. Below the trigger, an exit of at most one order of magnitude onto a window whose own
        # spread has collapsed under the suite's conditioning floor can still leave a wrong finite moment — that
        # regime is the tail of the same upstream defect, sits outside the fuzz-tested domain, and is inherited, not
        # guarded, here.
        deque_indices: deque[int] = deque()
        prefix_scale = 0.0
        for index, value in enumerate(values):
            magnitude = abs(value) if value is not None and math.isfinite(value) else 0.0
            prefix_scale = max(prefix_scale, magnitude)
            while deque_indices and deque_indices[0] <= index - window:
                deque_indices.popleft()
            while deque_indices:
                tail = values[deque_indices[-1]]
                tail_magnitude = abs(tail) if tail is not None and math.isfinite(tail) else 0.0
                if tail_magnitude <= magnitude:
                    deque_indices.pop()
                else:
                    break
            deque_indices.append(index)
            if index < window - 1:
                continue
            head = values[deque_indices[0]]
            window_scale = abs(head) if head is not None and math.isfinite(head) else 0.0
            if prefix_scale > window_scale * 1e1:
                result[index] = _exact(values[index - window + 1 : index + 1])
        return pl.Series(series.name, result, dtype=pl.Float64)

    moment = expr.map_batches(_kernel, return_dtype=pl.Float64)
    # A NaN window stays NaN; a bit-constant window has zero variance, so the standardized moment is 0/0 -> NaN. The
    # native incremental kernel can leave a residue (a spuriously huge finite) once a much larger value exits the
    # window -- a window constant only by sliding -- so guard both explicitly, mirroring volatility_rolling.
    return (
        pl.when(rolling_has_nan(expr, window))
        .then(float("nan"))
        .when(rolling_is_constant(expr, window))
        .then(float("nan"))
        .otherwise(moment)
        .name.keep()
    )


def _rolling_downside_deviation(
    expr: pl.Expr,
    window: int,
    periods_per_year: int,
    threshold: float,
) -> pl.Expr:
    """
    The rolling annualized downside deviation, with the no-downside window forced to exactly zero.

    The shortfall ``min(r - threshold, 0)`` is ``<= 0``, so a window's rolling minimum of it is ``0`` exactly when the
    window has no downside; in that case the mean-square is forced to ``0`` rather than the tiny residue Polars'
    sliding-window ``rolling_mean`` can leave once a large value exits the window (which would otherwise make
    :func:`sortino_ratio_rolling` report a huge finite instead of ``+/-inf``). Shared by
    :func:`downside_deviation_rolling` and :func:`sortino_ratio_rolling`.
    """
    shortfall = (expr - threshold).clip(upper_bound=0.0)
    mean_square = (shortfall**2).rolling_mean(window, min_samples=window)
    no_downside = shortfall.rolling_min(window, min_samples=window) == 0.0
    safe_mean_square = pl.when(no_downside).then(0.0).otherwise(mean_square).clip(lower_bound=0.0)
    return safe_mean_square.sqrt() * math.sqrt(periods_per_year)


def _dispersion(
    expr: pl.Expr,
) -> pl.Expr:
    """
    The sample standard deviation with an exactly-constant series pinned to exactly ``0.0``.

    ``std(ddof=1)`` of a bit-identical series is mathematically zero, but the chunked mean can round a non-dyadic
    constant (e.g. ``0.01``) a few ULP away from its own values, leaving a ~1e-18 residue that a downstream ratio then
    amplifies into a spuriously huge finite instead of the documented ``+/-inf``. Pin the constant case — two or more
    observations, no ``NaN``, one distinct non-null value — to exactly zero; a single observation keeps the native
    ``null`` (the sample deviation is undefined) and a ``NaN`` keeps poisoning through the native path. The core of
    :func:`volatility`, through which (at ``periods_per_year=1``) every ratio whose zero-dispersion contract is a
    signed infinity — the Sharpe family's denominator, the probabilistic Sharpe's inner Sharpe, the information
    ratio's tracking error — reaches the same pinning.
    """
    constant = (
        (expr.count() >= _MINIMUM_SAMPLE_OBSERVATIONS)
        & expr.is_nan().any().not_()
        & (expr.drop_nulls().n_unique() == 1)
    )
    return pl.when(constant).then(0.0).otherwise(expr.std(ddof=1))


def conditional_value_at_risk(
    returns: pl.Expr,
    *,
    confidence: float = 0.95,
) -> pl.Expr:
    r"""
    Historical Conditional Value-at-Risk (a.k.a. Expected Shortfall), the average of the worst-tail losses.

    The Rockafellar-Uryasev average of the worst ``1 - confidence`` of returns -- a coherent risk measure that reflects
    the severity of losses in the tail, not merely their cutoff. With the returns sorted ascending and
    :math:`k = (1 - c)\,n`, the worst :math:`\lfloor k \rfloor` are averaged in full and the next order statistic
    carries the fractional weight :math:`k - \lfloor k \rfloor`:

    .. math::

        \mathrm{CVaR}_{c} = \frac{1}{k} \left( \sum_{i=1}^{\lfloor k \rfloor} r_{(i)}
        + (k - \lfloor k \rfloor)\, r_{(\lfloor k \rfloor + 1)} \right), \qquad k = (1 - c)\, n,

    where :math:`r_{(1)} \le r_{(2)} \le \dots` are the order statistics of the ``n`` returns and :math:`c` is
    ``confidence``. The fractional boundary weight makes the estimator continuous in the data. The value is on the same
    scale as the returns and is negative for a loss.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        confidence: The tail confidence level (canonically ``0.95``); the shortfall is averaged over the worst
            ``1 - confidence`` of returns. Must be in the open interval ``(0, 1)``.

    Returns:
        A single ``Float64`` value: the expected shortfall (one value in ``select``, one per group under ``.over``).
        ``null`` when there are no returns.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``confidence`` is not in the open interval ``(0, 1)``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped (excluded from the count ``n`` and the tail average), so a leading
          warm-up ``null`` does not affect the result.
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **Historical, not parametric** — the shortfall is taken over the empirical return distribution, with no
          normality or other distributional assumption. An empty (or all-null) series yields ``null``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``conditional_value_at_risk(pl.col("returns")).over("ticker")``.

    See Also:
        - :func:`value_at_risk`: The tail cutoff quantile; this coherent average of the tail is always at least as deep.
        - :func:`conditional_drawdown_at_risk`: The same tail-averaging applied to the drawdown curve.
        - :func:`value_at_risk_parametric`: A parametric alternative to this historical tail estimate.

    References:
        - Rockafellar, R. T. & Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk." *Journal of Risk*, 2(3),
          21-41.
        - https://doi.org/10.21314/JOR.2000.038
        - https://en.wikipedia.org/wiki/Expected_shortfall

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import conditional_value_at_risk
        >>>
        >>> frame = pl.DataFrame({"returns": [0.03, -0.05, 0.02, -0.08, 0.01, -0.06, 0.04, -0.02]})
        >>> frame.select(conditional_value_at_risk(pl.col("returns"), confidence=0.75).round(4)).item()
        -0.07

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 8 + ["B"] * 8,
        ...         "returns": [
        ...             0.03,
        ...             -0.05,
        ...             0.02,
        ...             -0.08,
        ...             0.01,
        ...             -0.06,
        ...             0.04,
        ...             -0.02,
        ...             0.02,
        ...             -0.03,
        ...             0.05,
        ...             -0.04,
        ...             0.01,
        ...             -0.07,
        ...             0.03,
        ...             -0.01,
        ...         ],
        ...     }
        ... )
        >>> reduced = conditional_value_at_risk(pl.col("returns"), confidence=0.75).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [-0.07, -0.055]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.03, None, -0.05, 0.02, float("nan"), -0.08, 0.01, -0.06]})
        >>> frame.select(conditional_value_at_risk(pl.col("returns"), confidence=0.75).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    validate_confidence(confidence)
    # Rockafellar-Uryasev empirical expected shortfall: average the worst ``1 - confidence`` of returns, weighting the
    # boundary order statistic by the fractional part of ``k = (1 - confidence) * n`` (the 0-based rank-``r`` return
    # gets weight ``clip(k - r, 0, 1)``), so the estimator is continuous in the data (no quantile-tie knife-edge).
    rank = returns.rank(method="ordinal") - 1
    count = returns.count()
    weight = ((1.0 - confidence) * count - rank).clip(lower_bound=0.0, upper_bound=1.0)
    shortfall = (weight * returns).sum() / weight.sum()
    # An empty (or all-null) series has ``weight.sum() == 0``; report ``null`` rather than the ``0 / 0`` NaN.
    shortfall = pl.when(count == 0).then(pl.lit(None, dtype=pl.Float64)).otherwise(shortfall)
    return pl.when(returns.is_nan().any()).then(float("nan")).otherwise(shortfall).name.keep()


def downside_deviation(
    returns: pl.Expr,
    *,
    periods_per_year: int,
    threshold: float = 0.0,
) -> pl.Expr:
    r"""
    Annualized Downside Deviation, the dispersion of returns below a threshold.

    The root-mean-square of the returns' shortfall below a minimum acceptable return (the MAR), annualized by the
    square-root-of-time rule -- the downside-only counterpart of :func:`volatility` and the denominator of
    :func:`sortino_ratio`:

    .. math::

        \mathrm{DD}_{\mathrm{ann}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \min(r_i - \tau, 0)^2} \; \sqrt{P},

    where :math:`\tau` is ``threshold`` (the MAR), :math:`P` is ``periods_per_year``, and the mean runs over all
    :math:`n` returns -- returns at or above the threshold contribute zero, not nothing. Only downside dispersion is
    penalized, unlike the symmetric standard deviation.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        threshold: The **per-period** return level separating gains from losses / the minimum acceptable return
            (default ``0.0``); an annual target must be de-annualized by the caller before it is passed.
            Must be finite.

    Returns:
        A single ``Float64`` value: the annualized downside deviation (one value in ``select``, one per group under
        ``.over``). ``null`` when there are no returns.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``, or if ``threshold`` is not finite.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped (excluded from the mean), so a leading warm-up ``null`` does not
          affect the result.
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **No downside** — when every return is at or above the threshold the shortfall is all zero, so the result is
          ``0``; an empty (or all-null) series yields ``null``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the deviation is computed per
          series, e.g. ``downside_deviation(pl.col("returns"), periods_per_year=252).over("ticker")``.

    See Also:
        - :func:`sortino_ratio`: The risk-adjusted return that divides excess return by this.
        - :func:`volatility`: The symmetric (two-sided) dispersion.
        - :func:`downside_deviation_rolling`: The rolling (windowed) form.

    References:
        - Sortino, F. A. & Price, L. N. (1994). "Performance Measurement in a Downside Risk Framework." *The Journal of
          Investing*, 3(3), 59-64.
        - https://doi.org/10.3905/joi.3.3.59
        - https://en.wikipedia.org/wiki/Downside_risk

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import downside_deviation
        >>>
        >>> frame = pl.DataFrame({"returns": [0.02, -0.04, 0.01, -0.06, 0.03]})
        >>> frame.select(downside_deviation(pl.col("returns"), periods_per_year=252).round(4)).item()
        0.5119

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "returns": [0.02, -0.04, 0.01, -0.06, 0.03, 0.01, -0.02, 0.04, -0.03, 0.02],
        ...     }
        ... )
        >>> reduced = downside_deviation(pl.col("returns"), periods_per_year=252).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [0.256, 0.5119]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.02, None, -0.04, 0.01, float("nan"), -0.06, 0.03]})
        >>> frame.select(downside_deviation(pl.col("returns"), periods_per_year=252).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    validate_periods_per_year(periods_per_year)
    validate_finite(threshold, "threshold")
    shortfall = (returns - threshold).clip(upper_bound=0.0)
    return ((shortfall**2).mean().sqrt() * math.sqrt(periods_per_year)).name.keep()


def downside_deviation_rolling(
    returns: pl.Expr,
    window: int,
    *,
    periods_per_year: int,
    threshold: float = 0.0,
) -> pl.Expr:
    r"""
    Rolling Downside Deviation over a window — the windowed twin of :func:`downside_deviation`.

    The root-mean-square of the below-threshold shortfall over each trailing window, annualized by the
    square-root-of-time rule:

    .. math::

        \mathrm{DD}_t = \sqrt{\frac{1}{n} \sum_{i=t-n+1}^{t} \min(r_i - \tau, 0)^2}\,\sqrt{P}, \qquad n = \text{window},

    where :math:`\tau` is ``threshold`` and :math:`P` is ``periods_per_year``.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        window: Number of observations in the moving window. Must be ``>= 1``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        threshold: The **per-period** return level separating gains from losses / the minimum acceptable return
            (default ``0.0``); an annual target must be de-annualized by the caller before it is passed.
            Must be finite.

    Returns:
        The rolling downside deviation for each row, the same length as the input. The first ``window - 1`` rows are
        ``null`` (warm-up): the window must hold ``window`` non-null values before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``, ``periods_per_year < 1``, or if ``threshold`` is not finite.

    Note:
        **Correctness** -- each window matches an independent reference oracle (the reducing :func:`downside_deviation`
        recomputed over the window).

        **Edge-case behavior:**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **No downside** — a window with every return at or above the threshold has zero shortfall, so the result is
          exactly ``0``.
        - **Partitioning** — wrap the call in ``.over(...)`` so the window never spans series boundaries.

    See Also:
        - :func:`downside_deviation`: The whole-series reducing form.
        - :func:`sortino_ratio_rolling`: The risk-adjusted ratio that divides rolling excess return by this.
        - :func:`volatility_rolling`: The symmetric (two-sided) rolling dispersion.

    References:
        - Sortino, F. A. & Price, L. N. (1994). "Performance Measurement in a Downside Risk Framework." *The Journal of
          Investing*, 3(3), 59-64.
        - https://doi.org/10.3905/joi.3.3.59
        - https://en.wikipedia.org/wiki/Downside_risk

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import downside_deviation_rolling
        >>>
        >>> frame = pl.DataFrame({"returns": [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015]})
        >>> frame.select(
        ...     downside_deviation_rolling(pl.col("returns"), 3, periods_per_year=252).round(4)
        ... ).to_series().to_list()
        [None, None, 0.1833, 0.2049, 0.0917, 0.0917, 0.1375]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up on its own (the ``B`` group never
        borrows ``A``'s tail):

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [
        ...             0.01,
        ...             -0.02,
        ...             0.03,
        ...             -0.01,
        ...             0.02,
        ...             0.0,
        ...             -0.015,
        ...             0.02,
        ...             -0.01,
        ...             0.04,
        ...             -0.03,
        ...             0.01,
        ...             0.025,
        ...             -0.02,
        ...         ],
        ...     }
        ... )
        >>> rolled = downside_deviation_rolling(pl.col("returns"), 3, periods_per_year=252).over("ticker").round(4)
        >>> frame.select(rolled.alias("m"))["m"].to_list()
        [None, None, 0.1833, 0.2049, 0.0917, 0.0917, 0.1375, None, None, 0.0917, 0.2898, 0.275, 0.275, 0.1833]

        A leading ``null`` and a later ``NaN`` show the per-window masking, with the result recovering once both
        leave the window:

        >>> frame = pl.DataFrame({"returns": [None, 0.01, -0.02, float("nan"), 0.03, -0.01, 0.02]})
        >>> frame.select(
        ...     downside_deviation_rolling(pl.col("returns"), 3, periods_per_year=252).round(4)
        ... ).to_series().to_list()
        [None, None, None, nan, nan, nan, 0.0917]
    """
    returns = float64_expr(returns)
    validate_window(window)
    validate_periods_per_year(periods_per_year)
    validate_finite(threshold, "threshold")
    return _rolling_downside_deviation(returns, window, periods_per_year, threshold).name.keep()


def kelly_criterion(
    returns: pl.Expr,
) -> pl.Expr:
    r"""
    Kelly Criterion, the growth-optimal fraction of capital to risk per bet.

    The fraction that maximizes long-run logarithmic growth under the discrete win/loss model, from the win rate and the
    payoff ratio:

    .. math::

        f^{*} = p - \frac{1 - p}{W},

    where :math:`p` is the :func:`win_rate` (the probability of a win) and :math:`W` is the :func:`payoff_ratio` (the
    average win over the average loss).

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).

    Returns:
        A single ``Float64`` value: the Kelly fraction (one value in ``select``, one per group under ``.over``).
        ``null`` when the win rate or the payoff ratio is undefined (no decisive returns, or one-sided returns).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        This is the **discrete win/loss** form (from the win rate and payoff ratio). A common alternative for continuous
        returns is the ratio of the mean return to its variance; the two coincide only under specific assumptions.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped; an all-null (or empty) series yields ``null``.
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **One-sided / no decisive returns** — with no wins or no losses the payoff ratio is undefined, and with no
          non-zero returns the win rate is undefined, so the result is ``null``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``kelly_criterion(pl.col("returns")).over("ticker")``.

    See Also:
        - :func:`win_rate`: The win probability ``p``.
        - :func:`payoff_ratio`: The average-win to average-loss ratio ``W``.
        - :func:`risk_of_ruin`: The ruin probability from the same win-rate model.

    References:
        - Kelly, J. L. (1956). "A New Interpretation of Information Rate." *Bell System Technical Journal*, 35(4),
          917-926.
        - https://doi.org/10.1002/j.1538-7305.1956.tb03809.x
        - https://en.wikipedia.org/wiki/Kelly_criterion

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import kelly_criterion
        >>>
        >>> frame = pl.DataFrame({"returns": [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]})
        >>> frame.select(kelly_criterion(pl.col("returns")).round(4)).item()
        0.1758

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [
        ...             0.03,
        ...             -0.01,
        ...             0.02,
        ...             -0.015,
        ...             0.01,
        ...             0.005,
        ...             -0.02,
        ...             0.04,
        ...             -0.02,
        ...             0.03,
        ...             -0.01,
        ...             0.02,
        ...             0.01,
        ...             -0.03,
        ...         ],
        ...     }
        ... )
        >>> reduced = kelly_criterion(pl.col("returns")).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [0.1758, 0.2286]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.03, None, -0.01, 0.02, float("nan"), -0.015, 0.01]})
        >>> frame.select(kelly_criterion(pl.col("returns")).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    probability = win_rate(returns)
    return (probability - (1.0 - probability) / payoff_ratio(returns)).name.keep()


def kurtosis(
    returns: pl.Expr,
) -> pl.Expr:
    r"""
    Excess Kurtosis, the tailedness of a return distribution.

    The population (Fisher) excess kurtosis of the returns -- the standardized fourth moment minus three, so a normal
    distribution scores ``0`` and fat-tailed (leptokurtic) returns score positive:

    .. math::

        K = \frac{m_4}{m_2^2} - 3, \qquad m_k = \frac{1}{n} \sum_{i=1}^{n} (r_i - \bar{r})^k,

    where :math:`m_k` is the population central moment of order :math:`k`.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).

    Returns:
        A single ``Float64`` value: the excess kurtosis (one value in ``select``, one per group under ``.over``).
        ``null`` when there are no returns, and ``NaN`` when the returns have zero variance (fewer than two distinct
        values).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped; an all-null (or empty) series yields ``null``.
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **Zero variance** — a constant series (or single value) has no spread, so the standardized moment is a
          ``0 / 0`` and the result is ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``kurtosis(pl.col("returns")).over("ticker")``.

    See Also:
        - :func:`skewness`: The third-moment companion (asymmetry).
        - :func:`kurtosis_rolling`: The rolling (windowed) form.
        - :func:`value_at_risk_modified`: Uses this excess kurtosis in its Cornish-Fisher tail correction.

    References:
        - Joanes, D. N. & Gill, C. A. (1998). "Comparing Measures of Sample Skewness and Kurtosis." *Journal of the
          Royal Statistical Society: Series D (The Statistician)*, 47(1), 183-189.
        - https://doi.org/10.1111/1467-9884.00122
        - https://en.wikipedia.org/wiki/Kurtosis

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import kurtosis
        >>>
        >>> frame = pl.DataFrame({"returns": [0.01, -0.02, 0.015, -0.03, 0.005, -0.01, 0.02]})
        >>> frame.select(kurtosis(pl.col("returns")).round(4)).item()
        -1.3223

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [
        ...             0.01,
        ...             -0.02,
        ...             0.015,
        ...             -0.03,
        ...             0.005,
        ...             -0.01,
        ...             0.02,
        ...             0.02,
        ...             -0.01,
        ...             0.03,
        ...             -0.02,
        ...             0.01,
        ...             -0.005,
        ...             0.025,
        ...         ],
        ...     }
        ... )
        >>> reduced = kurtosis(pl.col("returns")).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [-1.4673, -1.3223]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.01, None, -0.02, 0.015, float("nan"), -0.03, 0.005]})
        >>> frame.select(kurtosis(pl.col("returns")).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    return (returns.kurtosis()).name.keep()


def kurtosis_rolling(
    returns: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Rolling Excess Kurtosis over a window — the windowed twin of :func:`kurtosis`.

    The population (biased) excess kurtosis of each trailing window -- its fourth standardized central moment, less
    three:

    .. math::

        \mathrm{Kurt}_t = \frac{m_{4,t}}{m_{2,t}^{2}} - 3, \qquad n = \text{window},

    where :math:`m_{k,t}` is the ``k``-th central moment over the window, from Polars' native mean-centered rolling
    kurtosis (numerically stable on a near-constant window, with a bit-constant window forced to ``NaN`` and the
    empty-series case guarded).

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        window: Number of observations in the moving window. Must be ``>= 2``.

    Returns:
        The rolling excess kurtosis for each row, the same length as the input. The first ``window - 1`` rows are
        ``null`` (warm-up): the window must hold ``window`` non-null values before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``.

    Note:
        **Correctness** -- each window matches an independent reference oracle (the reducing :func:`kurtosis` recomputed
        over the window).

        **Edge-case behavior:**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **Zero variance** — a constant window has an undefined kurtosis (``0 / 0``), yielding ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` so the window never spans series boundaries.

    See Also:
        - :func:`kurtosis`: The whole-series reducing form.
        - :func:`skewness_rolling`: The rolling third-moment counterpart.
        - :func:`value_at_risk_modified`: Uses excess kurtosis in its Cornish-Fisher tail correction.

    References:
        - Joanes, D. N. & Gill, C. A. (1998). "Comparing Measures of Sample Skewness and Kurtosis." *Journal of the
          Royal Statistical Society: Series D (The Statistician)*, 47(1), 183-189.
        - https://doi.org/10.1111/1467-9884.00122
        - https://en.wikipedia.org/wiki/Kurtosis

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import kurtosis_rolling
        >>>
        >>> frame = pl.DataFrame({"returns": [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015]})
        >>> frame.select(kurtosis_rolling(pl.col("returns"), 4).round(4))["returns"].to_list()
        [None, None, None, -1.4266, -1.7785, -1.64, -1.099]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up on its own (the ``B`` group never
        borrows ``A``'s tail):

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [
        ...             0.01,
        ...             -0.02,
        ...             0.03,
        ...             -0.01,
        ...             0.02,
        ...             0.0,
        ...             -0.015,
        ...             0.02,
        ...             -0.01,
        ...             0.04,
        ...             -0.03,
        ...             0.01,
        ...             0.025,
        ...             -0.02,
        ...         ],
        ...     }
        ... )
        >>> rolled = kurtosis_rolling(pl.col("returns"), 4).over("ticker").round(4)
        >>> frame.select(rolled.alias("m"))["m"].to_list()
        [None, None, None, -1.4266, -1.7785, -1.64, -1.099, None, None, None, -1.5244, -1.2555, -1.0441, -1.6961]

        A leading ``null`` and a later ``NaN`` show the per-window masking, with the result recovering once both
        leave the window:

        >>> frame = pl.DataFrame({"returns": [None, 0.01, float("nan"), -0.02, 0.03, -0.01, 0.02, 0.0, -0.015]})
        >>> frame.select(kurtosis_rolling(pl.col("returns"), 4).round(4))["returns"].to_list()
        [None, None, None, None, nan, nan, -1.7785, -1.64, -1.099]
    """
    returns = float64_expr(returns)
    validate_window(window, minimum=2)
    return (_rolling_moment(returns, window, kind="kurtosis")).name.keep()


def payoff_ratio(
    returns: pl.Expr,
) -> pl.Expr:
    r"""
    Payoff Ratio, the average winning return over the average losing return.

    The mean of the positive returns divided by the magnitude of the mean of the negative returns -- the average-win to
    average-loss ratio:

    .. math::

        \mathrm{payoff} = \frac{\overline{r_{+}}}{\lvert \overline{r_{-}} \rvert},

    where :math:`\overline{r_{+}}` is the mean of the strictly positive returns and :math:`\overline{r_{-}}` the mean of
    the strictly negative returns.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).

    Returns:
        A single ``Float64`` value: the payoff ratio (one value in ``select``, one per group under ``.over``). ``null``
        when there are no winning returns or no losing returns (one side of the ratio is undefined).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        This is a **bar-level** statistic: each return observation is treated as one win or loss. It is not a per-trade
        statistic -- true per-trade payoff needs trade-level fill data, which is outside this toolkit's scope.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped; an all-null (or empty) series yields ``null``.
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **Zero return** — a return of exactly ``0`` is neither a win nor a loss and is excluded from both means.
        - **One-sided** — with no winning (or no losing) returns the ratio is undefined, so the result is ``null``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``payoff_ratio(pl.col("returns")).over("ticker")``.

    See Also:
        - :func:`win_rate`: The companion frequency (how often returns win).
        - :func:`profit_factor`: The aggregate (total-gain to total-loss) counterpart.
        - :func:`kelly_criterion`: The growth-optimal fraction built from this and the win rate.

    References:
        - Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies* (2nd ed.). Wiley.

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import payoff_ratio
        >>>
        >>> frame = pl.DataFrame({"returns": [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]})
        >>> frame.select(payoff_ratio(pl.col("returns")).round(4)).item()
        1.0833

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [
        ...             0.03,
        ...             -0.01,
        ...             0.02,
        ...             -0.015,
        ...             0.01,
        ...             0.005,
        ...             -0.02,
        ...             0.04,
        ...             -0.02,
        ...             0.03,
        ...             -0.01,
        ...             0.02,
        ...             0.01,
        ...             -0.03,
        ...         ],
        ...     }
        ... )
        >>> reduced = payoff_ratio(pl.col("returns")).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [1.0833, 1.25]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.03, None, -0.01, 0.02, float("nan"), -0.015, 0.01]})
        >>> frame.select(payoff_ratio(pl.col("returns")).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    average_win = returns.filter(returns > 0.0).mean()
    average_loss = returns.filter(returns < 0.0).mean()
    ratio = average_win / -average_loss
    return pl.when(returns.is_nan().any()).then(float("nan")).otherwise(ratio).name.keep()


def profit_factor(
    returns: pl.Expr,
) -> pl.Expr:
    r"""
    Profit Factor, the total gain over the total loss.

    The sum of the positive returns divided by the magnitude of the sum of the negative returns -- the aggregate
    gross-profit to gross-loss ratio (equivalently the :func:`omega_ratio` ratio at a zero threshold):

    .. math::

        \mathrm{PF} = \frac{\sum_{r_i > 0} r_i}{\left\lvert \sum_{r_i < 0} r_i \right\rvert}.

    A value above one means the gains outweigh the losses.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).

    Returns:
        A single ``Float64`` value: the profit factor (one value in ``select``, one per group under ``.over``). ``null``
        when there are no returns.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        This is a **bar-level** statistic: each return observation is treated as one gain or loss. It is not a per-trade
        statistic -- true per-trade profit factor needs trade-level fill data, which is outside this toolkit's scope.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped; an all-null (or empty) series yields ``null``.
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **No losses** — with no negative returns the total loss is zero, so the ratio is ``+inf`` (or ``NaN`` when
          there are also no gains), reported rather than clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``profit_factor(pl.col("returns")).over("ticker")``.

    See Also:
        - :func:`payoff_ratio`: The average-win to average-loss counterpart.
        - :func:`omega_ratio`: The same ratio generalized to an arbitrary threshold.
        - :func:`common_sense_ratio`: Scales this profit factor by the tail ratio.

    References:
        - Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies* (2nd ed.). Wiley.

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import profit_factor
        >>>
        >>> frame = pl.DataFrame({"returns": [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]})
        >>> frame.select(profit_factor(pl.col("returns")).round(4)).item()
        1.4444

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [
        ...             0.03,
        ...             -0.01,
        ...             0.02,
        ...             -0.015,
        ...             0.01,
        ...             0.005,
        ...             -0.02,
        ...             0.04,
        ...             -0.02,
        ...             0.03,
        ...             -0.01,
        ...             0.02,
        ...             0.01,
        ...             -0.03,
        ...         ],
        ...     }
        ... )
        >>> reduced = profit_factor(pl.col("returns")).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [1.4444, 1.6667]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.03, None, -0.01, 0.02, float("nan"), -0.015, 0.01]})
        >>> frame.select(profit_factor(pl.col("returns")).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    # ``.mean()`` is load-bearing, not interchangeable with ``.sum()``: the identical non-null count cancels in the
    # ratio (so the value equals the documented gross-profit / gross-loss sum ratio), but an empty or all-null series
    # yields ``null`` rather than the ``0 / 0`` ``NaN`` a sum would give -- the documented "no returns" behavior.
    mean_gain = returns.clip(lower_bound=0.0).mean()
    mean_loss = (-returns).clip(lower_bound=0.0).mean()
    return (mean_gain / mean_loss).name.keep()


def risk_of_ruin(
    returns: pl.Expr,
) -> pl.Expr:
    r"""
    Risk of Ruin, the probability of losing the whole capital under a symmetric betting model.

    The classic gambler's-ruin probability built from the win rate and the number of bets:

    .. math::

        \mathrm{RoR} = \min\!\left[\left( \frac{1 - p}{p} \right)^{n},\; 1 \right],

    where :math:`p` is the :func:`win_rate` and :math:`n` is the number of (non-null) returns, taken as the capital
    cushion in unit bets. The ratio :math:`(1 - p)/p` is the gambler's-ruin odds ratio :math:`q/p`; with no edge
    (:math:`p \le 0.5`) it is :math:`\ge 1`, so the probability is capped at one -- ruin is certain.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).

    Returns:
        A single ``Float64`` value in ``[0, 1]``: the risk of ruin (one value in ``select``, one per group under
        ``.over``). ``null`` when there are no decisive (non-zero) returns (the win rate is undefined).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        This is the **symmetric** form: it depends only on the win rate and the bet count, assuming equal-sized wins and
        losses and ruin at the loss of all capital. It deliberately ignores win/loss size and capital units. Because the
        bet count ``n`` doubles as the capital cushion, the result is sensitive to the series length: more bars drive it
        toward ``0`` with an edge and toward ``1`` without one. Compare series of the same length. The win rate ``p``
        counts only decisive (non-zero) bars, while ``n`` counts every non-null bar, so padding a series with flat
        ``0`` bars raises ``n`` without moving ``p``.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped; an all-null (or empty) series yields ``null``.
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **No decisive returns** — with no non-zero returns the win rate is undefined, so the result is ``null``.
        - **No edge** — a win rate ``p <= 0.5`` makes the odds ratio ``>= 1``, so the probability saturates at ``1``
          (ruin is certain without an edge); an all-losing series (``p = 0``) likewise gives ``1``.
        - **All wins** — an all-winning series (``p = 1``) gives ``0`` (no ruin risk).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``risk_of_ruin(pl.col("returns")).over("ticker")``.

    See Also:
        - :func:`win_rate`: The win probability the model is built on.
        - :func:`kelly_criterion`: The growth-optimal bet fraction from the same inputs.
        - :func:`payoff_ratio`: The average win/loss size this symmetric model ignores.

    References:
        - Vince, R. (1990). *Portfolio Management Formulas*. Wiley.
        - https://en.wikipedia.org/wiki/Gambler%27s_ruin

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import risk_of_ruin
        >>>
        >>> frame = pl.DataFrame({"returns": [0.02, -0.01, 0.03, -0.02]})
        >>> frame.select(risk_of_ruin(pl.col("returns")).round(4)).item()
        1.0

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently (here ``A`` has no
        edge, so its ruin is certain, while the winning ``B`` is small):

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 4 + ["B"] * 4,
        ...         "returns": [0.02, -0.01, 0.03, -0.02, 0.02, 0.01, 0.03, -0.02],
        ...     }
        ... )
        >>> reduced = risk_of_ruin(pl.col("returns")).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [0.0123, 1.0]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.02, None, -0.01, 0.03, float("nan"), -0.02]})
        >>> frame.select(risk_of_ruin(pl.col("returns")).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    probability = win_rate(returns)
    observations = returns.drop_nulls().len()
    # Gambler's-ruin ratio q/p = (1 - p)/p raised to the capital cushion in unit bets; a system with no edge
    # (p <= 0.5) gives a ratio >= 1, so the probability is capped at 1 (ruin is certain at p <= 0.5).
    return ((1.0 - probability) / probability).pow(observations).clip(upper_bound=1.0).name.keep()


def skewness(
    returns: pl.Expr,
) -> pl.Expr:
    r"""
    Skewness, the asymmetry of a return distribution.

    The population skewness of the returns -- the standardized third moment, negative when the left tail is longer
    (losses more extreme than gains) and positive when the right tail is longer:

    .. math::

        S = \frac{m_3}{m_2^{3/2}}, \qquad m_k = \frac{1}{n} \sum_{i=1}^{n} (r_i - \bar{r})^k,

    where :math:`m_k` is the population central moment of order :math:`k`.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).

    Returns:
        A single ``Float64`` value: the skewness (one value in ``select``, one per group under ``.over``). ``null``
        when there are no returns, and ``NaN`` when the returns have zero variance (fewer than two distinct values).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped; an all-null (or empty) series yields ``null``.
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **Zero variance** — a constant series (or single value) has no spread, so the standardized moment is a
          ``0 / 0`` and the result is ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``skewness(pl.col("returns")).over("ticker")``.

    See Also:
        - :func:`kurtosis`: The fourth-moment companion (tailedness).
        - :func:`skewness_rolling`: The rolling (windowed) form.
        - :func:`value_at_risk_modified`: Uses this skewness in its Cornish-Fisher tail correction.

    References:
        - Joanes, D. N. & Gill, C. A. (1998). "Comparing Measures of Sample Skewness and Kurtosis." *Journal of the
          Royal Statistical Society: Series D (The Statistician)*, 47(1), 183-189.
        - https://doi.org/10.1111/1467-9884.00122
        - https://en.wikipedia.org/wiki/Skewness

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import skewness
        >>>
        >>> frame = pl.DataFrame({"returns": [0.01, -0.02, 0.015, -0.03, 0.005, -0.01, 0.02]})
        >>> frame.select(skewness(pl.col("returns")).round(4)).item()
        -0.384

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [
        ...             0.01,
        ...             -0.02,
        ...             0.015,
        ...             -0.03,
        ...             0.005,
        ...             -0.01,
        ...             0.02,
        ...             0.02,
        ...             -0.01,
        ...             0.03,
        ...             -0.02,
        ...             0.01,
        ...             -0.005,
        ...             0.025,
        ...         ],
        ...     }
        ... )
        >>> reduced = skewness(pl.col("returns")).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [-0.384, -0.1814]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.01, None, -0.02, 0.015, float("nan"), -0.03, 0.005]})
        >>> frame.select(skewness(pl.col("returns")).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    return (returns.skew()).name.keep()


def skewness_rolling(
    returns: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Rolling Skewness over a window — the windowed twin of :func:`skewness`.

    The population (biased) skewness of each trailing window -- its third standardized central moment:

    .. math::

        \mathrm{Skew}_t = \frac{m_{3,t}}{m_{2,t}^{3/2}}, \qquad n = \text{window},

    where :math:`m_{k,t}` is the ``k``-th central moment over the window, from Polars' native mean-centered rolling
    skewness (numerically stable on a near-constant window, with a bit-constant window forced to ``NaN`` and the
    empty-series case guarded).

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        window: Number of observations in the moving window. Must be ``>= 2``.

    Returns:
        The rolling skewness for each row, the same length as the input. The first ``window - 1`` rows are ``null``
        (warm-up): the window must hold ``window`` non-null values before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``.

    Note:
        **Correctness** -- each window matches an independent reference oracle (the reducing :func:`skewness` recomputed
        over the window).

        **Edge-case behavior:**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **Zero variance** — a constant window has an undefined skewness (``0 / 0``), yielding ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` so the window never spans series boundaries.

    See Also:
        - :func:`skewness`: The whole-series reducing form.
        - :func:`kurtosis_rolling`: The rolling fourth-moment counterpart.
        - :func:`value_at_risk_modified`: Uses skewness in its Cornish-Fisher tail correction.

    References:
        - Joanes, D. N. & Gill, C. A. (1998). "Comparing Measures of Sample Skewness and Kurtosis." *Journal of the
          Royal Statistical Society: Series D (The Statistician)*, 47(1), 183-189.
        - https://doi.org/10.1111/1467-9884.00122
        - https://en.wikipedia.org/wiki/Skewness

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import skewness_rolling
        >>>
        >>> frame = pl.DataFrame({"returns": [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015]})
        >>> frame.select(skewness_rolling(pl.col("returns"), 4).round(4))["returns"].to_list()
        [None, None, None, 0.278, -0.0, -0.0, 0.6568]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up on its own (the ``B`` group never
        borrows ``A``'s tail):

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [
        ...             0.01,
        ...             -0.02,
        ...             0.03,
        ...             -0.01,
        ...             0.02,
        ...             0.0,
        ...             -0.015,
        ...             0.02,
        ...             -0.01,
        ...             0.04,
        ...             -0.03,
        ...             0.01,
        ...             0.025,
        ...             -0.02,
        ...         ],
        ...     }
        ... )
        >>> rolled = skewness_rolling(pl.col("returns"), 4).over("ticker").round(4)
        >>> frame.select(rolled.alias("m"))["m"].to_list()
        [None, None, None, 0.278, -0.0, -0.0, 0.6568, None, None, None, -0.0, 0.2439, -0.6183, 0.0912]

        A leading ``null`` and a later ``NaN`` show the per-window masking, with the result recovering once both
        leave the window:

        >>> frame = pl.DataFrame({"returns": [None, 0.01, float("nan"), -0.02, 0.03, -0.01, 0.02, 0.0, -0.015]})
        >>> frame.select(skewness_rolling(pl.col("returns"), 4).round(4))["returns"].to_list()
        [None, None, None, None, nan, nan, -0.0, -0.0, 0.6568]
    """
    returns = float64_expr(returns)
    validate_window(window, minimum=2)
    return (_rolling_moment(returns, window, kind="skew")).name.keep()


def tail_ratio(
    returns: pl.Expr,
) -> pl.Expr:
    r"""
    Tail Ratio, the size of the right tail relative to the left tail of a return distribution.

    The magnitude of the 95th-percentile return divided by the 5th-percentile return -- above one the best outcomes
    outweigh the worst (a favorable, right-leaning tail profile), below one the reverse:

    .. math::

        \mathrm{TR} = \left| \frac{Q_{0.95}(r)}{Q_{0.05}(r)} \right|,

    where :math:`Q_{p}` is the type-7 (linear-interpolation) empirical quantile of the returns. Being a ratio of two
    quantiles it is scale-invariant -- rescaling the returns leaves it unchanged.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).

    Returns:
        A single ``Float64`` value: the tail ratio (one value in ``select``, one per group under ``.over``). ``null``
        when there are no returns.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped; an all-null (or empty) series yields ``null``.
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **Zero left tail** — when the 5th-percentile return is exactly ``0`` the ratio is ``+inf`` (or ``NaN`` when
          the 95th percentile is also ``0``), reported rather than clipped, following IEEE division.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``tail_ratio(pl.col("returns")).over("ticker")``.

    See Also:
        - :func:`tail_ratio_rolling`: The rolling (windowed) form.
        - :func:`common_sense_ratio`: Scales the profit factor by this tail ratio.
        - :func:`skewness`: The moment-based companion measure of distributional asymmetry.

    References:
        - https://en.wikipedia.org/wiki/Tail_risk

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import tail_ratio
        >>>
        >>> frame = pl.DataFrame({"returns": [0.02, -0.04, 0.01, -0.06, 0.03]})
        >>> frame.select(tail_ratio(pl.col("returns")).round(4)).item()
        0.5

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "returns": [0.02, -0.04, 0.01, -0.06, 0.03, 0.05, -0.02, 0.04, -0.03, 0.02],
        ...     }
        ... )
        >>> reduced = tail_ratio(pl.col("returns")).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [0.5, 1.7143]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.02, None, -0.04, 0.01, float("nan"), -0.06, 0.03]})
        >>> frame.select(tail_ratio(pl.col("returns")).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    right_tail = returns.quantile(0.95, interpolation="linear")
    left_tail = returns.quantile(0.05, interpolation="linear")
    ratio = (right_tail / left_tail).abs()
    return pl.when(returns.is_nan().any()).then(float("nan")).otherwise(ratio).name.keep()


def tail_ratio_rolling(
    returns: pl.Expr,
    window: int,
) -> pl.Expr:
    r"""
    Rolling Tail Ratio over a window — the windowed twin of :func:`tail_ratio`.

    The magnitude of the 95th-percentile return divided by the 5th-percentile return over each trailing window:

    .. math::

        \mathrm{TR}_t = \left| \frac{Q_{0.95}}{Q_{0.05}} \right|, \qquad n = \text{window},

    where :math:`Q_{p}` is the type-7 (linear-interpolation) empirical quantile over the window.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        window: Number of observations in the moving window. Must be ``>= 1``.

    Returns:
        The rolling tail ratio for each row, the same length as the input. The first ``window - 1`` rows are ``null``
        (warm-up): the window must hold ``window`` non-null values before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``.

    Note:
        **Correctness** -- each window matches an independent reference oracle (the reducing :func:`tail_ratio`
        recomputed over the window).

        **Edge-case behavior:**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **Zero left tail** — when the 5th-percentile return is exactly ``0`` the ratio is ``+inf`` (or ``NaN`` when
          the right tail is also ``0``), reported rather than clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` so the window never spans series boundaries.

    See Also:
        - :func:`tail_ratio`: The whole-series reducing form.
        - :func:`value_at_risk_rolling`: Another rolling tail-risk measure.
        - :func:`skewness_rolling`: The rolling moment-based companion measure of distributional asymmetry.

    References:
        - https://en.wikipedia.org/wiki/Tail_risk

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import tail_ratio_rolling
        >>>
        >>> frame = pl.DataFrame({"returns": [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015]})
        >>> frame.select(tail_ratio_rolling(pl.col("returns"), 5).round(4)).to_series().to_list()
        [None, None, None, None, 1.5556, 1.5556, 2.0]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up on its own (the ``B`` group never
        borrows ``A``'s tail):

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [
        ...             0.01,
        ...             -0.02,
        ...             0.03,
        ...             -0.01,
        ...             0.02,
        ...             0.0,
        ...             -0.015,
        ...             0.02,
        ...             -0.01,
        ...             0.04,
        ...             -0.03,
        ...             0.01,
        ...             0.025,
        ...             -0.02,
        ...         ],
        ...     }
        ... )
        >>> rolled = tail_ratio_rolling(pl.col("returns"), 5).over("ticker").round(4)
        >>> frame.select(rolled.alias("m"))["m"].to_list()
        [None, None, None, None, 1.5556, 1.5556, 2.0, None, None, None, None, 1.3846, 1.4231, 1.3214]

        A leading ``null`` and a later ``NaN`` show the per-window masking, with the result recovering once both
        leave the window:

        >>> frame = pl.DataFrame({"returns": [None, 0.01, float("nan"), -0.02, 0.03, -0.01, 0.02, 0.0, -0.015, 0.005]})
        >>> frame.select(tail_ratio_rolling(pl.col("returns"), 5).round(4)).to_series().to_list()
        [None, None, None, None, None, nan, nan, 1.5556, 2.0, 1.2143]
    """
    returns = float64_expr(returns)
    validate_window(window)
    right_tail = returns.rolling_quantile(0.95, interpolation="linear", window_size=window, min_samples=window)
    left_tail = returns.rolling_quantile(0.05, interpolation="linear", window_size=window, min_samples=window)
    ratio = (right_tail / left_tail).abs()
    return pl.when(rolling_has_nan(returns, window)).then(float("nan")).otherwise(ratio).name.keep()


def value_at_risk(
    returns: pl.Expr,
    *,
    confidence: float = 0.95,
) -> pl.Expr:
    r"""
    Historical Value-at-Risk, the loss threshold a return falls below only ``1 - confidence`` of the time.

    The ``1 - confidence`` empirical quantile of the returns -- the worst loss not exceeded at the given confidence
    level, estimated directly from the realized return distribution (historical simulation) rather than a parametric
    model:

    .. math::

        \mathrm{VaR}_{c} = Q_{1 - c}(r),

    where :math:`Q_{p}` is the type-7 (linear-interpolation) empirical quantile and :math:`c` is ``confidence``. The
    value is on the same scale as the returns and is negative for a loss (a result of ``-0.05`` is a 5% loss).

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        confidence: The tail confidence level (canonically ``0.95``); the quantile taken is ``1 - confidence``. Must be
            in the open interval ``(0, 1)``.

    Returns:
        A single ``Float64`` value: the historical value-at-risk (one value in ``select``, one per group under
        ``.over``). ``null`` when there are no returns.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``confidence`` is not in the open interval ``(0, 1)``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped; an all-null (or empty) series yields ``null``.
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **Sign convention** — returned as the signed return quantile (negative for a loss), not a positive loss
          magnitude; negate it if a positive figure is wanted.
        - **Historical, not parametric** — the quantile is taken over the empirical return distribution, with no
          normality or other distributional assumption.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``value_at_risk(pl.col("returns")).over("ticker")``.

    See Also:
        - :func:`conditional_value_at_risk`: The mean loss beyond this threshold (expected shortfall).
        - :func:`value_at_risk_parametric`: The Gaussian (parametric) estimate of the same quantile.
        - :func:`value_at_risk_modified`: The skewness/kurtosis-corrected (Cornish-Fisher) estimate.
        - :func:`value_at_risk_rolling`: The rolling (windowed) form.

    References:
        - J.P. Morgan / Reuters (1996). *RiskMetrics — Technical Document* (4th ed.).
        - https://en.wikipedia.org/wiki/Value_at_risk

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import value_at_risk
        >>>
        >>> frame = pl.DataFrame({"returns": [0.02, -0.04, 0.01, -0.06, 0.03]})
        >>> frame.select(value_at_risk(pl.col("returns"), confidence=0.95).round(4)).item()
        -0.056

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "returns": [0.02, -0.04, 0.01, -0.06, 0.03, 0.01, -0.02, 0.04, -0.03, 0.02],
        ...     }
        ... )
        >>> reduced = value_at_risk(pl.col("returns"), confidence=0.95).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [-0.056, -0.028]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.02, None, -0.04, 0.01, float("nan"), -0.06, 0.03]})
        >>> frame.select(value_at_risk(pl.col("returns"), confidence=0.95).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    validate_confidence(confidence)
    quantile = returns.quantile(1.0 - confidence, interpolation="linear")
    return pl.when(returns.is_nan().any()).then(float("nan")).otherwise(quantile).name.keep()


def value_at_risk_modified(
    returns: pl.Expr,
    *,
    confidence: float = 0.95,
) -> pl.Expr:
    r"""
    Modified Value-at-Risk (a.k.a. Cornish-Fisher VaR), the Gaussian VaR corrected for skewness and kurtosis.

    The parametric value-at-risk with its normal quantile replaced by the Cornish-Fisher expansion, which adjusts for
    the return distribution's skewness and excess kurtosis -- a fatter or more skewed tail shifts the estimate:

    .. math::

        z_{cf} = z + \frac{z^2 - 1}{6}\gamma_3 + \frac{z^3 - 3z}{24}\gamma_4 - \frac{2z^3 - 5z}{36}\gamma_3^2,
        \qquad \mathrm{mVaR} = \bar{r} + z_{cf}\,\sigma,

    where :math:`z = \Phi^{-1}(1 - \texttt{confidence})`, :math:`\gamma_3` is the (population) skewness,
    :math:`\gamma_4` the (population) excess kurtosis, and :math:`\sigma` the sample standard deviation (``ddof = 1``).
    The value is on the same scale as the returns and is negative for a loss.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        confidence: The tail confidence level (canonically ``0.95``); the quantile taken is ``1 - confidence``. Must
            be in the open interval ``(0, 1)``.

    Returns:
        A single ``Float64`` value: the modified value-at-risk (one value in ``select``, one per group under ``.over``).
        ``null`` when fewer than two returns are present (the sample standard deviation is undefined).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``confidence`` is not in the open interval ``(0, 1)``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Domain** — the one-term Cornish-Fisher expansion is valid only where its quantile map is order-preserving:
        locally monotonic at the requested quantile, and with the corrected quantile on the same side of the median
        as the Gaussian one. Tail moments outside that region make the corrected number statistically meaningless
        (it can even flip sign, reporting a crash-bearing series as a gain), so the result is a loud ``NaN``.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped (excluded from every moment).
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **Fewer than two returns** — the sample standard deviation is undefined, so the result is ``null``.
        - **Zero volatility** — a constant series has undefined skewness and kurtosis, yielding ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``value_at_risk_modified(pl.col("returns")).over("ticker")``.

    See Also:
        - :func:`value_at_risk_parametric`: The Gaussian form this corrects.
        - :func:`value_at_risk`: The historical (empirical) form.
        - :func:`conditional_value_at_risk`: The expected shortfall beyond the VaR threshold.

    References:
        - Favre, L. & Galeano, J.-A. (2002). "Mean-Modified Value-at-Risk Optimization with Hedge Funds." *Journal of
          Alternative Investments*, 5(2), 21-25.
        - https://doi.org/10.3905/jai.2002.319052
        - https://en.wikipedia.org/wiki/Cornish%E2%80%93Fisher_expansion

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import value_at_risk_modified
        >>>
        >>> frame = pl.DataFrame({"returns": [0.02, -0.04, 0.01, -0.06, 0.03, -0.05, 0.04, -0.02, 0.01, -0.03]})
        >>> frame.select(value_at_risk_modified(pl.col("returns")).round(4)).item()
        -0.069

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 10 + ["B"] * 10,
        ...         "returns": [
        ...             0.02,
        ...             -0.04,
        ...             0.01,
        ...             -0.06,
        ...             0.03,
        ...             -0.05,
        ...             0.04,
        ...             -0.02,
        ...             0.01,
        ...             -0.03,
        ...             0.03,
        ...             -0.03,
        ...             0.02,
        ...             -0.05,
        ...             0.04,
        ...             -0.04,
        ...             0.03,
        ...             -0.01,
        ...             0.02,
        ...             -0.02,
        ...         ],
        ...     }
        ... )
        >>> reduced = value_at_risk_modified(pl.col("returns")).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [-0.069, -0.0579]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.02, None, -0.04, 0.01, float("nan"), -0.06, 0.03, -0.05, 0.04, -0.02]})
        >>> frame.select(value_at_risk_modified(pl.col("returns")).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    validate_confidence(confidence)
    z = NormalDist().inv_cdf(1.0 - confidence)
    skew = returns.skew()
    excess_kurtosis = returns.kurtosis()
    z_cornish_fisher = (
        z
        + (z**2 - 1.0) / 6.0 * skew
        + (z**3 - 3.0 * z) / 24.0 * excess_kurtosis
        - (2.0 * z**3 - 5.0 * z) / 36.0 * skew**2
    )
    # Validity domain, two order-preservation requirements of the one-term expansion: (a) the quantile map stays
    # locally monotonic at the requested quantile — d(z_cf)/dz = 1 + z*S/3 + (z^2 - 1)*K/8 - (6*z^2 - 5)*S^2/36 > 0 —
    # and (b) the corrected quantile stays on its own side of the median (z_cf and z agree in sign): a tail moment
    # extreme enough to push the alpha-quantile across the median has left the expansion's validity region, and the
    # corrected number can even flip sign, reporting a crash-bearing series as a GAIN. Outside the domain the result
    # is a loud NaN, never a plausible wrong number. A NaN-poisoned skew/kurtosis fails both comparisons and falls
    # through to the (already NaN) estimate, preserving the poison contract.
    slope = 1.0 + z * skew / 3.0 + (z**2 - 1.0) / 8.0 * excess_kurtosis - (6.0 * z**2 - 5.0) / 36.0 * skew**2
    estimate = returns.mean() + z_cornish_fisher * returns.std(ddof=1)
    return (
        pl.when(slope <= 0.0)
        .then(float("nan"))
        .when(z_cornish_fisher * z < 0.0)
        .then(float("nan"))
        .otherwise(estimate)
        .name.keep()
    )


def value_at_risk_parametric(
    returns: pl.Expr,
    *,
    confidence: float = 0.95,
) -> pl.Expr:
    r"""
    Parametric Value-at-Risk (a.k.a. variance-covariance / Gaussian VaR), the normal-distribution loss quantile.

    The value-at-risk under a normal-distribution assumption: the mean plus the standard normal quantile of the tail
    scaled by the standard deviation:

    .. math::

        \mathrm{VaR}_{c} = \bar{r} + \Phi^{-1}(1 - c)\,\sigma,

    where :math:`\Phi^{-1}` is the standard-normal quantile function, :math:`c` is ``confidence``, and :math:`\sigma`
    the sample standard deviation (``ddof = 1``). The value is on the same scale as the returns and is negative for a
    loss.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        confidence: The tail confidence level (canonically ``0.95``); the quantile taken is ``1 - confidence``. Must
            be in the open interval ``(0, 1)``.

    Returns:
        A single ``Float64`` value: the parametric value-at-risk (one value in ``select``, one per group under
        ``.over``). ``null`` when fewer than two returns are present (the sample standard deviation is undefined).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``confidence`` is not in the open interval ``(0, 1)``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped (excluded from the mean and the standard deviation).
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **Fewer than two returns** — the sample standard deviation is undefined, so the result is ``null``.
        - **Gaussian assumption** — the estimate assumes normally distributed returns; for fat tails see
          :func:`value_at_risk_modified` (Cornish-Fisher) or :func:`value_at_risk` (historical).
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``value_at_risk_parametric(pl.col("returns")).over("ticker")``.

    See Also:
        - :func:`value_at_risk`: The historical (empirical) form.
        - :func:`value_at_risk_modified`: The skewness/kurtosis-corrected form (within its documented validity domain).
        - :func:`conditional_value_at_risk`: The expected shortfall beyond the VaR threshold.

    References:
        - Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk* (3rd ed.). McGraw-Hill.
        - https://en.wikipedia.org/wiki/Value_at_risk

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import value_at_risk_parametric
        >>>
        >>> frame = pl.DataFrame({"returns": [0.02, -0.04, 0.01, -0.06, 0.03]})
        >>> frame.select(value_at_risk_parametric(pl.col("returns")).round(4)).item()
        -0.0732

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "returns": [0.02, -0.04, 0.01, -0.06, 0.03, 0.01, -0.02, 0.04, -0.03, 0.02],
        ...     }
        ... )
        >>> reduced = value_at_risk_parametric(pl.col("returns")).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [-0.0732, -0.0434]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.02, None, -0.04, 0.01, float("nan"), -0.06, 0.03]})
        >>> frame.select(value_at_risk_parametric(pl.col("returns")).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    validate_confidence(confidence)
    z = NormalDist().inv_cdf(1.0 - confidence)
    return (returns.mean() + z * returns.std(ddof=1)).name.keep()


def value_at_risk_rolling(
    returns: pl.Expr,
    window: int,
    *,
    confidence: float = 0.95,
) -> pl.Expr:
    r"""
    Rolling historical Value-at-Risk over a window — the windowed twin of :func:`value_at_risk`.

    The type-7 (linear-interpolation) empirical quantile of each trailing window at the lower tail:

    .. math::

        \mathrm{VaR}_t = Q_{1 - c}\bigl(r_{t-n+1}, \dots, r_t\bigr), \qquad n = \text{window},

    where :math:`c` is ``confidence``. Returned as the signed return quantile (negative for a loss).

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        window: Number of observations in the moving window. Must be ``>= 1``.
        confidence: The tail confidence level (canonically ``0.95``); the quantile taken is ``1 - confidence``. Must be
            in the open interval ``(0, 1)``.

    Returns:
        The rolling value-at-risk for each row, the same length as the input. The first ``window - 1`` rows are ``null``
        (warm-up): the window must hold ``window`` non-null values before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``, or if ``confidence`` is not in the open interval ``(0, 1)``.

    Note:
        **Correctness** -- each window matches an independent reference oracle (the reducing :func:`value_at_risk`
        recomputed over the window).

        **Edge-case behavior:**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **Sign convention** — returned as the signed return quantile (negative for a loss), not a positive loss.
        - **Partitioning** — wrap the call in ``.over(...)`` so the window never spans series boundaries.

    See Also:
        - :func:`value_at_risk`: The whole-series reducing form.
        - :func:`tail_ratio_rolling`: Another rolling tail-risk measure.
        - :func:`downside_deviation_rolling`: Another rolling downside-risk measure.

    References:
        - https://en.wikipedia.org/wiki/Value_at_risk

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import value_at_risk_rolling
        >>>
        >>> frame = pl.DataFrame({"returns": [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015]})
        >>> frame.select(value_at_risk_rolling(pl.col("returns"), 4).round(4)).to_series().to_list()
        [None, None, None, -0.0185, -0.0185, -0.0085, -0.0142]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up on its own (the ``B`` group never
        borrows ``A``'s tail):

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [
        ...             0.01,
        ...             -0.02,
        ...             0.03,
        ...             -0.01,
        ...             0.02,
        ...             0.0,
        ...             -0.015,
        ...             0.02,
        ...             -0.01,
        ...             0.04,
        ...             -0.03,
        ...             0.01,
        ...             0.025,
        ...             -0.02,
        ...         ],
        ...     }
        ... )
        >>> rolled = value_at_risk_rolling(pl.col("returns"), 4).over("ticker").round(4)
        >>> frame.select(rolled.alias("m"))["m"].to_list()
        [None, None, None, -0.0185, -0.0185, -0.0085, -0.0142, None, None, None, -0.027, -0.027, -0.024, -0.0285]

        A leading ``null`` and a later ``NaN`` show the per-window masking, with the result recovering once both
        leave the window:

        >>> frame = pl.DataFrame({"returns": [None, 0.01, float("nan"), -0.02, 0.03, -0.01, 0.02, 0.0, -0.015]})
        >>> frame.select(value_at_risk_rolling(pl.col("returns"), 4).round(4)).to_series().to_list()
        [None, None, None, None, nan, nan, -0.0185, -0.0085, -0.0142]
    """
    returns = float64_expr(returns)
    validate_window(window)
    validate_confidence(confidence)
    quantile = returns.rolling_quantile(
        1.0 - confidence, interpolation="linear", window_size=window, min_samples=window
    )
    return pl.when(rolling_has_nan(returns, window)).then(float("nan")).otherwise(quantile).name.keep()


def volatility(
    returns: pl.Expr,
    *,
    periods_per_year: int,
) -> pl.Expr:
    r"""
    Annualized Volatility, the annualized sample standard deviation of returns.

    The sample standard deviation of the per-bar returns, scaled to a yearly figure by the square root of the number of
    periods per year — the standard "square-root-of-time" rule:

    .. math::

        \sigma_{\mathrm{ann}} = \sigma \, \sqrt{P}, \qquad
        \sigma = \sqrt{\frac{1}{n - 1} \sum_{i=1}^{n} (r_i - \bar{r})^2},

    where :math:`P` is ``periods_per_year`` and :math:`\sigma` is the sample standard deviation (``ddof = 1``) of the
    returns. The square-root-of-time scaling assumes the per-bar returns are serially uncorrelated.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.

    Returns:
        A single ``Float64`` value: the annualized volatility of the series (one value in ``select``, one per group
        under ``.over``). ``null`` when fewer than two returns are present (the sample standard deviation is undefined).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped (excluded from the standard deviation), so a leading warm-up ``null``
          (as produced by :func:`~pomata.pnl.returns_simple`) does not affect the result.
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **Fewer than two returns** — the sample standard deviation is undefined, so the result is ``null``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel so the volatility is computed per
          series, e.g. ``volatility(pl.col("returns"), periods_per_year=252).over("ticker")``.

    See Also:
        - :func:`volatility_rolling`: The rolling (windowed) form.
        - :func:`downside_deviation`: The downside-only (one-sided) counterpart.
        - :func:`~pomata.pnl.returns_net`: The usual source of the net-return series this measures.

    References:
        - https://en.wikipedia.org/wiki/Volatility_%28finance%29

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import volatility
        >>>
        >>> frame = pl.DataFrame({"returns": [0.01, -0.02, 0.015, 0.005, -0.01]})
        >>> frame.select(volatility(pl.col("returns"), periods_per_year=252).round(4)).item()
        0.2314

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker's volatility is computed independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 5 + ["B"] * 5,
        ...         "returns": [0.01, -0.02, 0.015, 0.005, -0.01, 0.02, 0.01, -0.03, 0.0, 0.01],
        ...     }
        ... )
        >>> annual = volatility(pl.col("returns"), periods_per_year=252).over("ticker").round(4)
        >>> frame.select(annual.alias("v"))["v"].unique().sort().to_list()
        [0.2314, 0.3054]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.01, None, -0.02, 0.015, float("nan"), 0.005, -0.01]})
        >>> frame.select(volatility(pl.col("returns"), periods_per_year=252).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    validate_periods_per_year(periods_per_year)
    # The constant series is pinned to exactly 0.0 (not the chunked-mean ULP residue), so a ratio dividing by this
    # volatility degenerates to the documented signed infinity rather than a spuriously huge finite.
    return (_dispersion(returns) * math.sqrt(periods_per_year)).name.keep()


def volatility_rolling(
    returns: pl.Expr,
    window: int,
    *,
    periods_per_year: int,
) -> pl.Expr:
    r"""
    Rolling Volatility over a window — the windowed twin of :func:`volatility`.

    The sample standard deviation (``ddof = 1``) of each trailing window, annualized by the square-root-of-time rule:

    .. math::

        \sigma_t = \sqrt{\frac{1}{n - 1} \sum_{i=t-n+1}^{t} (r_i - \bar{r}_t)^2}\,\sqrt{P}, \qquad n = \text{window},

    where :math:`P` is ``periods_per_year``.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        window: Number of observations in the moving window. Must be ``>= 2``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.

    Returns:
        The rolling annualized volatility for each row, the same length as the input. The first ``window - 1`` rows are
        ``null`` (warm-up): the window must hold ``window`` non-null values before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``, or if ``periods_per_year < 1``.

    Note:
        **Correctness** -- each window matches an independent reference oracle (the reducing :func:`volatility`
        recomputed over the window).

        **Edge-case behavior:**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **Constant window** — a window of equal returns has zero dispersion, so the result is ``0``.
        - **Partitioning** — wrap the call in ``.over(...)`` so the window never spans series boundaries.

    See Also:
        - :func:`volatility`: The whole-series reducing form.
        - :func:`sharpe_ratio_rolling`: The risk-adjusted ratio whose denominator is this.
        - :func:`downside_deviation_rolling`: The downside-only rolling counterpart.

    References:
        - https://en.wikipedia.org/wiki/Volatility_%28finance%29

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import volatility_rolling
        >>>
        >>> frame = pl.DataFrame({"returns": [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015]})
        >>> frame.select(volatility_rolling(pl.col("returns"), 3, periods_per_year=252).round(4))["returns"].to_list()
        [None, None, 0.3995, 0.42, 0.3305, 0.2425, 0.2787]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up on its own (the ``B`` group never
        borrows ``A``'s tail):

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [
        ...             0.01,
        ...             -0.02,
        ...             0.03,
        ...             -0.01,
        ...             0.02,
        ...             0.0,
        ...             -0.015,
        ...             0.02,
        ...             -0.01,
        ...             0.04,
        ...             -0.03,
        ...             0.01,
        ...             0.025,
        ...             -0.02,
        ...         ],
        ...     }
        ... )
        >>> rolled = volatility_rolling(pl.col("returns"), 3, periods_per_year=252).over("ticker").round(4)
        >>> frame.select(rolled.alias("m"))["m"].to_list()
        [None, None, 0.3995, 0.42, 0.3305, 0.2425, 0.2787, None, None, 0.3995, 0.5724, 0.5575, 0.4513, 0.3637]

        A leading ``null`` and a later ``NaN`` show the per-window masking, with the result recovering once both
        leave the window:

        >>> frame = pl.DataFrame({"returns": [None, 0.01, -0.02, float("nan"), 0.03, -0.01, 0.02]})
        >>> frame.select(volatility_rolling(pl.col("returns"), 3, periods_per_year=252).round(4))["returns"].to_list()
        [None, None, None, nan, nan, nan, 0.3305]
    """
    returns = float64_expr(returns)
    validate_window(window, minimum=2)
    validate_periods_per_year(periods_per_year)
    dispersion = returns.rolling_std(window, ddof=1, min_samples=window) * math.sqrt(periods_per_year)
    # A constant window has zero dispersion -> 0.0; the incremental rolling standard deviation can leave a residue after
    # a much larger value exits the window, so guard it explicitly. The NaN check comes first: a NaN window stays NaN.
    return (
        pl.when(rolling_has_nan(returns, window))
        .then(float("nan"))
        .when(rolling_is_constant(returns, window))
        .then(0.0)
        .otherwise(dispersion)
        .name.keep()
    )


def win_rate(
    returns: pl.Expr,
) -> pl.Expr:
    r"""
    Win Rate, the fraction of decisive returns that are positive.

    The count of winning (strictly positive) returns over the count of decisive (non-zero) returns:

    .. math::

        \mathrm{win\ rate} = \frac{\#\{r_i > 0\}}{\#\{r_i \neq 0\}}.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).

    Returns:
        A single ``Float64`` value in ``[0, 1]``: the win rate (one value in ``select``, one per group under ``.over``).
        ``null`` when there are no decisive (non-zero) returns.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        This is a **bar-level** statistic: each return observation is treated as one win or loss. It is not a per-trade
        statistic -- true per-trade win rate needs trade-level fill data, which is outside this toolkit's scope.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped; an all-null (or empty) series yields ``null``.
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **Zero return** — a return of exactly ``0`` is neither a win nor a loss and is excluded from the denominator;
          a series with no non-zero returns yields ``null``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``win_rate(pl.col("returns")).over("ticker")``.

    See Also:
        - :func:`payoff_ratio`: The average size of a win versus a loss.
        - :func:`profit_factor`: The aggregate gain-to-loss ratio.
        - :func:`kelly_criterion`: The growth-optimal bet fraction built on this rate.

    References:
        - Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies* (2nd ed.). Wiley.

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import win_rate
        >>>
        >>> frame = pl.DataFrame({"returns": [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]})
        >>> frame.select(win_rate(pl.col("returns")).round(4)).item()
        0.5714

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [
        ...             0.03,
        ...             -0.01,
        ...             0.02,
        ...             -0.015,
        ...             0.01,
        ...             0.005,
        ...             -0.02,
        ...             0.04,
        ...             -0.02,
        ...             0.03,
        ...             0.01,
        ...             0.02,
        ...             0.01,
        ...             -0.03,
        ...         ],
        ...     }
        ... )
        >>> reduced = win_rate(pl.col("returns")).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [0.5714, 0.7143]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.03, None, -0.01, 0.02, float("nan"), -0.015, 0.01]})
        >>> frame.select(win_rate(pl.col("returns")).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    wins = (returns > 0.0).sum()
    decisive = (returns != 0.0).sum()
    ratio = pl.when(decisive == 0).then(None).otherwise(wins / decisive)
    return pl.when(returns.is_nan().any()).then(float("nan")).otherwise(ratio).name.keep()
