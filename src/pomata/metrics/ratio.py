"""
Risk-adjusted performance ratios — return per unit of risk, composed from the dispersion, drawdown, and growth metrics.

This is the composing layer of the family: every ratio is built on top of the base theme modules (``risk``,
``drawdown``, ``performance``), imported from the specific module rather than the package root so the theme dependency
graph stays acyclic.
"""

import math
from statistics import NormalDist

import polars as pl

from pomata._expr import float64_expr, per_period_rate, validate_finite, validate_periods_per_year, validate_window
from pomata.metrics.drawdown import drawdown, max_drawdown, pain_index, ulcer_index
from pomata.metrics.performance import cagr, total_return
from pomata.metrics.risk import (
    downside_deviation,
    downside_deviation_rolling,
    profit_ratio,
    tail_ratio,
    volatility,
    volatility_rolling,
)

__all__ = (
    "adjusted_sharpe_ratio",
    "burke_ratio",
    "calmar_ratio",
    "common_sense_ratio",
    "gain_to_pain_ratio",
    "omega_ratio",
    "omega_ratio_rolling",
    "pain_ratio",
    "probabilistic_sharpe_ratio",
    "recovery_ratio",
    "sharpe_ratio",
    "sharpe_ratio_rolling",
    "sortino_ratio",
    "sortino_ratio_rolling",
    "sterling_ratio",
    "ulcer_performance_ratio",
)


def _normal_cdf(
    values: pl.Series,
) -> pl.Series:
    """
    The standard-normal CDF applied to a one-element aggregation, with ``null`` / ``NaN`` passed through unchanged.

    Polars has no native error function, so the cumulative-distribution step of the probabilistic Sharpe ratio is the
    one place the family reaches for a Python callback -- applied once per group to a scalar, never per row.
    """
    return pl.Series(
        values.name,
        [None if value is None else (math.nan if math.isnan(value) else NormalDist().cdf(value)) for value in values],
        dtype=pl.Float64,
    )


def adjusted_sharpe_ratio(
    returns: pl.Expr,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> pl.Expr:
    r"""
    Adjusted Sharpe Ratio, the Sharpe ratio penalized for negative skewness and excess kurtosis.

    The Pezier & White correction to the :func:`sharpe_ratio` ratio that discounts a non-normal return profile -- it
    rewards positive skewness and penalizes fat tails:

    .. math::

        \mathrm{ASR} = \sqrt{P}\;\mathrm{SR}_p\left(1 + \frac{\gamma_3}{6}\mathrm{SR}_p
        - \frac{\gamma_4}{24}\mathrm{SR}_p^{2}\right),

    where :math:`\mathrm{SR}_p` is the **per-period** :func:`sharpe_ratio` (the annualized ratio divided by
    :math:`\sqrt{P}`), :math:`P` is ``periods_per_year``, :math:`\gamma_3` the (population) skewness, and
    :math:`\gamma_4` the (population) excess kurtosis. The correction is applied at the data frequency -- so it captures
    the distribution shape and not the annualization factor -- and the corrected ratio is then annualized by the
    square-root-of-time rule.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate, converted to a per-period rate geometrically (default ``0.0``).
            Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + risk_free_rate >= 0``).

    Returns:
        A single ``Float64`` value: the adjusted Sharpe ratio (one value in ``select``, one per group under ``.over``).
        ``null`` when fewer than two returns are present (the Sharpe ratio is undefined).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite or is ``< -1``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped (excluded from every moment).
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **Fewer than two returns** — the sample Sharpe ratio is undefined, so the result is ``null``.
        - **Zero volatility** — a constant series has an undefined Sharpe ratio and undefined moments, yielding ``NaN``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``adjusted_sharpe_ratio(pl.col("returns"), periods_per_year=252).over("ticker")``.

    See Also:
        - :func:`sharpe_ratio`: The base ratio this adjusts.
        - :func:`probabilistic_sharpe_ratio`: The confidence-level alternative correction for non-normality.
        - :func:`sortino_ratio`: The downside-deviation variant that captures the same return asymmetry differently.

    References:
        - Pezier, J. & White, A. (2008). "The Relative Merits of Alternative Investments in Passive Portfolios."
          *Journal of Alternative Investments*, 10(4), 37-49.
        - https://doi.org/10.3905/jai.2008.705531
        - https://en.wikipedia.org/wiki/Sharpe_ratio

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import adjusted_sharpe_ratio
        >>>
        >>> frame = pl.DataFrame({"returns": [0.03, -0.02, 0.04, -0.03, 0.02, -0.01, 0.025, -0.015]})
        >>> frame.select(adjusted_sharpe_ratio(pl.col("returns"), periods_per_year=252).round(4)).item()
        2.992

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]
        ...         + [0.02, -0.005, 0.015, -0.01, 0.025, 0.0, -0.012],
        ...     }
        ... )
        >>> reduced = adjusted_sharpe_ratio(pl.col("returns"), periods_per_year=252).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [2.4414, 5.0532]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.03, None, 0.02, -0.015, float("nan"), 0.005, -0.02]})
        >>> frame.select(adjusted_sharpe_ratio(pl.col("returns"), periods_per_year=252).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    annualization = math.sqrt(periods_per_year)
    # Pezier & White correct the PER-PERIOD Sharpe -- so the skew/kurtosis penalty reflects the distribution shape and
    # not the annualization factor -- then the corrected ratio is annualized by the square-root-of-time rule.
    sharpe = sharpe_ratio(returns, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate) / annualization
    return (
        annualization * sharpe * (1.0 + returns.skew() / 6.0 * sharpe - returns.kurtosis() / 24.0 * sharpe**2)
    ).name.keep()


def burke_ratio(
    equity_curve: pl.Expr,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> pl.Expr:
    r"""
    Burke Ratio, the excess compound annual growth rate per unit of drawdown energy.

    The annualized excess return divided by the square root of the sum of squared drawdowns -- a return-to-pain ratio
    whose denominator (the "drawdown energy") penalizes a few deep declines more than many shallow ones:

    .. math::

        \mathrm{Burke} = \frac{\mathrm{CAGR} - r_f}{\sqrt{\sum_i D_i^2}},

    where :math:`\mathrm{CAGR}` is :func:`cagr` and :math:`D_i` the :func:`drawdown`
    series. The risk-free rate is already annualized, matching the annualized growth.

    Args:
        equity_curve: Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate subtracted from the growth (default ``0.0``). Must be finite.

    Returns:
        A single ``Float64`` value: the Burke ratio (one value in ``select``, one per group under ``.over``). ``null``
        when there are no observations.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        The denominator is the sum (not the mean) of the squared drawdowns, taken over the per-period drawdown series
        (not the maxima of distinct decline episodes, as in some Burke variants), so it grows with the record length.

        **Edge-case behavior:**

        - **Null** — a ``null`` equity is skipped (excluded from both the growth and the drawdown energy).
        - **NaN** — a ``NaN`` equity propagates, yielding ``NaN``.
        - **No drawdown** — a monotonically non-decreasing curve has zero drawdown energy, so the ratio is
          ``+/-inf`` (or ``NaN`` when the excess growth is also zero), reported rather than clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``burke_ratio(pl.col("equity"), periods_per_year=252).over("ticker")``.

    See Also:
        - :func:`ulcer_index`: The root-mean-square drawdown penalty.
        - :func:`calmar_ratio`: The single-worst-drawdown counterpart.
        - :func:`sterling_ratio`: The average-drawdown-plus-cushion counterpart.

    References:
        - Burke, G. (1994). "A Sharper Sharpe Ratio." *Futures Magazine*.
        - https://en.wikipedia.org/wiki/Drawdown_%28economics%29

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import burke_ratio
        >>>
        >>> frame = pl.DataFrame({"equity": [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4]})
        >>> frame.select(burke_ratio(pl.col("equity"), periods_per_year=1).round(4)).item()
        0.6776

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "equity_curve": [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4] + [1.0, 1.02, 1.01, 1.05, 1.08, 1.06, 1.12],
        ...     }
        ... )
        >>> reduced = burke_ratio(pl.col("equity_curve"), periods_per_year=1).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [0.6776, 0.7789]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"equity_curve": [1.1, None, 1.2, 1.15, float("nan"), 1.25, 1.4]})
        >>> frame.select(burke_ratio(pl.col("equity_curve"), periods_per_year=1).round(4)).item()
        nan
    """
    equity_curve = float64_expr(equity_curve)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    return (
        (cagr(equity_curve, periods_per_year=periods_per_year) - risk_free_rate)
        / (drawdown(equity_curve) ** 2).sum().sqrt()
    ).name.keep()


def calmar_ratio(
    equity_curve: pl.Expr,
    *,
    periods_per_year: int,
) -> pl.Expr:
    r"""
    Calmar Ratio, the compound annual growth rate per unit of maximum drawdown.

    The annualized return divided by the magnitude of the worst peak-to-trough decline -- a return-to-pain ratio that
    rewards growth and penalizes the deepest loss an investor would have lived through:

    .. math::

        \mathrm{Calmar} = \frac{\mathrm{CAGR}}{\lvert \mathrm{MDD} \rvert},

    where :math:`\mathrm{CAGR}` is :func:`cagr` and :math:`\mathrm{MDD}` is the (non-positive)
    :func:`max_drawdown`. Both are taken over the whole input series; unlike Young's original trailing 36-month
    window, the lookback here is the full sample.

    Args:
        equity_curve: Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.

    Returns:
        A single ``Float64`` value: the Calmar ratio (one value in ``select``, one per group under ``.over``). ``null``
        when there are no observations.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — a ``null`` equity is skipped (excluded from both the growth and the drawdown), so a leading warm-up
          ``null`` does not affect the result.
        - **NaN** — a ``NaN`` equity propagates, yielding ``NaN``.
        - **No drawdown** — a monotonically non-decreasing curve has zero maximum drawdown, so the ratio is ``+/-inf``
          (or ``NaN`` when the growth is also zero), reported rather than clipped. An empty (or all-null) series yields
          ``null``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``calmar_ratio(pl.col("equity"), periods_per_year=252).over("ticker")``.

    See Also:
        - :func:`cagr`: The numerator (annualized growth).
        - :func:`max_drawdown`: The denominator (worst decline).
        - :func:`recovery_ratio`: The same worst-drawdown denominator with a total-return numerator.

    References:
        - Young, T. W. (1991). "Calmar Ratio: A Smoother Tool." *Futures Magazine*.
        - https://en.wikipedia.org/wiki/Calmar_ratio

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import calmar_ratio
        >>>
        >>> frame = pl.DataFrame({"equity": [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4]})
        >>> frame.select(calmar_ratio(pl.col("equity"), periods_per_year=1).round(4)).item()
        1.0833

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "equity_curve": [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4] + [1.0, 1.02, 1.01, 1.05, 1.08, 1.06, 1.12],
        ...     }
        ... )
        >>> reduced = calmar_ratio(pl.col("equity_curve"), periods_per_year=1).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [0.8814, 1.0833]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"equity_curve": [1.1, None, 1.2, 1.15, float("nan"), 1.25, 1.4]})
        >>> frame.select(calmar_ratio(pl.col("equity_curve"), periods_per_year=1).round(4)).item()
        nan
    """
    equity_curve = float64_expr(equity_curve)
    validate_periods_per_year(periods_per_year)
    return (cagr(equity_curve, periods_per_year=periods_per_year) / max_drawdown(equity_curve).abs()).name.keep()


def common_sense_ratio(
    returns: pl.Expr,
) -> pl.Expr:
    r"""
    Common Sense Ratio, the profit factor scaled by the tail ratio.

    The product of the :func:`profit_ratio` (aggregate gain over loss) and the :func:`tail_ratio` (right-tail over
    left-tail magnitude) -- a single number that rewards both a profitable edge and a favorable tail profile:

    .. math::

        \mathrm{CSR} = \mathrm{PF} \cdot \mathrm{TR}
        = \frac{\sum_{r_i > 0} r_i}{\left\lvert \sum_{r_i < 0} r_i \right\rvert} \cdot
          \left\lvert \frac{Q_{0.95}(r)}{Q_{0.05}(r)} \right\rvert.

    A value above one means the combined profitability and tail behavior are favorable.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).

    Returns:
        A single ``Float64`` value: the common sense ratio (one value in ``select``, one per group under ``.over``).
        ``null`` when there are no returns.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped; an all-null (or empty) series yields ``null``.
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **Degenerate factors** — it inherits the degeneracies of its two factors: ``+inf`` when there are no losses
          (the profit factor diverges) or a zero left tail (the tail ratio diverges), and ``NaN`` where a ``0 * inf``
          arises; all reported rather than clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``common_sense_ratio(pl.col("returns")).over("ticker")``.

    See Also:
        - :func:`profit_ratio`: The aggregate gain-to-loss factor.
        - :func:`tail_ratio`: The right-tail to left-tail factor.
        - :func:`omega_ratio`: The whole-distribution gain-to-loss ratio about a threshold.

    References:
        - https://en.wikipedia.org/wiki/Tail_risk

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import common_sense_ratio
        >>>
        >>> frame = pl.DataFrame({"returns": [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]})
        >>> frame.select(common_sense_ratio(pl.col("returns")).round(4)).item()
        2.1081

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]
        ...         + [0.02, -0.005, 0.015, -0.01, 0.025, 0.0, -0.012],
        ...     }
        ... )
        >>> reduced = common_sense_ratio(pl.col("returns")).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [2.1081, 4.5809]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.03, None, 0.02, -0.015, float("nan"), 0.005, -0.02]})
        >>> frame.select(common_sense_ratio(pl.col("returns")).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    return profit_ratio(returns) * tail_ratio(returns)


def gain_to_pain_ratio(
    returns: pl.Expr,
) -> pl.Expr:
    r"""
    Gain to Pain Ratio, the net return over the total loss (Schwager).

    The sum of all returns divided by the magnitude of the sum of the negative returns -- Schwager's measure of return
    per unit of downside "pain":

    .. math::

        \mathrm{GPR} = \frac{\sum_i r_i}{\left\lvert \sum_{r_i < 0} r_i \right\rvert}.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).

    Returns:
        A single ``Float64`` value: the gain to pain ratio (one value in ``select``, one per group under ``.over``).
        ``null`` when there are no returns.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        It is computed on the return series as given, with no calendar resampling and no risk-free adjustment (the pure
        Schwager ratio).

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped; an all-null (or empty) series yields ``null``.
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **No losses** — with no negative returns the total loss is zero, so the ratio is ``+inf`` (or ``NaN`` when the
          net return is also zero), reported rather than clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``gain_to_pain_ratio(pl.col("returns")).over("ticker")``.

    See Also:
        - :func:`profit_ratio`: The gross-gain to gross-loss counterpart.
        - :func:`omega_ratio`: The probability-weighted gain-to-loss ratio about a threshold.
        - :func:`ulcer_performance_ratio`: A drawdown-based return-to-pain ratio.

    References:
        - Schwager, J. D. (2012). *Hedge Fund Market Wizards*. Wiley.

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import gain_to_pain_ratio
        >>>
        >>> frame = pl.DataFrame({"returns": [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]})
        >>> frame.select(gain_to_pain_ratio(pl.col("returns")).round(4)).item()
        0.4444

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]
        ...         + [0.02, -0.005, 0.015, -0.01, 0.025, 0.0, -0.012],
        ...     }
        ... )
        >>> reduced = gain_to_pain_ratio(pl.col("returns")).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [0.4444, 1.2222]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.03, None, 0.02, -0.015, float("nan"), 0.005, -0.02]})
        >>> frame.select(gain_to_pain_ratio(pl.col("returns")).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    return returns.mean() / (-returns).clip(lower_bound=0.0).mean()


def omega_ratio(
    returns: pl.Expr,
    *,
    threshold: float = 0.0,
) -> pl.Expr:
    r"""
    Omega Ratio, the ratio of probability-weighted gains to losses about a threshold.

    The mean gain above a threshold divided by the mean loss below it -- a measure that, unlike the Sharpe ratio, uses
    the whole return distribution rather than only its first two moments:

    .. math::

        \Omega(\tau) = \frac{\mathbb{E}\!\left[\max(r - \tau, 0)\right]}{\mathbb{E}\!\left[\max(\tau - r, 0)\right]},

    where :math:`\tau` is ``threshold`` (the minimum acceptable return). A value above one means the upside outweighs
    the downside at that threshold.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        threshold: The return level separating gains from losses / the minimum acceptable return (default ``0.0``).
            Must be finite.

    Returns:
        A single ``Float64`` value: the omega ratio (one value in ``select``, one per group under ``.over``). ``null``
        when there are no returns.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``threshold`` is not finite.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped; an all-null (or empty) series yields ``null``.
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **No downside** — when no return is below the threshold the mean loss is zero, so the ratio is ``+inf`` (or
          ``NaN`` when there is also no upside), reported rather than clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``omega_ratio(pl.col("returns")).over("ticker")``.

    See Also:
        - :func:`gain_to_pain_ratio`: The net-return over total-loss sibling about a zero threshold.
        - :func:`sortino_ratio`: The downside-deviation risk-adjusted alternative.
        - :func:`sharpe_ratio`: The moment-based risk-adjusted ratio.

    References:
        - Keating, C. & Shadwick, W. F. (2002). "A Universal Performance Measure." *The Journal of Performance
          Measurement*, 6(3), 59-84.
        - https://en.wikipedia.org/wiki/Omega_ratio

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import omega_ratio
        >>>
        >>> frame = pl.DataFrame({"returns": [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]})
        >>> frame.select(omega_ratio(pl.col("returns")).round(4)).item()
        1.4444

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]
        ...         + [0.02, -0.005, 0.015, -0.01, 0.025, 0.0, -0.012],
        ...     }
        ... )
        >>> reduced = omega_ratio(pl.col("returns")).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [1.4444, 2.2222]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.03, None, 0.02, -0.015, float("nan"), 0.005, -0.02]})
        >>> frame.select(omega_ratio(pl.col("returns")).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    validate_finite(threshold, "threshold")
    excess = returns - threshold
    mean_gain = excess.clip(lower_bound=0.0).mean()
    mean_loss = (-excess).clip(lower_bound=0.0).mean()
    return mean_gain / mean_loss


def omega_ratio_rolling(
    returns: pl.Expr,
    window: int,
    *,
    threshold: float = 0.0,
) -> pl.Expr:
    r"""
    Rolling Omega Ratio over a window — the windowed twin of :func:`omega_ratio`.

    The mean gain above a threshold divided by the mean loss below it, over each trailing window:

    .. math::

        \Omega_t = \frac{\overline{\max(r - \tau, 0)}}{\overline{\max(\tau - r, 0)}}, \qquad n = \text{window},

    where :math:`\tau` is ``threshold`` and the means are taken over the window.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        window: Number of observations in the moving window. Must be ``>= 1``.
        threshold: The return level separating gains from losses / the minimum acceptable return (default ``0.0``).
            Must be finite.

    Returns:
        The rolling omega ratio for each row, the same length as the input. The first ``window - 1`` rows are ``null``
        (warm-up): the window must hold ``window`` non-null values before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``, or if ``threshold`` is not finite.

    Note:
        **Correctness** -- each window matches an independent reference oracle (the reducing :func:`omega_ratio`
        recomputed over the window) within the documented dynamic range. The clipped means ride Polars' incremental
        sliding kernel, so a window whose scale sits tens of orders of magnitude below a value that has already slid
        out can inherit a stale residue -- the float-conditioning limit ``CORRECTNESS.md`` documents for the rolling
        sums; the bit-constant edge is guarded exactly, and no real market series builds that spread.

        **Edge-case behavior:**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **No downside** — a window with no return below the threshold has zero mean loss, so the ratio is ``+inf`` (or
          ``NaN`` when there is also no upside), reported rather than clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` so the window never spans series boundaries.

    See Also:
        - :func:`omega_ratio`: The whole-series reducing form.
        - :func:`sortino_ratio_rolling`: The rolling downside-deviation risk-adjusted ratio.
        - :func:`sharpe_ratio_rolling`: The rolling total-volatility risk-adjusted ratio.

    References:
        - Keating, C. & Shadwick, W. F. (2002). "A Universal Performance Measure." *The Journal of Performance
          Measurement*, 6(3), 59-84.
        - https://en.wikipedia.org/wiki/Omega_ratio

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import omega_ratio_rolling
        >>>
        >>> frame = pl.DataFrame({"returns": [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015]})
        >>> frame.select(omega_ratio_rolling(pl.col("returns"), 3).round(4))["returns"].to_list()
        [None, None, 2.0, 1.0, 5.0, 2.0, 1.3333]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015]
        ...         + [0.02, -0.005, 0.015, -0.01, 0.025, 0.0, -0.012],
        ...     }
        ... )
        >>> rolling = omega_ratio_rolling(pl.col("returns"), 3).over("ticker").round(4)
        >>> frame.select(rolling.alias("m"))["m"].to_list()
        [None, None, 2.0, 1.0, 5.0, 2.0, 1.3333, None, None, 7.0, 1.0, 4.0, 2.5, 2.0833]

        A ``null`` (which voids every window that spans it) and a ``NaN`` (which propagates to its windows) make the
        missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.01, None, 0.03, -0.01, 0.02, float("nan"), -0.015, 0.02, 0.01]})
        >>> frame.select(omega_ratio_rolling(pl.col("returns"), 3).round(4))["returns"].to_list()
        [None, None, None, None, 5.0, nan, nan, nan, 2.0]
    """
    returns = float64_expr(returns)
    validate_window(window)
    validate_finite(threshold, "threshold")
    excess = returns - threshold
    gain = excess.clip(lower_bound=0.0)
    loss = (-excess).clip(lower_bound=0.0)
    mean_gain_raw = gain.rolling_mean(window, min_samples=window).clip(lower_bound=0.0)
    mean_loss_raw = loss.rolling_mean(window, min_samples=window).clip(lower_bound=0.0)
    # Force a window with no gain (resp. no loss) to exactly zero, not the tiny residue the one-pass sliding sum can
    # leave once a large value exits the window: an all-at-threshold window is then ``0 / 0`` = ``NaN`` (not a spurious
    # ``+inf`` from a gain residue over a zeroed loss), and a one-sided window gives ``0`` or ``+inf`` as it should.
    no_gain = gain.rolling_max(window, min_samples=window) == 0.0
    no_loss = loss.rolling_max(window, min_samples=window) == 0.0
    mean_gain = pl.when(no_gain).then(0.0).otherwise(mean_gain_raw)
    mean_loss = pl.when(no_loss).then(0.0).otherwise(mean_loss_raw)
    return (mean_gain / mean_loss).name.keep()


def pain_ratio(
    equity_curve: pl.Expr,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> pl.Expr:
    r"""
    Pain Ratio, the excess compound annual growth rate per unit of pain index.

    The annualized excess return divided by the :func:`pain_index` (the average drawdown depth) -- a
    return-to-pain ratio that uses the mean, rather than the worst or the root-mean-square, drawdown as the denominator:

    .. math::

        \mathrm{pain\ ratio} = \frac{\mathrm{CAGR} - r_f}{\mathrm{PI}},

    where :math:`\mathrm{CAGR}` is :func:`cagr` and :math:`\mathrm{PI}` the
    :func:`pain_index`. The risk-free rate is already annualized, matching the annualized growth.

    Args:
        equity_curve: Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate subtracted from the growth (default ``0.0``). Must be finite.

    Returns:
        A single ``Float64`` value: the pain ratio (one value in ``select``, one per group under ``.over``). ``null``
        when there are no observations.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — a ``null`` equity is skipped (excluded from both the growth and the pain index).
        - **NaN** — a ``NaN`` equity propagates, yielding ``NaN``.
        - **No drawdown** — a monotonically non-decreasing curve has a zero pain index, so the ratio is ``+/-inf`` (or
          ``NaN`` when the excess growth is also zero), reported rather than clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``pain_ratio(pl.col("equity"), periods_per_year=252).over("ticker")``.

    See Also:
        - :func:`pain_index`: The denominator (average drawdown depth).
        - :func:`sterling_ratio`: The same average-drawdown denominator offset by a fixed cushion.
        - :func:`ulcer_performance_ratio`: The root-mean-square-drawdown counterpart.

    References:
        - Becker, T. (2006). "The Pain Index and Pain Ratio." *Zephyr Associates*.

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import pain_ratio
        >>>
        >>> frame = pl.DataFrame({"equity": [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4]})
        >>> frame.select(pain_ratio(pl.col("equity"), periods_per_year=1).round(4)).item()
        2.7447

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "equity_curve": [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4] + [1.0, 1.02, 1.01, 1.05, 1.08, 1.06, 1.12],
        ...     }
        ... )
        >>> reduced = pain_ratio(pl.col("equity_curve"), periods_per_year=1).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [2.7447, 4.0339]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"equity_curve": [1.1, None, 1.2, 1.15, float("nan"), 1.25, 1.4]})
        >>> frame.select(pain_ratio(pl.col("equity_curve"), periods_per_year=1).round(4)).item()
        nan
    """
    equity_curve = float64_expr(equity_curve)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    return (
        (cagr(equity_curve, periods_per_year=periods_per_year) - risk_free_rate) / pain_index(equity_curve)
    ).name.keep()


def probabilistic_sharpe_ratio(
    returns: pl.Expr,
    *,
    periods_per_year: int,
    benchmark_sharpe: float = 0.0,
    risk_free_rate: float = 0.0,
) -> pl.Expr:
    r"""
    Probabilistic Sharpe Ratio (PSR), the confidence that the true Sharpe ratio exceeds a benchmark.

    The probability that the observed (per-period) Sharpe ratio is greater than a benchmark Sharpe ratio, correcting the
    estimation error for the track-record length and for the returns' skewness and kurtosis (non-normal returns inflate
    the estimator's variance):

    .. math::

        \mathrm{PSR}(\mathrm{SR}^{*}) = \Phi\!\left(
            \frac{(\widehat{\mathrm{SR}} - \mathrm{SR}^{*}) \, \sqrt{n - 1}}
                 {\sqrt{1 - \gamma_3 \widehat{\mathrm{SR}} + \tfrac{\gamma_4 - 1}{4} \widehat{\mathrm{SR}}^{2}}}
        \right),

    where :math:`\Phi` is the standard-normal CDF, :math:`\widehat{\mathrm{SR}}` the non-annualized excess Sharpe ratio,
    :math:`\mathrm{SR}^{*}` the ``benchmark_sharpe``, :math:`n` the number of returns, :math:`\gamma_3` the (population)
    skewness, and :math:`\gamma_4` the (population, non-excess) kurtosis. The per-period risk-free rate is the geometric
    conversion :math:`(1 + \texttt{risk\_free\_rate})^{1/P} - 1`.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        periods_per_year: Observations per year, used only to convert the annualized risk-free rate to a per-period rate
            (canonically ``252`` for daily). Must be ``>= 1``.
        benchmark_sharpe: The (non-annualized) benchmark Sharpe ratio :math:`\mathrm{SR}^{*}` to beat (default ``0.0``).
            Must be finite.
        risk_free_rate: The annualized risk-free rate, converted to a per-period rate geometrically (default ``0.0``).
            Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + risk_free_rate >= 0``).

    Returns:
        A single ``Float64`` value in ``[0, 1]``: the probabilistic Sharpe ratio (one value in ``select``, one per group
        under ``.over``). ``null`` when fewer than two returns are present (the sample Sharpe ratio is undefined).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``, or if ``benchmark_sharpe`` or ``risk_free_rate`` is not finite, or if
            ``risk_free_rate < -1``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        The kurtosis term uses the non-excess (raw) kurtosis :math:`\gamma_4`, exactly as in Bailey & López de Prado: a
        normal sample (:math:`\gamma_4 = 3`) recovers the classic Lo standard error :math:`\sqrt{(1 + \mathrm{SR}^2 / 2)
        / (n - 1)}`.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped (excluded from every moment).
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **Fewer than two returns** — the sample Sharpe ratio is undefined, so the result is ``null``.
        - **Zero volatility** — a constant series has an undefined Sharpe ratio and undefined moments, yielding ``NaN``.
        - **Degenerate** — a negative variance under the inner square root (extreme skewness or kurtosis) yields
          ``NaN``; an exactly-zero inner variance (a measure-zero boundary) yields the limiting ``0`` or ``1``,
          reported rather than forced into range.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``probabilistic_sharpe_ratio(pl.col("returns"), periods_per_year=252).over("ticker")``.

    See Also:
        - :func:`sharpe_ratio`: The point estimate this attaches a confidence level to.
        - :func:`adjusted_sharpe_ratio`: The point-estimate correction for the same non-normality.
        - :func:`sortino_ratio`: The downside-deviation Sharpe variant for the same asymmetric returns.

    References:
        - Bailey, D. H. & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *Journal of Risk*, 15(2),
          3-44.
        - https://doi.org/10.21314/JOR.2012.255
        - https://en.wikipedia.org/wiki/Sharpe_ratio

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import probabilistic_sharpe_ratio
        >>>
        >>> frame = pl.DataFrame({"returns": [0.012, 0.008, 0.015, -0.004, 0.02, 0.006, 0.011, -0.003, 0.014, 0.009]})
        >>> frame.select(probabilistic_sharpe_ratio(pl.col("returns"), periods_per_year=252).round(4)).item()
        0.9922

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]
        ...         + [0.02, -0.005, 0.015, -0.01, 0.025, 0.0, -0.012],
        ...     }
        ... )
        >>> reduced = probabilistic_sharpe_ratio(pl.col("returns"), periods_per_year=252).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [0.6475, 0.7851]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.03, None, 0.02, -0.015, float("nan"), 0.005, -0.02]})
        >>> frame.select(probabilistic_sharpe_ratio(pl.col("returns"), periods_per_year=252).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    validate_periods_per_year(periods_per_year)
    validate_finite(benchmark_sharpe, "benchmark_sharpe")
    validate_finite(risk_free_rate, "risk_free_rate")
    rf_period = per_period_rate(risk_free_rate, periods_per_year, name="risk_free_rate")
    excess = returns - rf_period
    # volatility at periods_per_year=1 is the per-period sample deviation with the exactly-constant series pinned to
    # exactly zero, so a zero-dispersion excess degenerates to a signed infinity, never a residue-driven huge finite.
    sharpe = excess.mean() / volatility(excess, periods_per_year=1)
    raw_kurtosis = returns.kurtosis() + 3.0
    observations = returns.drop_nulls().len().cast(pl.Int64)
    standard_error = (1.0 - returns.skew() * sharpe + (raw_kurtosis - 1.0) / 4.0 * sharpe**2).sqrt()
    argument = (sharpe - benchmark_sharpe) * (observations - 1).sqrt() / standard_error
    return argument.map_batches(_normal_cdf, return_dtype=pl.Float64, returns_scalar=True)


def recovery_ratio(
    equity_curve: pl.Expr,
) -> pl.Expr:
    r"""
    Recovery Factor, the total return per unit of maximum drawdown.

    The total return divided by the magnitude of the worst peak-to-trough decline -- how many times the strategy
    recovered its deepest loss over the whole period (a losing curve, a negative total return, reports a negative
    factor):

    .. math::

        \mathrm{recovery} = \frac{\mathrm{total\ return}}{\lvert \mathrm{MDD} \rvert},

    where the total return is :func:`total_return` and :math:`\mathrm{MDD}` is the (non-positive)
    :func:`max_drawdown`.

    Args:
        equity_curve: Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.

    Returns:
        A single ``Float64`` value: the recovery factor (one value in ``select``, one per group under ``.over``).
        ``null`` when there are no observations.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        Only the drawdown denominator is taken in magnitude; the total-return numerator keeps its sign, so a losing
        curve (a negative total return) reports a negative recovery factor.

        **Edge-case behavior:**

        - **Null** — a ``null`` equity is skipped; an all-null (or empty) series yields ``null``.
        - **NaN** — a ``NaN`` equity propagates, yielding ``NaN``.
        - **No drawdown** — a monotonically non-decreasing curve has zero maximum drawdown, so the ratio is ``+/-inf``
          with the sign of the total return (or ``NaN`` when the total return is also zero), reported rather than
          clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``recovery_ratio(pl.col("equity")).over("ticker")``.

    See Also:
        - :func:`total_return`: The numerator (overall growth).
        - :func:`max_drawdown`: The denominator (worst decline).
        - :func:`calmar_ratio`: The annualized-growth counterpart over the same drawdown.

    References:
        - Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies* (2nd ed.). Wiley.

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import recovery_ratio
        >>>
        >>> frame = pl.DataFrame({"equity": [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4]})
        >>> frame.select(recovery_ratio(pl.col("equity")).round(4)).item()
        8.8

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "equity_curve": [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4] + [1.0, 1.02, 1.01, 1.05, 1.08, 1.06, 1.12],
        ...     }
        ... )
        >>> reduced = recovery_ratio(pl.col("equity_curve")).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [6.48, 8.8]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"equity_curve": [1.1, None, 1.2, 1.15, float("nan"), 1.25, 1.4]})
        >>> frame.select(recovery_ratio(pl.col("equity_curve")).round(4)).item()
        nan
    """
    equity_curve = float64_expr(equity_curve)
    # The numerator keeps its sign (a losing curve reports a negative factor); only the drawdown is taken in magnitude.
    return (total_return(equity_curve) / max_drawdown(equity_curve).abs()).name.keep()


def sharpe_ratio(
    returns: pl.Expr,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> pl.Expr:
    r"""
    Sharpe Ratio, the annualized excess return per unit of total volatility.

    The mean excess return divided by its standard deviation, annualized by the square-root-of-time rule -- the textbook
    reward-to-variability measure:

    .. math::

        \mathrm{Sharpe} = \frac{\bar{e}}{\sigma_e} \, \sqrt{P}, \qquad e_i = r_i - r_f,

    where :math:`\sigma_e` is the sample standard deviation (``ddof = 1``) of the excess returns :math:`e_i`, :math:`P`
    is ``periods_per_year``, and the per-period risk-free rate is the geometric conversion
    :math:`r_f = (1 + \texttt{risk\_free\_rate})^{1/P} - 1`.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate, converted to a per-period rate geometrically (default ``0.0``).
            Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + risk_free_rate >= 0``).

    Returns:
        A single ``Float64`` value: the annualized Sharpe ratio (one value in ``select``, one per group under
        ``.over``). ``null`` when fewer than two returns are present (the sample standard deviation is undefined).

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite or is ``< -1``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped (excluded from the mean and the standard deviation).
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **Zero volatility** — a constant excess series has zero dispersion, so the ratio is ``+/-inf`` (or ``NaN``
          when the mean excess is also zero), reported rather than clipped.
        - **Fewer than two returns** — the sample standard deviation is undefined, so the result is ``null``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``sharpe_ratio(pl.col("returns"), periods_per_year=252).over("ticker")``.

    See Also:
        - :func:`sortino_ratio`: The downside-only counterpart (penalizes only harmful volatility).
        - :func:`volatility`: The denominator (total dispersion).
        - :func:`adjusted_sharpe_ratio`: The higher-moment correction for non-normal returns.

    References:
        - Sharpe, W. F. (1994). "The Sharpe Ratio." *The Journal of Portfolio Management*, 21(1), 49-58.
        - https://doi.org/10.3905/jpm.1994.409501
        - https://en.wikipedia.org/wiki/Sharpe_ratio

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import sharpe_ratio
        >>>
        >>> frame = pl.DataFrame({"returns": [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]})
        >>> frame.select(sharpe_ratio(pl.col("returns"), periods_per_year=252).round(4)).item()
        2.4285

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]
        ...         + [0.02, -0.005, 0.015, -0.01, 0.025, 0.0, -0.012],
        ...     }
        ... )
        >>> reduced = sharpe_ratio(pl.col("returns"), periods_per_year=252).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [2.4285, 4.9645]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.03, None, 0.02, -0.015, float("nan"), 0.005, -0.02]})
        >>> frame.select(sharpe_ratio(pl.col("returns"), periods_per_year=252).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    rf_period = per_period_rate(risk_free_rate, periods_per_year, name="risk_free_rate")
    excess = returns - rf_period
    return excess.mean() * periods_per_year / volatility(excess, periods_per_year=periods_per_year)


def sharpe_ratio_rolling(
    returns: pl.Expr,
    window: int,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> pl.Expr:
    r"""
    Rolling Sharpe Ratio over a window — the windowed twin of :func:`sharpe_ratio`.

    The mean excess return of each trailing window divided by its sample standard deviation, annualized by the
    square-root-of-time rule:

    .. math::

        \mathrm{Sharpe}_t = \frac{\bar{e}_t}{\sigma_{e,t}}\,\sqrt{P}, \qquad e_i = r_i - r_f, \quad n = \text{window},

    where :math:`\sigma_{e,t}` is the sample standard deviation (``ddof = 1``) over the window, :math:`P` is
    ``periods_per_year``, and the per-period risk-free rate is :math:`r_f = (1 + \texttt{risk\_free\_rate})^{1/P} - 1`.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        window: Number of observations in the moving window. Must be ``>= 2``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate, converted to a per-period rate geometrically (default ``0.0``).
            Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + risk_free_rate >= 0``).

    Returns:
        The rolling Sharpe ratio for each row, the same length as the input. The first ``window - 1`` rows are ``null``
        (warm-up): the window must hold ``window`` non-null values before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 2``, ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite or is ``< -1``.

    Note:
        **Correctness** -- each window matches an independent reference oracle (the reducing :func:`sharpe_ratio`
        recomputed over the window).

        **Edge-case behavior:**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **Zero volatility** — a constant window has zero dispersion, so the ratio is ``+/-inf`` (or ``NaN`` when the
          mean excess is also zero), reported rather than clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` so the window never spans series boundaries.

    See Also:
        - :func:`sharpe_ratio`: The whole-series reducing form.
        - :func:`volatility_rolling`: The denominator.
        - :func:`sortino_ratio_rolling`: The downside-only rolling counterpart.

    References:
        - Sharpe, W. F. (1994). "The Sharpe Ratio." *The Journal of Portfolio Management*, 21(1), 49-58.
        - https://doi.org/10.3905/jpm.1994.409501
        - https://en.wikipedia.org/wiki/Sharpe_ratio

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import sharpe_ratio_rolling
        >>>
        >>> frame = pl.DataFrame({"returns": [0.03, -0.01, 0.02, -0.015, 0.025, -0.005, 0.02]})
        >>> frame.select(sharpe_ratio_rolling(pl.col("returns"), 3, periods_per_year=252).round(4))["returns"].to_list()
        [None, None, 10.1678, -1.3977, 7.2837, 1.271, 13.1689]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [0.03, -0.01, 0.02, -0.015, 0.025, -0.005, 0.02]
        ...         + [0.02, -0.005, 0.015, -0.01, 0.025, 0.0, -0.012],
        ...     }
        ... )
        >>> rolling = sharpe_ratio_rolling(pl.col("returns"), 3, periods_per_year=252).over("ticker").round(4)
        >>> frame.select(rolling.alias("m"))["m"].to_list()
        [None, None, 10.1678, -1.3977, 7.2837, 1.271, 13.1689, None, None, 12.0, -0.0, 8.8056, 4.4028, 3.6441]

        A ``null`` (which voids every window that spans it) and a ``NaN`` (which propagates to its windows) make the
        missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.03, None, 0.02, -0.015, 0.025, float("nan"), 0.02, -0.01, 0.015]})
        >>> frame.select(sharpe_ratio_rolling(pl.col("returns"), 3, periods_per_year=252).round(4))["returns"].to_list()
        [None, None, None, None, 7.2837, nan, nan, nan, 8.2305]
    """
    returns = float64_expr(returns)
    validate_window(window, minimum=2)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    rf_period = per_period_rate(risk_free_rate, periods_per_year, name="risk_free_rate")
    excess = returns - rf_period
    return (
        excess.rolling_mean(window, min_samples=window)
        * periods_per_year
        / volatility_rolling(excess, window, periods_per_year=periods_per_year)
    )


def sortino_ratio(
    returns: pl.Expr,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> pl.Expr:
    r"""
    Sortino Ratio, the annualized excess return per unit of downside deviation.

    The mean excess return divided by the downside deviation about the same target, annualized by the
    square-root-of-time rule -- a Sharpe variant that penalizes only harmful (below-target) volatility:

    .. math::

        \mathrm{Sortino} = \frac{\bar{e}}{\mathrm{DD}} \, \sqrt{P}, \qquad e_i = r_i - r_f,

    where :math:`\mathrm{DD} = \sqrt{\tfrac{1}{n} \sum_i \min(e_i, 0)^2}` is the per-period downside deviation about the
    target, :math:`P` is ``periods_per_year``, and the per-period risk-free target is
    :math:`r_f = (1 + \texttt{risk\_free\_rate})^{1/P} - 1`.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate, converted to a per-period rate geometrically (default ``0.0``).
            Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + risk_free_rate >= 0``).

    Returns:
        A single ``Float64`` value: the annualized Sortino ratio (one value in ``select``, one per group under
        ``.over``). ``null`` when there are no returns.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite or is ``< -1``.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — a ``null`` return is skipped (excluded from the mean and the downside deviation).
        - **NaN** — a ``NaN`` return propagates, yielding ``NaN``.
        - **No downside** — when every excess return is at or above the target the downside deviation is zero, so the
          ratio is ``+/-inf`` (or ``NaN`` when the mean excess is also zero), reported rather than clipped. An empty (or
          all-null) series yields ``null``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``sortino_ratio(pl.col("returns"), periods_per_year=252).over("ticker")``.

    See Also:
        - :func:`sharpe_ratio`: The two-sided counterpart (penalizes all volatility).
        - :func:`downside_deviation`: The denominator (downside-only dispersion).
        - :func:`omega_ratio`: The threshold-based gain-to-loss alternative.

    References:
        - Sortino, F. A. & Price, L. N. (1994). "Performance Measurement in a Downside Risk Framework." *The Journal of
          Investing*, 3(3), 59-64.
        - https://doi.org/10.3905/joi.3.3.59
        - https://en.wikipedia.org/wiki/Sortino_ratio

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import sortino_ratio
        >>>
        >>> frame = pl.DataFrame({"returns": [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]})
        >>> frame.select(sortino_ratio(pl.col("returns"), periods_per_year=252).round(4)).item()
        4.4567

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02]
        ...         + [0.02, -0.005, 0.015, -0.01, 0.025, 0.0, -0.012],
        ...     }
        ... )
        >>> reduced = sortino_ratio(pl.col("returns"), periods_per_year=252).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [4.4567, 12.0723]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.03, None, 0.02, -0.015, float("nan"), 0.005, -0.02]})
        >>> frame.select(sortino_ratio(pl.col("returns"), periods_per_year=252).round(4)).item()
        nan
    """
    returns = float64_expr(returns)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    rf_period = per_period_rate(risk_free_rate, periods_per_year, name="risk_free_rate")
    excess = returns - rf_period
    return (
        excess.mean()
        * periods_per_year
        / downside_deviation(returns, periods_per_year=periods_per_year, threshold=rf_period)
    )


def sortino_ratio_rolling(
    returns: pl.Expr,
    window: int,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> pl.Expr:
    r"""
    Rolling Sortino Ratio over a window — the windowed twin of :func:`sortino_ratio`.

    The mean excess return of each trailing window divided by its downside deviation about the same target, annualized
    by the square-root-of-time rule:

    .. math::

        \mathrm{Sortino}_t = \frac{\bar{e}_t}{\mathrm{DD}_t}\,\sqrt{P}, \qquad e_i = r_i - r_f, \quad n = \text{window},

    where :math:`\mathrm{DD}_t` is the rolling :func:`downside_deviation_rolling` about the target, :math:`P` is
    ``periods_per_year``, and the per-period target is :math:`r_f = (1 + \texttt{risk\_free\_rate})^{1/P} - 1`.

    Args:
        returns: Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).
        window: Number of observations in the moving window. Must be ``>= 1``.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate, converted to a per-period rate geometrically (default ``0.0``).
            Must be finite and ``>= -1`` (the geometric per-period conversion needs ``1 + risk_free_rate >= 0``).

    Returns:
        The rolling Sortino ratio for each row, the same length as the input. The first ``window - 1`` rows are ``null``
        (warm-up): the window must hold ``window`` non-null values before a result is emitted.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``window < 1``, ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite or is ``< -1``.

    Note:
        **Correctness** -- each window matches an independent reference oracle (the reducing :func:`sortino_ratio`
        recomputed over the window).

        **Edge-case behavior:**

        - **Null** — a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values).
        - **NaN** — a ``NaN`` inside the window propagates, yielding ``NaN`` there.
        - **No downside** — a window with every excess return at or above the target has zero downside deviation, so the
          ratio is ``+/-inf`` (or ``NaN`` when the mean excess is also zero), reported rather than clipped.
        - **Partitioning** — wrap the call in ``.over(...)`` so the window never spans series boundaries.

    See Also:
        - :func:`sortino_ratio`: The whole-series reducing form.
        - :func:`downside_deviation_rolling`: The denominator.
        - :func:`sharpe_ratio_rolling`: The two-sided rolling counterpart.

    References:
        - Sortino, F. A. & Price, L. N. (1994). "Performance Measurement in a Downside Risk Framework." *The Journal of
          Investing*, 3(3), 59-64.
        - https://doi.org/10.3905/joi.3.3.59
        - https://en.wikipedia.org/wiki/Sortino_ratio

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import sortino_ratio_rolling
        >>>
        >>> frame = pl.DataFrame({"returns": [0.03, -0.01, 0.02, -0.015, 0.025, -0.005, 0.02]})
        >>> frame.select(sortino_ratio_rolling(pl.col("returns"), 3, periods_per_year=252).round(4))[
        ...     "returns"
        ... ].to_list()
        [None, None, 36.6606, -2.542, 18.3303, 2.8983, 73.3212]

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "returns": [0.03, -0.01, 0.02, -0.015, 0.025, -0.005, 0.02]
        ...         + [0.02, -0.005, 0.015, -0.01, 0.025, 0.0, -0.012],
        ...     }
        ... )
        >>> rolling = sortino_ratio_rolling(pl.col("returns"), 3, periods_per_year=252).over("ticker").round(4)
        >>> frame.select(rolling.alias("m"))["m"].to_list()
        [None, None, 36.6606, -2.542, 18.3303, 2.8983, 73.3212, None, None, 54.9909, -0.0, 27.4955, 13.7477, 9.9289]

        A ``null`` (which voids every window that spans it) and a ``NaN`` (which propagates to its windows) make the
        missing-data handling visible:

        >>> frame = pl.DataFrame({"returns": [0.03, None, 0.02, -0.015, 0.025, float("nan"), 0.02, -0.01, 0.015]})
        >>> frame.select(sortino_ratio_rolling(pl.col("returns"), 3, periods_per_year=252).round(4))[
        ...     "returns"
        ... ].to_list()
        [None, None, None, None, 18.3303, nan, nan, nan, 22.9129]
    """
    returns = float64_expr(returns)
    validate_window(window)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    rf_period = per_period_rate(risk_free_rate, periods_per_year, name="risk_free_rate")
    excess = returns - rf_period
    return (
        excess.rolling_mean(window, min_samples=window)
        * periods_per_year
        / downside_deviation_rolling(returns, window, periods_per_year=periods_per_year, threshold=rf_period)
    )


def sterling_ratio(
    equity_curve: pl.Expr,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
    excess: float = 0.10,
) -> pl.Expr:
    r"""
    Sterling Ratio, the excess compound annual growth rate per unit of average drawdown plus a cushion.

    The annualized excess return divided by the average drawdown depth (the pain index) offset by a fixed cushion -- a
    return-to-pain ratio whose ``+10%`` term keeps the denominator away from zero for low-drawdown records. This is the
    pain-index variant; the classic Deane Sterling Jones form instead averages the largest per-year drawdowns:

    .. math::

        \mathrm{Sterling} = \frac{\mathrm{CAGR} - r_f}{\mathrm{PI} + \texttt{excess}},

    where :math:`\mathrm{CAGR}` is :func:`cagr`, :math:`\mathrm{PI}` the
    :func:`pain_index` (the average drawdown), and ``excess`` the cushion (canonically ``0.10``).

    Args:
        equity_curve: Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate subtracted from the growth (default ``0.0``). Must be finite.
        excess: The fixed cushion added to the average drawdown denominator (default ``0.10``). Must be finite.

    Returns:
        A single ``Float64`` value: the Sterling ratio (one value in ``select``, one per group under ``.over``).
        ``null`` when there are no observations.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` or ``excess`` is not finite.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — a ``null`` equity is skipped (excluded from both the growth and the average drawdown).
        - **NaN** — a ``NaN`` equity propagates, yielding ``NaN``.
        - **Zero denominator** — with the default positive cushion the denominator never vanishes; an ``excess`` of zero
          with a drawdown-free curve gives ``+/-inf`` (or ``NaN`` when the excess growth is also zero), reported rather
          than clipped. An empty (or all-null) series yields ``null``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``sterling_ratio(pl.col("equity"), periods_per_year=252).over("ticker")``.

    See Also:
        - :func:`pain_index`: The average drawdown in the denominator.
        - :func:`pain_ratio`: The same average-drawdown denominator without the cushion.
        - :func:`calmar_ratio`: The single-worst-drawdown counterpart.

    References:
        - Kestner, L. N. (1996). "Getting a Handle on True Performance." *Futures Magazine*.
        - https://en.wikipedia.org/wiki/Sterling_ratio

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import sterling_ratio
        >>>
        >>> frame = pl.DataFrame({"equity": [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4]})
        >>> frame.select(sterling_ratio(pl.col("equity"), periods_per_year=1).round(4)).item()
        0.4175

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "equity_curve": [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4] + [1.0, 1.02, 1.01, 1.05, 1.08, 1.06, 1.12],
        ...     }
        ... )
        >>> reduced = sterling_ratio(pl.col("equity_curve"), periods_per_year=1).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [0.1569, 0.4175]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"equity_curve": [1.1, None, 1.2, 1.15, float("nan"), 1.25, 1.4]})
        >>> frame.select(sterling_ratio(pl.col("equity_curve"), periods_per_year=1).round(4)).item()
        nan
    """
    equity_curve = float64_expr(equity_curve)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    validate_finite(excess, "excess")
    return (
        (cagr(equity_curve, periods_per_year=periods_per_year) - risk_free_rate) / (pain_index(equity_curve) + excess)
    ).name.keep()


def ulcer_performance_ratio(
    equity_curve: pl.Expr,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> pl.Expr:
    r"""
    Ulcer Performance Index (a.k.a. Martin Ratio), the excess compound annual growth rate per unit of ulcer index.

    The annualized excess return divided by the ulcer index -- a return-to-pain ratio whose denominator weights drawdown
    by both depth and duration (the root-mean-square drawdown), so prolonged declines weigh more than brief ones:

    .. math::

        \mathrm{UPI} = \frac{\mathrm{CAGR} - \texttt{risk\_free\_rate}}{\mathrm{UI}},

    where :math:`\mathrm{CAGR}` is :func:`cagr` and :math:`\mathrm{UI}` is the (non-negative)
    :func:`ulcer_index`. The risk-free rate is already annualized, matching the annualized growth.

    Args:
        equity_curve: Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.
        periods_per_year: Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.
        risk_free_rate: The annualized risk-free rate subtracted from the growth (default ``0.0``). Must be finite.

    Returns:
        A single ``Float64`` value: the ulcer performance index (one value in ``select``, one per group under
        ``.over``). ``null`` when there are no observations.

    Raises:
        TypeError: If any input is not a ``pl.Expr``.
        ValueError: If ``periods_per_year < 1``, or if ``risk_free_rate`` is not finite.

    Note:
        **Correctness** -- the result is checked against an independent reference oracle on every input, and every edge
        case (missing data and boundaries) is given a defined behavior.

        **Edge-case behavior:**

        - **Null** — a ``null`` equity is skipped (excluded from both the growth and the ulcer index).
        - **NaN** — a ``NaN`` equity propagates, yielding ``NaN``.
        - **No drawdown** — a monotonically non-decreasing curve has a zero ulcer index, so the ratio is ``+/-inf`` (or
          ``NaN`` when the excess growth is also zero), reported rather than clipped. An empty (or all-null) series
          yields ``null``.
        - **Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel, e.g.
          ``ulcer_performance_ratio(pl.col("equity"), periods_per_year=252).over("ticker")``.

    See Also:
        - :func:`ulcer_index`: The denominator (depth-and-duration drawdown).
        - :func:`pain_ratio`: The average-drawdown counterpart in the same return-to-pain family.
        - :func:`calmar_ratio`: The companion return-to-pain ratio scaled by the single worst drawdown.

    References:
        - Martin, P. G. & McCann, B. B. (1989). *The Investor's Guide to Fidelity Funds*. Wiley.
        - https://en.wikipedia.org/wiki/Ulcer_index

    Examples:
        >>> import polars as pl
        >>> from pomata.metrics import ulcer_performance_ratio
        >>>
        >>> frame = pl.DataFrame({"equity": [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4]})
        >>> frame.select(ulcer_performance_ratio(pl.col("equity"), periods_per_year=1).round(4)).item()
        1.7927

        On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:

        >>> frame = pl.DataFrame(
        ...     {
        ...         "ticker": ["A"] * 7 + ["B"] * 7,
        ...         "equity_curve": [1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4] + [1.0, 1.02, 1.01, 1.05, 1.08, 1.06, 1.12],
        ...     }
        ... )
        >>> reduced = ulcer_performance_ratio(pl.col("equity_curve"), periods_per_year=1).over("ticker").round(4)
        >>> frame.select(reduced.alias("m"))["m"].unique().sort().to_list()
        [1.7927, 2.0609]

        A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data handling visible:

        >>> frame = pl.DataFrame({"equity_curve": [1.1, None, 1.2, 1.15, float("nan"), 1.25, 1.4]})
        >>> frame.select(ulcer_performance_ratio(pl.col("equity_curve"), periods_per_year=1).round(4)).item()
        nan
    """
    equity_curve = float64_expr(equity_curve)
    validate_periods_per_year(periods_per_year)
    validate_finite(risk_free_rate, "risk_free_rate")
    return (
        (cagr(equity_curve, periods_per_year=periods_per_year) - risk_free_rate) / ulcer_index(equity_curve)
    ).name.keep()
