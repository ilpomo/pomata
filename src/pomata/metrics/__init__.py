"""
Performance & risk metrics — composable, Polars-native, and fed directly by the ``pnl`` family: each is a free-standing
``pl.Expr`` factory that turns a return or equity series into a performance or risk figure, so you never leave the
toolkit or hand-roll a metric (mis-handling ``null`` / ``NaN`` / a short sample).

Each metric reads exactly one of the two series the ``pnl`` family emits; pick by what the metric measures:

- **Return-based** — pass a per-bar net return series (e.g. from :func:`pomata.pnl.returns_net`): the dispersion, shape,
  tail, and return-based risk-adjusted-ratio metrics live here (e.g. :func:`volatility`, :func:`sharpe_ratio`).
- **Equity-based** — pass a compounded growth-factor series (e.g. from :func:`pomata.pnl.equity_curve`): the cumulative
  return, drawdown, and equity-based ratio metrics live here (e.g. :func:`max_drawdown`, :func:`calmar_ratio`).

Most metrics **reduce** a series to a single value: one value in ``select``, and one value per group when wrapped in
``.over(...)`` for a multi-series panel; the running ``drawdown`` is the series-valued exception. Every metric with a
native-fast windowed form also ships a ``*_rolling`` twin (e.g. :func:`sharpe_ratio_rolling`, :func:`beta_rolling`) that
takes a positional ``window`` and is **series-valued** — one value per trailing window, with ``window - 1`` leading-null
warm-up. Annualized metrics take a required keyword-only ``periods_per_year`` (canonically ``252`` for daily) — never
silently assumed. Every function is a free-standing ``pl.Expr`` factory: compose it in ``select`` / ``with_columns``,
eager or lazy, on a single series or a long panel via ``.over(...)``. Source is organized into theme modules for
maintainability; this package re-exports a flat public API.
"""

from pomata.metrics.drawdown import (
    conditional_drawdown_at_risk,
    drawdown,
    drawdown_rolling,
    max_drawdown,
    max_drawdown_duration,
    pain_index,
    ulcer_index,
)
from pomata.metrics.performance import cagr, cagr_rolling, stability, total_return, total_return_rolling
from pomata.metrics.ratio import (
    adjusted_sharpe_ratio,
    burke_ratio,
    calmar_ratio,
    common_sense_ratio,
    gain_to_pain_ratio,
    omega_ratio,
    omega_ratio_rolling,
    pain_ratio,
    probabilistic_sharpe_ratio,
    recovery_ratio,
    sharpe_ratio,
    sharpe_ratio_rolling,
    sortino_ratio,
    sortino_ratio_rolling,
    sterling_ratio,
    ulcer_performance_ratio,
)
from pomata.metrics.relative import (
    alpha,
    alpha_rolling,
    beta,
    beta_rolling,
    capture_downside_ratio,
    capture_ratio,
    capture_upside_ratio,
    information_ratio,
    information_ratio_rolling,
    modigliani_risk_adjusted_performance,
    treynor_ratio,
    treynor_ratio_rolling,
)
from pomata.metrics.risk import (
    conditional_value_at_risk,
    downside_deviation,
    downside_deviation_rolling,
    kelly_criterion,
    kurtosis,
    kurtosis_rolling,
    payoff_ratio,
    profit_factor,
    risk_of_ruin,
    skewness,
    skewness_rolling,
    tail_ratio,
    tail_ratio_rolling,
    value_at_risk,
    value_at_risk_modified,
    value_at_risk_parametric,
    value_at_risk_rolling,
    volatility,
    volatility_rolling,
    win_rate,
)

__all__ = (
    "adjusted_sharpe_ratio",
    "alpha",
    "alpha_rolling",
    "beta",
    "beta_rolling",
    "burke_ratio",
    "cagr",
    "cagr_rolling",
    "calmar_ratio",
    "capture_downside_ratio",
    "capture_ratio",
    "capture_upside_ratio",
    "common_sense_ratio",
    "conditional_drawdown_at_risk",
    "conditional_value_at_risk",
    "downside_deviation",
    "downside_deviation_rolling",
    "drawdown",
    "drawdown_rolling",
    "gain_to_pain_ratio",
    "information_ratio",
    "information_ratio_rolling",
    "kelly_criterion",
    "kurtosis",
    "kurtosis_rolling",
    "max_drawdown",
    "max_drawdown_duration",
    "modigliani_risk_adjusted_performance",
    "omega_ratio",
    "omega_ratio_rolling",
    "pain_index",
    "pain_ratio",
    "payoff_ratio",
    "probabilistic_sharpe_ratio",
    "profit_factor",
    "recovery_ratio",
    "risk_of_ruin",
    "sharpe_ratio",
    "sharpe_ratio_rolling",
    "skewness",
    "skewness_rolling",
    "sortino_ratio",
    "sortino_ratio_rolling",
    "stability",
    "sterling_ratio",
    "tail_ratio",
    "tail_ratio_rolling",
    "total_return",
    "total_return_rolling",
    "treynor_ratio",
    "treynor_ratio_rolling",
    "ulcer_index",
    "ulcer_performance_ratio",
    "value_at_risk",
    "value_at_risk_modified",
    "value_at_risk_parametric",
    "value_at_risk_rolling",
    "volatility",
    "volatility_rolling",
    "win_rate",
)
