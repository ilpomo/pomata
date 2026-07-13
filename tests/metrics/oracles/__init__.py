"""
Naive reference oracles for the metrics family.

Each function recomputes one metric from scratch in plain Python from its canonical definition, sharing no code with the
Polars implementation it certifies, so agreement is evidence of correctness rather than coincidence.
"""

from tests.metrics.oracles.adjusted_sharpe_ratio import adjusted_sharpe_ratio_reference
from tests.metrics.oracles.alpha import alpha_reference
from tests.metrics.oracles.alpha_rolling import alpha_rolling_reference
from tests.metrics.oracles.beta import beta_reference
from tests.metrics.oracles.beta_rolling import beta_rolling_reference
from tests.metrics.oracles.burke_ratio import burke_ratio_reference
from tests.metrics.oracles.cagr import cagr_reference
from tests.metrics.oracles.cagr_rolling import cagr_rolling_reference
from tests.metrics.oracles.calmar_ratio import calmar_ratio_reference
from tests.metrics.oracles.capture_downside_ratio import capture_downside_ratio_reference
from tests.metrics.oracles.capture_ratio import capture_ratio_reference
from tests.metrics.oracles.capture_upside_ratio import capture_upside_ratio_reference
from tests.metrics.oracles.common_sense_ratio import common_sense_ratio_reference
from tests.metrics.oracles.conditional_drawdown_at_risk import conditional_drawdown_at_risk_reference
from tests.metrics.oracles.conditional_value_at_risk import conditional_value_at_risk_reference
from tests.metrics.oracles.downside_deviation import downside_deviation_reference
from tests.metrics.oracles.downside_deviation_rolling import downside_deviation_rolling_reference
from tests.metrics.oracles.drawdown import drawdown_reference
from tests.metrics.oracles.drawdown_rolling import drawdown_rolling_reference
from tests.metrics.oracles.gain_to_pain_ratio import gain_to_pain_ratio_reference
from tests.metrics.oracles.information_ratio import information_ratio_reference
from tests.metrics.oracles.information_ratio_rolling import information_ratio_rolling_reference
from tests.metrics.oracles.kelly_criterion import kelly_criterion_reference
from tests.metrics.oracles.kurtosis import kurtosis_reference
from tests.metrics.oracles.kurtosis_rolling import kurtosis_rolling_reference
from tests.metrics.oracles.max_drawdown import max_drawdown_reference
from tests.metrics.oracles.max_drawdown_duration import max_drawdown_duration_reference
from tests.metrics.oracles.modigliani_risk_adjusted_performance import (
    modigliani_risk_adjusted_performance_reference,
)
from tests.metrics.oracles.omega_ratio import omega_ratio_reference
from tests.metrics.oracles.omega_ratio_rolling import omega_ratio_rolling_reference
from tests.metrics.oracles.pain_index import pain_index_reference
from tests.metrics.oracles.pain_ratio import pain_ratio_reference
from tests.metrics.oracles.payoff_ratio import payoff_ratio_reference
from tests.metrics.oracles.probabilistic_sharpe_ratio import probabilistic_sharpe_ratio_reference
from tests.metrics.oracles.profit_factor import profit_factor_reference
from tests.metrics.oracles.recovery_ratio import recovery_ratio_reference
from tests.metrics.oracles.risk_of_ruin import risk_of_ruin_reference
from tests.metrics.oracles.sharpe_ratio import sharpe_ratio_reference
from tests.metrics.oracles.sharpe_ratio_rolling import sharpe_ratio_rolling_reference
from tests.metrics.oracles.skewness import skewness_reference
from tests.metrics.oracles.skewness_rolling import skewness_rolling_reference
from tests.metrics.oracles.sortino_ratio import sortino_ratio_reference
from tests.metrics.oracles.sortino_ratio_rolling import sortino_ratio_rolling_reference
from tests.metrics.oracles.stability import stability_reference
from tests.metrics.oracles.sterling_ratio import sterling_ratio_reference
from tests.metrics.oracles.tail_ratio import tail_ratio_reference
from tests.metrics.oracles.tail_ratio_rolling import tail_ratio_rolling_reference
from tests.metrics.oracles.total_return import total_return_reference
from tests.metrics.oracles.total_return_rolling import total_return_rolling_reference
from tests.metrics.oracles.treynor_ratio import treynor_ratio_reference
from tests.metrics.oracles.treynor_ratio_rolling import treynor_ratio_rolling_reference
from tests.metrics.oracles.ulcer_index import ulcer_index_reference
from tests.metrics.oracles.ulcer_performance_ratio import ulcer_performance_ratio_reference
from tests.metrics.oracles.value_at_risk import value_at_risk_reference
from tests.metrics.oracles.value_at_risk_modified import value_at_risk_modified_reference
from tests.metrics.oracles.value_at_risk_parametric import value_at_risk_parametric_reference
from tests.metrics.oracles.value_at_risk_rolling import value_at_risk_rolling_reference
from tests.metrics.oracles.volatility import volatility_reference
from tests.metrics.oracles.volatility_rolling import volatility_rolling_reference
from tests.metrics.oracles.win_rate import win_rate_reference

__all__ = (
    "adjusted_sharpe_ratio_reference",
    "alpha_reference",
    "alpha_rolling_reference",
    "beta_reference",
    "beta_rolling_reference",
    "burke_ratio_reference",
    "cagr_reference",
    "cagr_rolling_reference",
    "calmar_ratio_reference",
    "capture_downside_ratio_reference",
    "capture_ratio_reference",
    "capture_upside_ratio_reference",
    "common_sense_ratio_reference",
    "conditional_drawdown_at_risk_reference",
    "conditional_value_at_risk_reference",
    "downside_deviation_reference",
    "downside_deviation_rolling_reference",
    "drawdown_reference",
    "drawdown_rolling_reference",
    "gain_to_pain_ratio_reference",
    "information_ratio_reference",
    "information_ratio_rolling_reference",
    "kelly_criterion_reference",
    "kurtosis_reference",
    "kurtosis_rolling_reference",
    "max_drawdown_duration_reference",
    "max_drawdown_reference",
    "modigliani_risk_adjusted_performance_reference",
    "omega_ratio_reference",
    "omega_ratio_rolling_reference",
    "pain_index_reference",
    "pain_ratio_reference",
    "payoff_ratio_reference",
    "probabilistic_sharpe_ratio_reference",
    "profit_factor_reference",
    "recovery_ratio_reference",
    "risk_of_ruin_reference",
    "sharpe_ratio_reference",
    "sharpe_ratio_rolling_reference",
    "skewness_reference",
    "skewness_rolling_reference",
    "sortino_ratio_reference",
    "sortino_ratio_rolling_reference",
    "stability_reference",
    "sterling_ratio_reference",
    "tail_ratio_reference",
    "tail_ratio_rolling_reference",
    "total_return_reference",
    "total_return_rolling_reference",
    "treynor_ratio_reference",
    "treynor_ratio_rolling_reference",
    "ulcer_index_reference",
    "ulcer_performance_ratio_reference",
    "value_at_risk_modified_reference",
    "value_at_risk_parametric_reference",
    "value_at_risk_reference",
    "value_at_risk_rolling_reference",
    "volatility_reference",
    "volatility_rolling_reference",
    "win_rate_reference",
)
