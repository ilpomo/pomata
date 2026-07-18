"""
The metrics-family import side-effect aggregator: importing this imports every metrics declaration module, so
each ``suite_metrics(...)`` call runs and registers its declaration in ``registry_metrics``.

``tests.all_declarations`` imports this once so the collectible rung and registry modules see a populated
metrics registry before they parametrize over it. The declarations are gathered into ``METRIC_DECLARATIONS`` so
the imports are referenced (not dead), mirroring the public surface order of ``pomata.metrics.__all__``.
"""

from tests.metrics.adjusted_sharpe_ratio import ADJUSTED_SHARPE_RATIO
from tests.metrics.alpha import ALPHA
from tests.metrics.alpha_rolling import ALPHA_ROLLING
from tests.metrics.beta import BETA
from tests.metrics.beta_rolling import BETA_ROLLING
from tests.metrics.burke_ratio import BURKE_RATIO
from tests.metrics.cagr import CAGR
from tests.metrics.cagr_rolling import CAGR_ROLLING
from tests.metrics.calmar_ratio import CALMAR_RATIO
from tests.metrics.capture_downside_ratio import CAPTURE_DOWNSIDE_RATIO
from tests.metrics.capture_ratio import CAPTURE_RATIO
from tests.metrics.capture_upside_ratio import CAPTURE_UPSIDE_RATIO
from tests.metrics.common_sense_ratio import COMMON_SENSE_RATIO
from tests.metrics.conditional_drawdown_at_risk import CONDITIONAL_DRAWDOWN_AT_RISK
from tests.metrics.conditional_value_at_risk import CONDITIONAL_VALUE_AT_RISK
from tests.metrics.downside_deviation import DOWNSIDE_DEVIATION
from tests.metrics.downside_deviation_rolling import DOWNSIDE_DEVIATION_ROLLING
from tests.metrics.drawdown import DRAWDOWN
from tests.metrics.drawdown_rolling import DRAWDOWN_ROLLING
from tests.metrics.gain_to_pain_ratio import GAIN_TO_PAIN_RATIO
from tests.metrics.information_ratio import INFORMATION_RATIO
from tests.metrics.information_ratio_rolling import INFORMATION_RATIO_ROLLING
from tests.metrics.kelly_criterion import KELLY_CRITERION
from tests.metrics.kurtosis import KURTOSIS
from tests.metrics.kurtosis_rolling import KURTOSIS_ROLLING
from tests.metrics.max_drawdown import MAX_DRAWDOWN
from tests.metrics.max_drawdown_duration import MAX_DRAWDOWN_DURATION
from tests.metrics.modigliani_risk_adjusted_performance import MODIGLIANI_RISK_ADJUSTED_PERFORMANCE
from tests.metrics.omega_ratio import OMEGA_RATIO
from tests.metrics.omega_ratio_rolling import OMEGA_RATIO_ROLLING
from tests.metrics.pain_index import PAIN_INDEX
from tests.metrics.pain_ratio import PAIN_RATIO
from tests.metrics.payoff_ratio import PAYOFF_RATIO
from tests.metrics.probabilistic_sharpe_ratio import PROBABILISTIC_SHARPE_RATIO
from tests.metrics.profit_factor import PROFIT_FACTOR
from tests.metrics.recovery_ratio import RECOVERY_RATIO
from tests.metrics.risk_of_ruin import RISK_OF_RUIN
from tests.metrics.sharpe_ratio import SHARPE_RATIO
from tests.metrics.sharpe_ratio_rolling import SHARPE_RATIO_ROLLING
from tests.metrics.skewness import SKEWNESS
from tests.metrics.skewness_rolling import SKEWNESS_ROLLING
from tests.metrics.sortino_ratio import SORTINO_RATIO
from tests.metrics.sortino_ratio_rolling import SORTINO_RATIO_ROLLING
from tests.metrics.stability import STABILITY
from tests.metrics.sterling_ratio import STERLING_RATIO
from tests.metrics.tail_ratio import TAIL_RATIO
from tests.metrics.tail_ratio_rolling import TAIL_RATIO_ROLLING
from tests.metrics.total_return import TOTAL_RETURN
from tests.metrics.total_return_rolling import TOTAL_RETURN_ROLLING
from tests.metrics.treynor_ratio import TREYNOR_RATIO
from tests.metrics.treynor_ratio_rolling import TREYNOR_RATIO_ROLLING
from tests.metrics.ulcer_index import ULCER_INDEX
from tests.metrics.ulcer_performance_ratio import ULCER_PERFORMANCE_RATIO
from tests.metrics.value_at_risk import VALUE_AT_RISK
from tests.metrics.value_at_risk_modified import VALUE_AT_RISK_MODIFIED
from tests.metrics.value_at_risk_parametric import VALUE_AT_RISK_PARAMETRIC
from tests.metrics.value_at_risk_rolling import VALUE_AT_RISK_ROLLING
from tests.metrics.volatility import VOLATILITY
from tests.metrics.volatility_rolling import VOLATILITY_ROLLING
from tests.metrics.win_rate import WIN_RATE
from tests.support.declaration import Declaration

METRIC_DECLARATIONS: tuple[Declaration, ...] = (
    ADJUSTED_SHARPE_RATIO,
    ALPHA,
    ALPHA_ROLLING,
    BETA,
    BETA_ROLLING,
    BURKE_RATIO,
    CAGR,
    CAGR_ROLLING,
    CALMAR_RATIO,
    CAPTURE_DOWNSIDE_RATIO,
    CAPTURE_RATIO,
    CAPTURE_UPSIDE_RATIO,
    COMMON_SENSE_RATIO,
    CONDITIONAL_DRAWDOWN_AT_RISK,
    CONDITIONAL_VALUE_AT_RISK,
    DOWNSIDE_DEVIATION,
    DOWNSIDE_DEVIATION_ROLLING,
    DRAWDOWN,
    DRAWDOWN_ROLLING,
    GAIN_TO_PAIN_RATIO,
    INFORMATION_RATIO,
    INFORMATION_RATIO_ROLLING,
    KELLY_CRITERION,
    KURTOSIS,
    KURTOSIS_ROLLING,
    MAX_DRAWDOWN,
    MAX_DRAWDOWN_DURATION,
    MODIGLIANI_RISK_ADJUSTED_PERFORMANCE,
    OMEGA_RATIO,
    OMEGA_RATIO_ROLLING,
    PAIN_INDEX,
    PAIN_RATIO,
    PAYOFF_RATIO,
    PROBABILISTIC_SHARPE_RATIO,
    PROFIT_FACTOR,
    RECOVERY_RATIO,
    RISK_OF_RUIN,
    SHARPE_RATIO,
    SHARPE_RATIO_ROLLING,
    SKEWNESS,
    SKEWNESS_ROLLING,
    SORTINO_RATIO,
    SORTINO_RATIO_ROLLING,
    STABILITY,
    STERLING_RATIO,
    TAIL_RATIO,
    TAIL_RATIO_ROLLING,
    TOTAL_RETURN,
    TOTAL_RETURN_ROLLING,
    TREYNOR_RATIO,
    TREYNOR_RATIO_ROLLING,
    ULCER_INDEX,
    ULCER_PERFORMANCE_RATIO,
    VALUE_AT_RISK,
    VALUE_AT_RISK_MODIFIED,
    VALUE_AT_RISK_PARAMETRIC,
    VALUE_AT_RISK_ROLLING,
    VOLATILITY,
    VOLATILITY_ROLLING,
    WIN_RATE,
)
