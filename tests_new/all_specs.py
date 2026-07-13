"""
The aggregator: one explicit import per migrated function, the per-family tuples, and the surface guards.

A forgotten import is a red build. The guards run at import (so any ``pytest tests_new`` collection enforces them):
the per-family tuple must be in exact two-way correspondence with ``MIGRATED`` — a stray spec (one not listed) fails
as loudly as a gap (a listed name with no spec) — every migrated name must be in its family's public ``__all__``, and
the names must be unique. At cutover ``MIGRATED`` is replaced by the ``__all__`` tuples themselves and this becomes
the bijection guard of the whole suite.
"""

from collections import Counter

from tests_new.indicators.ichimoku import ICHIMOKU
from tests_new.indicators.mama import MAMA
from tests_new.metrics.adjusted_sharpe_ratio import ADJUSTED_SHARPE_RATIO
from tests_new.metrics.alpha import ALPHA
from tests_new.metrics.alpha_rolling import ALPHA_ROLLING
from tests_new.metrics.beta import BETA
from tests_new.metrics.beta_rolling import BETA_ROLLING
from tests_new.metrics.burke_ratio import BURKE_RATIO
from tests_new.metrics.cagr import CAGR
from tests_new.metrics.cagr_rolling import CAGR_ROLLING
from tests_new.metrics.calmar_ratio import CALMAR_RATIO
from tests_new.metrics.capture_downside_ratio import CAPTURE_DOWNSIDE_RATIO
from tests_new.metrics.capture_ratio import CAPTURE_RATIO
from tests_new.metrics.capture_upside_ratio import CAPTURE_UPSIDE_RATIO
from tests_new.metrics.common_sense_ratio import COMMON_SENSE_RATIO
from tests_new.metrics.conditional_drawdown_at_risk import CONDITIONAL_DRAWDOWN_AT_RISK
from tests_new.metrics.conditional_value_at_risk import CONDITIONAL_VALUE_AT_RISK
from tests_new.metrics.downside_deviation import DOWNSIDE_DEVIATION
from tests_new.metrics.downside_deviation_rolling import DOWNSIDE_DEVIATION_ROLLING
from tests_new.metrics.drawdown import DRAWDOWN
from tests_new.metrics.drawdown_rolling import DRAWDOWN_ROLLING
from tests_new.metrics.gain_to_pain_ratio import GAIN_TO_PAIN_RATIO
from tests_new.metrics.information_ratio import INFORMATION_RATIO
from tests_new.metrics.information_ratio_rolling import INFORMATION_RATIO_ROLLING
from tests_new.metrics.kelly_criterion import KELLY_CRITERION
from tests_new.metrics.kurtosis import KURTOSIS
from tests_new.metrics.kurtosis_rolling import KURTOSIS_ROLLING
from tests_new.metrics.max_drawdown import MAX_DRAWDOWN
from tests_new.metrics.max_drawdown_duration import MAX_DRAWDOWN_DURATION
from tests_new.metrics.modigliani_risk_adjusted_performance import MODIGLIANI_RISK_ADJUSTED_PERFORMANCE
from tests_new.metrics.omega_ratio import OMEGA_RATIO
from tests_new.metrics.omega_ratio_rolling import OMEGA_RATIO_ROLLING
from tests_new.metrics.pain_index import PAIN_INDEX
from tests_new.metrics.pain_ratio import PAIN_RATIO
from tests_new.metrics.payoff_ratio import PAYOFF_RATIO
from tests_new.metrics.probabilistic_sharpe_ratio import PROBABILISTIC_SHARPE_RATIO
from tests_new.metrics.profit_factor import PROFIT_FACTOR
from tests_new.metrics.recovery_ratio import RECOVERY_RATIO
from tests_new.metrics.risk_of_ruin import RISK_OF_RUIN
from tests_new.metrics.sharpe_ratio import SHARPE_RATIO
from tests_new.metrics.sharpe_ratio_rolling import SHARPE_RATIO_ROLLING
from tests_new.metrics.skewness import SKEWNESS
from tests_new.metrics.skewness_rolling import SKEWNESS_ROLLING
from tests_new.metrics.sortino_ratio import SORTINO_RATIO
from tests_new.metrics.sortino_ratio_rolling import SORTINO_RATIO_ROLLING
from tests_new.metrics.stability import STABILITY
from tests_new.metrics.sterling_ratio import STERLING_RATIO
from tests_new.metrics.tail_ratio import TAIL_RATIO
from tests_new.metrics.tail_ratio_rolling import TAIL_RATIO_ROLLING
from tests_new.metrics.total_return import TOTAL_RETURN
from tests_new.metrics.total_return_rolling import TOTAL_RETURN_ROLLING
from tests_new.metrics.treynor_ratio import TREYNOR_RATIO
from tests_new.metrics.treynor_ratio_rolling import TREYNOR_RATIO_ROLLING
from tests_new.metrics.ulcer_index import ULCER_INDEX
from tests_new.metrics.ulcer_performance_ratio import ULCER_PERFORMANCE_RATIO
from tests_new.metrics.value_at_risk import VALUE_AT_RISK
from tests_new.metrics.value_at_risk_modified import VALUE_AT_RISK_MODIFIED
from tests_new.metrics.value_at_risk_parametric import VALUE_AT_RISK_PARAMETRIC
from tests_new.metrics.value_at_risk_rolling import VALUE_AT_RISK_ROLLING
from tests_new.metrics.volatility import VOLATILITY
from tests_new.metrics.volatility_rolling import VOLATILITY_ROLLING
from tests_new.metrics.win_rate import WIN_RATE
from tests_new.pnl.cost_borrow import COST_BORROW
from tests_new.pnl.cost_fixed import COST_FIXED
from tests_new.pnl.cost_funding import COST_FUNDING
from tests_new.pnl.cost_notional import COST_NOTIONAL
from tests_new.pnl.cost_per_share import COST_PER_SHARE
from tests_new.pnl.cost_proportional import COST_PROPORTIONAL
from tests_new.pnl.cost_slippage import COST_SLIPPAGE
from tests_new.pnl.cumulative_pnl import CUMULATIVE_PNL
from tests_new.pnl.dividend import DIVIDEND
from tests_new.pnl.equity_curve import EQUITY_CURVE
from tests_new.pnl.pnl_gross import PNL_GROSS
from tests_new.pnl.pnl_gross_inverse import PNL_GROSS_INVERSE
from tests_new.pnl.pnl_net import PNL_NET
from tests_new.pnl.returns_gross import RETURNS_GROSS
from tests_new.pnl.returns_log import RETURNS_LOG
from tests_new.pnl.returns_net import RETURNS_NET
from tests_new.pnl.returns_simple import RETURNS_SIMPLE
from tests_new.pnl.turnover import TURNOVER
from tests_new.support.spec import Spec

import pomata.indicators
import pomata.metrics
import pomata.pnl

INDICATORS_SPECS: tuple[Spec, ...] = (ICHIMOKU, MAMA)
METRICS_SPECS: tuple[Spec, ...] = (
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
PNL_SPECS: tuple[Spec, ...] = (
    COST_BORROW,
    COST_FIXED,
    COST_FUNDING,
    COST_NOTIONAL,
    COST_PER_SHARE,
    COST_PROPORTIONAL,
    COST_SLIPPAGE,
    CUMULATIVE_PNL,
    DIVIDEND,
    EQUITY_CURVE,
    PNL_GROSS,
    PNL_GROSS_INVERSE,
    PNL_NET,
    RETURNS_GROSS,
    RETURNS_LOG,
    RETURNS_NET,
    RETURNS_SIMPLE,
    TURNOVER,
)
ALL_SPECS: tuple[Spec, ...] = (*INDICATORS_SPECS, *METRICS_SPECS, *PNL_SPECS)

# The functions whose spec has landed, per family; each family extends both a tuple above and the matching set here.
MIGRATED: dict[str, frozenset[str]] = {
    "indicators": frozenset({"ichimoku", "mama"}),
    "metrics": frozenset(
        {
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
        }
    ),
    "pnl": frozenset(
        {
            "cost_borrow",
            "cost_fixed",
            "cost_funding",
            "cost_notional",
            "cost_per_share",
            "cost_proportional",
            "cost_slippage",
            "cumulative_pnl",
            "dividend",
            "equity_curve",
            "pnl_gross",
            "pnl_gross_inverse",
            "pnl_net",
            "returns_gross",
            "returns_log",
            "returns_net",
            "returns_simple",
            "turnover",
        }
    ),
}

_FAMILY_ALL = {
    "indicators": pomata.indicators.__all__,
    "metrics": pomata.metrics.__all__,
    "pnl": pomata.pnl.__all__,
}
_FAMILY_SPECS = {"indicators": INDICATORS_SPECS, "metrics": METRICS_SPECS, "pnl": PNL_SPECS}


def _check_surface() -> None:
    """The two-way bijection, the public-name subset, and the uniqueness guard — born red, run at import."""
    duplicates = sorted(name for name, count in Counter(spec.name for spec in ALL_SPECS).items() if count > 1)
    if duplicates:
        msg = f"duplicate spec names: {duplicates}"
        raise ValueError(msg)
    for family, specs in _FAMILY_SPECS.items():
        declared = {spec.name for spec in specs}
        if declared != set(MIGRATED[family]):
            msg = f"{family}: declared specs {sorted(declared)} disagree with MIGRATED {sorted(MIGRATED[family])}"
            raise ValueError(msg)
        stray = MIGRATED[family] - set(_FAMILY_ALL[family])
        if stray:
            msg = f"{family}: migrated names outside the public __all__: {sorted(stray)}"
            raise ValueError(msg)
        misfiled = sorted(spec.name for spec in specs if spec.family != family)
        if misfiled:
            msg = f"{family}: specs whose derived family is not {family}: {misfiled}"
            raise ValueError(msg)


_check_surface()
