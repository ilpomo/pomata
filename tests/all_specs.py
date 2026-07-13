"""
The aggregator: one explicit import per public function, the per-family tuples, and the surface guard.

A forgotten import is a red build. The guard runs at import (so any ``pytest tests`` collection enforces it): each
family's spec tuple must be in exact two-way correspondence with that family's public ``__all__`` — a stray spec
(a name the package does not export) fails as loudly as a gap (an exported name with no spec) — and the names must
be unique across the whole suite. The public surface itself is the single source of truth: adding a function to an
``__all__`` without landing its spec cannot collect, and a spec cannot outlive its function.
"""

from collections import Counter

from tests.indicators.absolute_price_oscillator import ABSOLUTE_PRICE_OSCILLATOR
from tests.indicators.accumulation_distribution import ACCUMULATION_DISTRIBUTION
from tests.indicators.accumulation_distribution_oscillator import ACCUMULATION_DISTRIBUTION_OSCILLATOR
from tests.indicators.adx import ADX
from tests.indicators.adxr import ADXR
from tests.indicators.aroon import AROON
from tests.indicators.aroon_oscillator import AROON_OSCILLATOR
from tests.indicators.atr import ATR
from tests.indicators.atr_normalized import ATR_NORMALIZED
from tests.indicators.awesome_oscillator import AWESOME_OSCILLATOR
from tests.indicators.balance_of_power import BALANCE_OF_POWER
from tests.indicators.bollinger_bands import BOLLINGER_BANDS
from tests.indicators.cci import CCI
from tests.indicators.chaikin_money_flow import CHAIKIN_MONEY_FLOW
from tests.indicators.chande_momentum_oscillator import CHANDE_MOMENTUM_OSCILLATOR
from tests.indicators.dema import DEMA
from tests.indicators.di_minus import DI_MINUS
from tests.indicators.di_plus import DI_PLUS
from tests.indicators.dm_minus import DM_MINUS
from tests.indicators.dm_plus import DM_PLUS
from tests.indicators.dominant_cycle_period import DOMINANT_CYCLE_PERIOD
from tests.indicators.dominant_cycle_phase import DOMINANT_CYCLE_PHASE
from tests.indicators.donchian_channels import DONCHIAN_CHANNELS
from tests.indicators.dx import DX
from tests.indicators.ema import EMA
from tests.indicators.fisher_transform import FISHER_TRANSFORM
from tests.indicators.hilbert_phasor import HILBERT_PHASOR
from tests.indicators.hilbert_trendline import HILBERT_TRENDLINE
from tests.indicators.hma import HMA
from tests.indicators.ichimoku import ICHIMOKU
from tests.indicators.kama import KAMA
from tests.indicators.keltner_channels import KELTNER_CHANNELS
from tests.indicators.linear_regression import LINEAR_REGRESSION
from tests.indicators.linear_regression_angle import LINEAR_REGRESSION_ANGLE
from tests.indicators.linear_regression_intercept import LINEAR_REGRESSION_INTERCEPT
from tests.indicators.linear_regression_slope import LINEAR_REGRESSION_SLOPE
from tests.indicators.macd import MACD
from tests.indicators.mama import MAMA
from tests.indicators.midpoint import MIDPOINT
from tests.indicators.midprice import MIDPRICE
from tests.indicators.mom import MOM
from tests.indicators.money_flow_index import MONEY_FLOW_INDEX
from tests.indicators.obv import OBV
from tests.indicators.parabolic_sar import PARABOLIC_SAR
from tests.indicators.percentage_price_oscillator import PERCENTAGE_PRICE_OSCILLATOR
from tests.indicators.price_average import PRICE_AVERAGE
from tests.indicators.price_median import PRICE_MEDIAN
from tests.indicators.price_typical import PRICE_TYPICAL
from tests.indicators.price_weighted_close import PRICE_WEIGHTED_CLOSE
from tests.indicators.rma import RMA
from tests.indicators.roc import ROC
from tests.indicators.rsi import RSI
from tests.indicators.rsi_stochastic import RSI_STOCHASTIC
from tests.indicators.sine_wave import SINE_WAVE
from tests.indicators.sma import SMA
from tests.indicators.standard_deviation_ewma import STANDARD_DEVIATION_EWMA
from tests.indicators.standard_deviation_rolling import STANDARD_DEVIATION_ROLLING
from tests.indicators.stochastic_fast import STOCHASTIC_FAST
from tests.indicators.stochastic_slow import STOCHASTIC_SLOW
from tests.indicators.supertrend import SUPERTREND
from tests.indicators.t3 import T3
from tests.indicators.tema import TEMA
from tests.indicators.time_series_forecast import TIME_SERIES_FORECAST
from tests.indicators.trend_mode import TREND_MODE
from tests.indicators.trima import TRIMA
from tests.indicators.trix import TRIX
from tests.indicators.true_range import TRUE_RANGE
from tests.indicators.ultimate_oscillator import ULTIMATE_OSCILLATOR
from tests.indicators.variance_ewma import VARIANCE_EWMA
from tests.indicators.variance_rolling import VARIANCE_ROLLING
from tests.indicators.vortex import VORTEX
from tests.indicators.vwap import VWAP
from tests.indicators.vwma import VWMA
from tests.indicators.williams_r import WILLIAMS_R
from tests.indicators.wma import WMA
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
from tests.pnl.cost_borrow import COST_BORROW
from tests.pnl.cost_fixed import COST_FIXED
from tests.pnl.cost_funding import COST_FUNDING
from tests.pnl.cost_notional import COST_NOTIONAL
from tests.pnl.cost_per_share import COST_PER_SHARE
from tests.pnl.cost_proportional import COST_PROPORTIONAL
from tests.pnl.cost_slippage import COST_SLIPPAGE
from tests.pnl.cumulative_pnl import CUMULATIVE_PNL
from tests.pnl.dividend import DIVIDEND
from tests.pnl.equity_curve import EQUITY_CURVE
from tests.pnl.pnl_gross import PNL_GROSS
from tests.pnl.pnl_gross_inverse import PNL_GROSS_INVERSE
from tests.pnl.pnl_net import PNL_NET
from tests.pnl.returns_gross import RETURNS_GROSS
from tests.pnl.returns_log import RETURNS_LOG
from tests.pnl.returns_net import RETURNS_NET
from tests.pnl.returns_simple import RETURNS_SIMPLE
from tests.pnl.turnover import TURNOVER
from tests.support.spec import Spec

import pomata.indicators
import pomata.metrics
import pomata.pnl

INDICATORS_SPECS: tuple[Spec, ...] = (
    ABSOLUTE_PRICE_OSCILLATOR,
    ACCUMULATION_DISTRIBUTION,
    ACCUMULATION_DISTRIBUTION_OSCILLATOR,
    ADX,
    ADXR,
    AROON,
    AROON_OSCILLATOR,
    ATR,
    ATR_NORMALIZED,
    AWESOME_OSCILLATOR,
    BALANCE_OF_POWER,
    BOLLINGER_BANDS,
    CCI,
    CHAIKIN_MONEY_FLOW,
    CHANDE_MOMENTUM_OSCILLATOR,
    DEMA,
    DI_MINUS,
    DI_PLUS,
    DM_MINUS,
    DM_PLUS,
    DOMINANT_CYCLE_PERIOD,
    DOMINANT_CYCLE_PHASE,
    DONCHIAN_CHANNELS,
    DX,
    EMA,
    FISHER_TRANSFORM,
    HILBERT_PHASOR,
    HILBERT_TRENDLINE,
    HMA,
    ICHIMOKU,
    KAMA,
    KELTNER_CHANNELS,
    LINEAR_REGRESSION,
    LINEAR_REGRESSION_ANGLE,
    LINEAR_REGRESSION_INTERCEPT,
    LINEAR_REGRESSION_SLOPE,
    MACD,
    MAMA,
    MIDPOINT,
    MIDPRICE,
    MOM,
    MONEY_FLOW_INDEX,
    OBV,
    PARABOLIC_SAR,
    PERCENTAGE_PRICE_OSCILLATOR,
    PRICE_AVERAGE,
    PRICE_MEDIAN,
    PRICE_TYPICAL,
    PRICE_WEIGHTED_CLOSE,
    RMA,
    ROC,
    RSI,
    RSI_STOCHASTIC,
    SINE_WAVE,
    SMA,
    STANDARD_DEVIATION_EWMA,
    STANDARD_DEVIATION_ROLLING,
    STOCHASTIC_FAST,
    STOCHASTIC_SLOW,
    SUPERTREND,
    T3,
    TEMA,
    TIME_SERIES_FORECAST,
    TREND_MODE,
    TRIMA,
    TRIX,
    TRUE_RANGE,
    ULTIMATE_OSCILLATOR,
    VARIANCE_EWMA,
    VARIANCE_ROLLING,
    VORTEX,
    VWAP,
    VWMA,
    WILLIAMS_R,
    WMA,
)
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
        public = set(_FAMILY_ALL[family])
        if declared != public:
            gaps = sorted(public - declared)
            strays = sorted(declared - public)
            msg = f"{family}: specs and __all__ disagree — missing specs {gaps}, stray specs {strays}"
            raise ValueError(msg)
        misfiled = sorted(spec.name for spec in specs if spec.family != family)
        if misfiled:
            msg = f"{family}: specs whose derived family is not {family}: {misfiled}"
            raise ValueError(msg)


_check_surface()
