"""
The aggregator: one explicit import per migrated function, the per-family tuples, and the surface guards.

A forgotten import is a red build. The guards run at import (so any ``pytest tests_new`` collection enforces them):
the per-family tuple must be in exact two-way correspondence with ``MIGRATED`` — a stray spec (one not listed) fails
as loudly as a gap (a listed name with no spec) — every migrated name must be in its family's public ``__all__``, and
the names must be unique. At cutover ``MIGRATED`` is replaced by the ``__all__`` tuples themselves and this becomes
the bijection guard of the whole suite.
"""

from collections import Counter

from tests_new.indicators.absolute_price_oscillator import ABSOLUTE_PRICE_OSCILLATOR
from tests_new.indicators.accumulation_distribution import ACCUMULATION_DISTRIBUTION
from tests_new.indicators.accumulation_distribution_oscillator import ACCUMULATION_DISTRIBUTION_OSCILLATOR
from tests_new.indicators.adx import ADX
from tests_new.indicators.adxr import ADXR
from tests_new.indicators.aroon import AROON
from tests_new.indicators.aroon_oscillator import AROON_OSCILLATOR
from tests_new.indicators.atr import ATR
from tests_new.indicators.atr_normalized import ATR_NORMALIZED
from tests_new.indicators.awesome_oscillator import AWESOME_OSCILLATOR
from tests_new.indicators.balance_of_power import BALANCE_OF_POWER
from tests_new.indicators.bollinger_bands import BOLLINGER_BANDS
from tests_new.indicators.cci import CCI
from tests_new.indicators.chaikin_money_flow import CHAIKIN_MONEY_FLOW
from tests_new.indicators.chande_momentum_oscillator import CHANDE_MOMENTUM_OSCILLATOR
from tests_new.indicators.dema import DEMA
from tests_new.indicators.di_minus import DI_MINUS
from tests_new.indicators.di_plus import DI_PLUS
from tests_new.indicators.dm_minus import DM_MINUS
from tests_new.indicators.dm_plus import DM_PLUS
from tests_new.indicators.dominant_cycle_period import DOMINANT_CYCLE_PERIOD
from tests_new.indicators.dominant_cycle_phase import DOMINANT_CYCLE_PHASE
from tests_new.indicators.donchian_channels import DONCHIAN_CHANNELS
from tests_new.indicators.dx import DX
from tests_new.indicators.ema import EMA
from tests_new.indicators.fisher_transform import FISHER_TRANSFORM
from tests_new.indicators.hilbert_phasor import HILBERT_PHASOR
from tests_new.indicators.hilbert_trendline import HILBERT_TRENDLINE
from tests_new.indicators.hma import HMA
from tests_new.indicators.ichimoku import ICHIMOKU
from tests_new.indicators.kama import KAMA
from tests_new.indicators.keltner_channels import KELTNER_CHANNELS
from tests_new.indicators.linear_regression import LINEAR_REGRESSION
from tests_new.indicators.linear_regression_angle import LINEAR_REGRESSION_ANGLE
from tests_new.indicators.linear_regression_intercept import LINEAR_REGRESSION_INTERCEPT
from tests_new.indicators.linear_regression_slope import LINEAR_REGRESSION_SLOPE
from tests_new.indicators.macd import MACD
from tests_new.indicators.mama import MAMA
from tests_new.indicators.midpoint import MIDPOINT
from tests_new.indicators.midprice import MIDPRICE
from tests_new.indicators.mom import MOM
from tests_new.indicators.money_flow_index import MONEY_FLOW_INDEX
from tests_new.indicators.obv import OBV
from tests_new.indicators.parabolic_sar import PARABOLIC_SAR
from tests_new.indicators.percentage_price_oscillator import PERCENTAGE_PRICE_OSCILLATOR
from tests_new.indicators.price_average import PRICE_AVERAGE
from tests_new.indicators.price_median import PRICE_MEDIAN
from tests_new.indicators.price_typical import PRICE_TYPICAL
from tests_new.indicators.price_weighted_close import PRICE_WEIGHTED_CLOSE
from tests_new.indicators.rma import RMA
from tests_new.indicators.roc import ROC
from tests_new.indicators.rsi import RSI
from tests_new.indicators.rsi_stochastic import RSI_STOCHASTIC
from tests_new.indicators.sine_wave import SINE_WAVE
from tests_new.indicators.sma import SMA
from tests_new.indicators.standard_deviation_ewma import STANDARD_DEVIATION_EWMA
from tests_new.indicators.standard_deviation_rolling import STANDARD_DEVIATION_ROLLING
from tests_new.indicators.stochastic_fast import STOCHASTIC_FAST
from tests_new.indicators.stochastic_slow import STOCHASTIC_SLOW
from tests_new.indicators.supertrend import SUPERTREND
from tests_new.indicators.t3 import T3
from tests_new.indicators.tema import TEMA
from tests_new.indicators.time_series_forecast import TIME_SERIES_FORECAST
from tests_new.indicators.trend_mode import TREND_MODE
from tests_new.indicators.trima import TRIMA
from tests_new.indicators.trix import TRIX
from tests_new.indicators.true_range import TRUE_RANGE
from tests_new.indicators.ultimate_oscillator import ULTIMATE_OSCILLATOR
from tests_new.indicators.variance_ewma import VARIANCE_EWMA
from tests_new.indicators.variance_rolling import VARIANCE_ROLLING
from tests_new.indicators.vortex import VORTEX
from tests_new.indicators.vwap import VWAP
from tests_new.indicators.vwma import VWMA
from tests_new.indicators.williams_r import WILLIAMS_R
from tests_new.indicators.wma import WMA
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

# The functions whose spec has landed, per family; each family extends both a tuple above and the matching set here.
MIGRATED: dict[str, frozenset[str]] = {
    "indicators": frozenset(
        {
            "absolute_price_oscillator",
            "accumulation_distribution",
            "accumulation_distribution_oscillator",
            "adx",
            "adxr",
            "aroon",
            "aroon_oscillator",
            "atr",
            "atr_normalized",
            "awesome_oscillator",
            "balance_of_power",
            "bollinger_bands",
            "cci",
            "chaikin_money_flow",
            "chande_momentum_oscillator",
            "dema",
            "di_minus",
            "di_plus",
            "dm_minus",
            "dm_plus",
            "dominant_cycle_period",
            "dominant_cycle_phase",
            "donchian_channels",
            "dx",
            "ema",
            "fisher_transform",
            "hilbert_phasor",
            "hilbert_trendline",
            "hma",
            "ichimoku",
            "kama",
            "keltner_channels",
            "linear_regression",
            "linear_regression_angle",
            "linear_regression_intercept",
            "linear_regression_slope",
            "macd",
            "mama",
            "midpoint",
            "midprice",
            "mom",
            "money_flow_index",
            "obv",
            "parabolic_sar",
            "percentage_price_oscillator",
            "price_average",
            "price_median",
            "price_typical",
            "price_weighted_close",
            "rma",
            "roc",
            "rsi",
            "rsi_stochastic",
            "sine_wave",
            "sma",
            "standard_deviation_ewma",
            "standard_deviation_rolling",
            "stochastic_fast",
            "stochastic_slow",
            "supertrend",
            "t3",
            "tema",
            "time_series_forecast",
            "trend_mode",
            "trima",
            "trix",
            "true_range",
            "ultimate_oscillator",
            "variance_ewma",
            "variance_rolling",
            "vortex",
            "vwap",
            "vwma",
            "williams_r",
            "wma",
        }
    ),
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
