"""
The indicators-family import side-effect aggregator: importing this imports every indicators declaration
module, so each ``suite_indicators(...)`` call runs and registers its declaration in ``registry_indicators``.

``tests_new.all_declarations`` imports this once so the collectible rung and registry modules see a populated
indicators registry before they parametrize over it. The declarations are gathered into ``INDICATOR_DECLARATIONS``
so the imports are referenced (not dead), mirroring the public surface order of ``pomata.indicators.__all__``.
"""

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
from tests_new.support.declaration import Declaration

INDICATOR_DECLARATIONS: tuple[Declaration, ...] = (
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
