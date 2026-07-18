"""
Naive reference implementations — the correctness oracles, one module per indicator, each renamed
reference_{indicator} so the declaration binding guard finds it by name. For the great majority — anything
expressible as composed Polars expressions — the oracle shares no code with the implementation, so agreement is
evidence of correctness rather than coincidence. The irreducibly-sequential kernels (parabolic SAR, KAMA, the
Hilbert-transform cycle cluster) are the exception: a one-shape path-dependent recurrence has essentially one
transcription, so those oracles necessarily mirror the implementation and their real correctness evidence is the
hand-derived golden masters and the TA-Lib differential, not the oracle agreement. This package re-exports them
flat, mirroring pomata.indicators: from tests.indicators.oracles import reference_sma.
"""

from tests.indicators.oracles.absolute_price_oscillator import reference_absolute_price_oscillator
from tests.indicators.oracles.accumulation_distribution import reference_accumulation_distribution
from tests.indicators.oracles.accumulation_distribution_oscillator import (
    reference_accumulation_distribution_oscillator,
)
from tests.indicators.oracles.adx import reference_adx
from tests.indicators.oracles.adxr import reference_adxr
from tests.indicators.oracles.aroon import reference_aroon
from tests.indicators.oracles.aroon_oscillator import reference_aroon_oscillator
from tests.indicators.oracles.atr import reference_atr
from tests.indicators.oracles.atr_normalized import reference_atr_normalized
from tests.indicators.oracles.awesome_oscillator import reference_awesome_oscillator
from tests.indicators.oracles.balance_of_power import reference_balance_of_power
from tests.indicators.oracles.bollinger_bands import reference_bollinger_bands
from tests.indicators.oracles.cci import reference_cci
from tests.indicators.oracles.chaikin_money_flow import reference_chaikin_money_flow
from tests.indicators.oracles.chande_momentum_oscillator import reference_chande_momentum_oscillator
from tests.indicators.oracles.cycle import (
    reference_dominant_cycle_period,
    reference_dominant_cycle_phase,
    reference_hilbert_phasor,
    reference_hilbert_trendline,
    reference_mama,
    reference_sine_wave,
    reference_trend_mode,
)
from tests.indicators.oracles.dema import reference_dema
from tests.indicators.oracles.di_minus import reference_di_minus
from tests.indicators.oracles.di_plus import reference_di_plus
from tests.indicators.oracles.dm_minus import reference_dm_minus
from tests.indicators.oracles.dm_plus import reference_dm_plus
from tests.indicators.oracles.donchian_channels import reference_donchian_channels
from tests.indicators.oracles.dx import reference_dx
from tests.indicators.oracles.ema import reference_ema
from tests.indicators.oracles.fisher_transform import reference_fisher_transform
from tests.indicators.oracles.hma import reference_hma
from tests.indicators.oracles.ichimoku import reference_ichimoku
from tests.indicators.oracles.kama import reference_kama
from tests.indicators.oracles.keltner_channels import reference_keltner_channels
from tests.indicators.oracles.linear_regression import reference_linear_regression
from tests.indicators.oracles.linear_regression_angle import reference_linear_regression_angle
from tests.indicators.oracles.linear_regression_intercept import reference_linear_regression_intercept
from tests.indicators.oracles.linear_regression_slope import reference_linear_regression_slope
from tests.indicators.oracles.macd import reference_macd
from tests.indicators.oracles.midpoint import reference_midpoint
from tests.indicators.oracles.midprice import reference_midprice
from tests.indicators.oracles.mom import reference_mom
from tests.indicators.oracles.money_flow_index import reference_money_flow_index
from tests.indicators.oracles.obv import reference_obv
from tests.indicators.oracles.parabolic_sar import reference_parabolic_sar
from tests.indicators.oracles.percentage_price_oscillator import reference_percentage_price_oscillator
from tests.indicators.oracles.price_average import reference_price_average
from tests.indicators.oracles.price_median import reference_price_median
from tests.indicators.oracles.price_typical import reference_price_typical
from tests.indicators.oracles.price_weighted_close import reference_price_weighted_close
from tests.indicators.oracles.rma import reference_rma
from tests.indicators.oracles.roc import reference_roc
from tests.indicators.oracles.rsi import reference_rsi
from tests.indicators.oracles.rsi_stochastic import reference_rsi_stochastic
from tests.indicators.oracles.sma import reference_sma
from tests.indicators.oracles.standard_deviation_ewma import reference_standard_deviation_ewma
from tests.indicators.oracles.standard_deviation_rolling import reference_standard_deviation_rolling
from tests.indicators.oracles.stochastic_fast import reference_stochastic_fast
from tests.indicators.oracles.stochastic_slow import reference_stochastic_slow
from tests.indicators.oracles.supertrend import reference_supertrend
from tests.indicators.oracles.t3 import reference_t3
from tests.indicators.oracles.tema import reference_tema
from tests.indicators.oracles.time_series_forecast import reference_time_series_forecast
from tests.indicators.oracles.trima import reference_trima
from tests.indicators.oracles.trix import reference_trix
from tests.indicators.oracles.true_range import reference_true_range
from tests.indicators.oracles.ultimate_oscillator import reference_ultimate_oscillator
from tests.indicators.oracles.variance_ewma import reference_variance_ewma
from tests.indicators.oracles.variance_rolling import reference_variance_rolling
from tests.indicators.oracles.vortex import reference_vortex
from tests.indicators.oracles.vwap import reference_vwap
from tests.indicators.oracles.vwma import reference_vwma
from tests.indicators.oracles.williams_r import reference_williams_r
from tests.indicators.oracles.wma import reference_wma

__all__ = (
    "reference_absolute_price_oscillator",
    "reference_accumulation_distribution",
    "reference_accumulation_distribution_oscillator",
    "reference_adx",
    "reference_adxr",
    "reference_aroon",
    "reference_aroon_oscillator",
    "reference_atr",
    "reference_atr_normalized",
    "reference_awesome_oscillator",
    "reference_balance_of_power",
    "reference_bollinger_bands",
    "reference_cci",
    "reference_chaikin_money_flow",
    "reference_chande_momentum_oscillator",
    "reference_dema",
    "reference_di_minus",
    "reference_di_plus",
    "reference_dm_minus",
    "reference_dm_plus",
    "reference_dominant_cycle_period",
    "reference_dominant_cycle_phase",
    "reference_donchian_channels",
    "reference_dx",
    "reference_ema",
    "reference_fisher_transform",
    "reference_hilbert_phasor",
    "reference_hilbert_trendline",
    "reference_hma",
    "reference_ichimoku",
    "reference_kama",
    "reference_keltner_channels",
    "reference_linear_regression",
    "reference_linear_regression_angle",
    "reference_linear_regression_intercept",
    "reference_linear_regression_slope",
    "reference_macd",
    "reference_mama",
    "reference_midpoint",
    "reference_midprice",
    "reference_mom",
    "reference_money_flow_index",
    "reference_obv",
    "reference_parabolic_sar",
    "reference_percentage_price_oscillator",
    "reference_price_average",
    "reference_price_median",
    "reference_price_typical",
    "reference_price_weighted_close",
    "reference_rma",
    "reference_roc",
    "reference_rsi",
    "reference_rsi_stochastic",
    "reference_sine_wave",
    "reference_sma",
    "reference_standard_deviation_ewma",
    "reference_standard_deviation_rolling",
    "reference_stochastic_fast",
    "reference_stochastic_slow",
    "reference_supertrend",
    "reference_t3",
    "reference_tema",
    "reference_time_series_forecast",
    "reference_trend_mode",
    "reference_trima",
    "reference_trix",
    "reference_true_range",
    "reference_ultimate_oscillator",
    "reference_variance_ewma",
    "reference_variance_rolling",
    "reference_vortex",
    "reference_vwap",
    "reference_vwma",
    "reference_williams_r",
    "reference_wma",
)
