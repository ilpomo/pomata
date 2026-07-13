"""
Naive reference implementations — the correctness oracles, one module per indicator.

Each function recomputes one indicator from scratch in plain Python. For the great majority — anything expressible as
composed Polars expressions — the oracle shares no code with the implementation, so agreement is evidence of correctness
rather than coincidence. The irreducibly-sequential kernels (parabolic SAR, KAMA, the Hilbert-transform cycle cluster)
are the exception: a one-shape path-dependent recurrence has essentially one transcription, so those oracles necessarily
mirror the implementation's structure — their real correctness evidence is the hand-derived golden masters, not the
oracle agreement (see ``CORRECTNESS.md``). They target the semantics of the project's Polars floor (declared in
``pyproject.toml``); each docstring states the definition, the subtle points the reimplementation must reproduce, and
its null / NaN / degeneracy contract. This package re-exports them flat, mirroring
``pomata.indicators``: ``from tests_new.indicators.oracles import sma_reference``.
"""

from tests_new.indicators.oracles.absolute_price_oscillator import absolute_price_oscillator_reference
from tests_new.indicators.oracles.accumulation_distribution import accumulation_distribution_reference
from tests_new.indicators.oracles.accumulation_distribution_oscillator import (
    accumulation_distribution_oscillator_reference,
)
from tests_new.indicators.oracles.adx import adx_reference
from tests_new.indicators.oracles.adxr import adxr_reference
from tests_new.indicators.oracles.aroon import aroon_reference
from tests_new.indicators.oracles.aroon_oscillator import aroon_oscillator_reference
from tests_new.indicators.oracles.atr import atr_reference
from tests_new.indicators.oracles.atr_normalized import atr_normalized_reference
from tests_new.indicators.oracles.awesome_oscillator import awesome_oscillator_reference
from tests_new.indicators.oracles.balance_of_power import balance_of_power_reference
from tests_new.indicators.oracles.bollinger_bands import bollinger_bands_reference
from tests_new.indicators.oracles.cci import cci_reference
from tests_new.indicators.oracles.chaikin_money_flow import chaikin_money_flow_reference
from tests_new.indicators.oracles.chande_momentum_oscillator import chande_momentum_oscillator_reference
from tests_new.indicators.oracles.cycle import (
    dominant_cycle_period_reference,
    dominant_cycle_phase_reference,
    hilbert_phasor_reference,
    hilbert_trendline_reference,
    mama_reference,
    sine_wave_reference,
    trend_mode_reference,
)
from tests_new.indicators.oracles.dema import dema_reference
from tests_new.indicators.oracles.di_minus import di_minus_reference
from tests_new.indicators.oracles.di_plus import di_plus_reference
from tests_new.indicators.oracles.dm_minus import dm_minus_reference
from tests_new.indicators.oracles.dm_plus import dm_plus_reference
from tests_new.indicators.oracles.donchian_channels import donchian_channels_reference
from tests_new.indicators.oracles.dx import dx_reference
from tests_new.indicators.oracles.ema import ema_reference
from tests_new.indicators.oracles.fisher_transform import fisher_transform_reference
from tests_new.indicators.oracles.hma import hma_reference
from tests_new.indicators.oracles.ichimoku import ichimoku_reference
from tests_new.indicators.oracles.kama import kama_reference
from tests_new.indicators.oracles.keltner_channels import keltner_channels_reference
from tests_new.indicators.oracles.linear_regression import linear_regression_reference
from tests_new.indicators.oracles.linear_regression_angle import linear_regression_angle_reference
from tests_new.indicators.oracles.linear_regression_intercept import linear_regression_intercept_reference
from tests_new.indicators.oracles.linear_regression_slope import linear_regression_slope_reference
from tests_new.indicators.oracles.macd import macd_reference
from tests_new.indicators.oracles.midpoint import midpoint_reference
from tests_new.indicators.oracles.midprice import midprice_reference
from tests_new.indicators.oracles.mom import mom_reference
from tests_new.indicators.oracles.money_flow_index import money_flow_index_reference
from tests_new.indicators.oracles.obv import obv_reference
from tests_new.indicators.oracles.parabolic_sar import parabolic_sar_reference
from tests_new.indicators.oracles.percentage_price_oscillator import percentage_price_oscillator_reference
from tests_new.indicators.oracles.price_average import price_average_reference
from tests_new.indicators.oracles.price_median import price_median_reference
from tests_new.indicators.oracles.price_typical import price_typical_reference
from tests_new.indicators.oracles.price_weighted_close import price_weighted_close_reference
from tests_new.indicators.oracles.rma import rma_reference
from tests_new.indicators.oracles.roc import roc_reference
from tests_new.indicators.oracles.rsi import rsi_reference
from tests_new.indicators.oracles.rsi_stochastic import rsi_stochastic_reference
from tests_new.indicators.oracles.sma import sma_reference
from tests_new.indicators.oracles.standard_deviation_ewma import standard_deviation_ewma_reference
from tests_new.indicators.oracles.standard_deviation_rolling import standard_deviation_rolling_reference
from tests_new.indicators.oracles.stochastic_fast import stochastic_fast_reference
from tests_new.indicators.oracles.stochastic_slow import stochastic_slow_reference
from tests_new.indicators.oracles.supertrend import supertrend_reference
from tests_new.indicators.oracles.t3 import t3_reference
from tests_new.indicators.oracles.tema import tema_reference
from tests_new.indicators.oracles.time_series_forecast import time_series_forecast_reference
from tests_new.indicators.oracles.trima import trima_reference
from tests_new.indicators.oracles.trix import trix_reference
from tests_new.indicators.oracles.true_range import true_range_reference
from tests_new.indicators.oracles.ultimate_oscillator import ultimate_oscillator_reference
from tests_new.indicators.oracles.variance_ewma import variance_ewma_reference
from tests_new.indicators.oracles.variance_rolling import variance_rolling_reference
from tests_new.indicators.oracles.vortex import vortex_reference
from tests_new.indicators.oracles.vwap import vwap_reference
from tests_new.indicators.oracles.vwma import vwma_reference
from tests_new.indicators.oracles.williams_r import williams_r_reference
from tests_new.indicators.oracles.wma import wma_reference

__all__ = (
    "absolute_price_oscillator_reference",
    "accumulation_distribution_oscillator_reference",
    "accumulation_distribution_reference",
    "adx_reference",
    "adxr_reference",
    "aroon_oscillator_reference",
    "aroon_reference",
    "atr_normalized_reference",
    "atr_reference",
    "awesome_oscillator_reference",
    "balance_of_power_reference",
    "bollinger_bands_reference",
    "cci_reference",
    "chaikin_money_flow_reference",
    "chande_momentum_oscillator_reference",
    "dema_reference",
    "di_minus_reference",
    "di_plus_reference",
    "dm_minus_reference",
    "dm_plus_reference",
    "dominant_cycle_period_reference",
    "dominant_cycle_phase_reference",
    "donchian_channels_reference",
    "dx_reference",
    "ema_reference",
    "fisher_transform_reference",
    "hilbert_phasor_reference",
    "hilbert_trendline_reference",
    "hma_reference",
    "ichimoku_reference",
    "kama_reference",
    "keltner_channels_reference",
    "linear_regression_angle_reference",
    "linear_regression_intercept_reference",
    "linear_regression_reference",
    "linear_regression_slope_reference",
    "macd_reference",
    "mama_reference",
    "midpoint_reference",
    "midprice_reference",
    "mom_reference",
    "money_flow_index_reference",
    "obv_reference",
    "parabolic_sar_reference",
    "percentage_price_oscillator_reference",
    "price_average_reference",
    "price_median_reference",
    "price_typical_reference",
    "price_weighted_close_reference",
    "rma_reference",
    "roc_reference",
    "rsi_reference",
    "rsi_stochastic_reference",
    "sine_wave_reference",
    "sma_reference",
    "standard_deviation_ewma_reference",
    "standard_deviation_rolling_reference",
    "stochastic_fast_reference",
    "stochastic_slow_reference",
    "supertrend_reference",
    "t3_reference",
    "tema_reference",
    "time_series_forecast_reference",
    "trend_mode_reference",
    "trima_reference",
    "trix_reference",
    "true_range_reference",
    "ultimate_oscillator_reference",
    "variance_ewma_reference",
    "variance_rolling_reference",
    "vortex_reference",
    "vwap_reference",
    "vwma_reference",
    "williams_r_reference",
    "wma_reference",
)
