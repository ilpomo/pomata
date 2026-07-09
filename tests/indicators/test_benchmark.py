"""
Performance benchmarks and complexity-scaling checks for every indicator (non-gating).

Kept out of the default run: the module skips unless ``POMATA_BENCHMARKS`` is set, and a dedicated, non-gating CI job
runs it. The pytest-benchmark timings give performance visibility (and a baseline for ``--benchmark-compare``); the
scaling check guards against an accidental super-linear regression — an O(n) kernel slipping to O(n^2) — which the
absolute timings alone would not reveal. Every public indicator is covered, so a regression is caught wherever it lands;
of them only the two Python-kernel recursions (kama, parabolic_sar) are slower than vectorized (Rust under Polars) and
would gain from a future native kernel.
"""

import os
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from tests.support import fastest_eval

from pomata import indicators
from pomata.indicators import (
    absolute_price_oscillator,
    accumulation_distribution,
    accumulation_distribution_oscillator,
    adx,
    adxr,
    aroon,
    aroon_oscillator,
    atr,
    atr_normalized,
    awesome_oscillator,
    balance_of_power,
    bollinger_bands,
    cci,
    chaikin_money_flow,
    chande_momentum_oscillator,
    dema,
    di_minus,
    di_plus,
    dm_minus,
    dm_plus,
    dominant_cycle_period,
    dominant_cycle_phase,
    donchian_channels,
    dx,
    ema,
    fisher_transform,
    hilbert_phasor,
    hilbert_trendline,
    hma,
    ichimoku,
    kama,
    keltner_channels,
    linear_regression,
    linear_regression_angle,
    linear_regression_intercept,
    linear_regression_slope,
    macd,
    mama,
    midpoint,
    midprice,
    mom,
    money_flow_index,
    obv,
    parabolic_sar,
    percentage_price_oscillator,
    price_average,
    price_median,
    price_typical,
    price_weighted_close,
    rma,
    roc,
    rsi,
    rsi_stochastic,
    sine_wave,
    sma,
    standard_deviation_ewma,
    standard_deviation_rolling,
    stochastic_fast,
    stochastic_slow,
    supertrend,
    t3,
    tema,
    time_series_forecast,
    trend_mode,
    trima,
    trix,
    true_range,
    ultimate_oscillator,
    variance_ewma,
    variance_rolling,
    vortex,
    vwap,
    vwma,
    williams_r,
    wma,
)

if not os.environ.get("POMATA_BENCHMARKS"):
    pytest.skip("set POMATA_BENCHMARKS=1 to run the benchmark tier", allow_module_level=True)


def _frame(size: int) -> pl.DataFrame:
    """
    A deterministic OHLCV frame of ``size`` rows for timing.
    """
    rng = np.random.default_rng(0)
    close = 100.0 + rng.normal(0.0, 1.0, size).cumsum()
    high = close + 0.5
    low = close - 0.5
    return pl.DataFrame(
        {
            "open": low + 0.5 * (high - low),
            "high": high,
            "low": low,
            "close": close,
            "volume": np.abs(rng.normal(1e4, 2e3, size)) + 1.0,
        }
    )


# Column handles, kept short so the case table fits one line each.
OPEN, HIGH, LOW, CLOSE, VOLUME = pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")


CASES: dict[str, Callable[[], pl.Expr]] = {
    "accumulation_distribution": lambda: accumulation_distribution(HIGH, LOW, CLOSE, VOLUME),
    "accumulation_distribution_oscillator": lambda: accumulation_distribution_oscillator(
        HIGH, LOW, CLOSE, VOLUME, window_fast=3, window_slow=10
    ),
    "adx": lambda: adx(HIGH, LOW, CLOSE, 14),
    "adxr": lambda: adxr(HIGH, LOW, CLOSE, 14),
    "absolute_price_oscillator": lambda: absolute_price_oscillator(CLOSE, window_fast=12, window_slow=26),
    "aroon": lambda: aroon(HIGH, LOW, 14).struct.field("up"),
    "aroon_oscillator": lambda: aroon_oscillator(HIGH, LOW, 14),
    "atr": lambda: atr(HIGH, LOW, CLOSE, 14),
    "atr_normalized": lambda: atr_normalized(HIGH, LOW, CLOSE, 14),
    "awesome_oscillator": lambda: awesome_oscillator(HIGH, LOW, window_fast=5, window_slow=34),
    "bollinger_bands": lambda: bollinger_bands(CLOSE, 14).struct.field("lower"),
    "balance_of_power": lambda: balance_of_power(OPEN, HIGH, LOW, CLOSE),
    "cci": lambda: cci(HIGH, LOW, CLOSE, 14),
    "chaikin_money_flow": lambda: chaikin_money_flow(HIGH, LOW, CLOSE, VOLUME, 14),
    "chande_momentum_oscillator": lambda: chande_momentum_oscillator(CLOSE, 14),
    "dema": lambda: dema(CLOSE, 14),
    "di_minus": lambda: di_minus(HIGH, LOW, CLOSE, 14),
    "di_plus": lambda: di_plus(HIGH, LOW, CLOSE, 14),
    "dm_minus": lambda: dm_minus(HIGH, LOW, 14),
    "dm_plus": lambda: dm_plus(HIGH, LOW, 14),
    "dominant_cycle_period": lambda: dominant_cycle_period(CLOSE),
    "dominant_cycle_phase": lambda: dominant_cycle_phase(CLOSE),
    "donchian_channels": lambda: donchian_channels(HIGH, LOW, 20).struct.field("lower"),
    "dx": lambda: dx(HIGH, LOW, CLOSE, 14),
    "ema": lambda: ema(CLOSE, 14),
    "fisher_transform": lambda: fisher_transform(HIGH, LOW, 9).struct.field("fisher"),
    "hilbert_phasor": lambda: hilbert_phasor(CLOSE).struct.field("in_phase"),
    "hilbert_trendline": lambda: hilbert_trendline(CLOSE),
    "hma": lambda: hma(CLOSE, 14),
    "ichimoku": lambda: ichimoku(HIGH, LOW, window_tenkan=9, window_kijun=26, window_senkou=52).struct.field("tenkan"),
    "kama": lambda: kama(CLOSE, window=14, window_fast=2, window_slow=30),
    "keltner_channels": lambda: keltner_channels(HIGH, LOW, CLOSE, window=20, window_atr=10).struct.field("lower"),
    "linear_regression": lambda: linear_regression(CLOSE, 14),
    "linear_regression_angle": lambda: linear_regression_angle(CLOSE, 14),
    "linear_regression_intercept": lambda: linear_regression_intercept(CLOSE, 14),
    "linear_regression_slope": lambda: linear_regression_slope(CLOSE, 14),
    "macd": lambda: macd(CLOSE, window_fast=12, window_slow=26, window_signal=9).struct.field("macd"),
    "mama": lambda: mama(CLOSE).struct.field("mama"),
    "midpoint": lambda: midpoint(CLOSE, 14),
    "midprice": lambda: midprice(HIGH, LOW, 14),
    "mom": lambda: mom(CLOSE, 14),
    "money_flow_index": lambda: money_flow_index(HIGH, LOW, CLOSE, VOLUME, 14),
    "obv": lambda: obv(CLOSE, VOLUME),
    "parabolic_sar": lambda: parabolic_sar(HIGH, LOW),
    "percentage_price_oscillator": lambda: percentage_price_oscillator(CLOSE, window_fast=12, window_slow=26),
    "price_average": lambda: price_average(OPEN, HIGH, LOW, CLOSE),
    "price_median": lambda: price_median(HIGH, LOW),
    "price_typical": lambda: price_typical(HIGH, LOW, CLOSE),
    "price_weighted_close": lambda: price_weighted_close(HIGH, LOW, CLOSE),
    "rma": lambda: rma(CLOSE, 14),
    "roc": lambda: roc(CLOSE, 14),
    "rsi": lambda: rsi(CLOSE, 14),
    "rsi_stochastic": lambda: rsi_stochastic(CLOSE, window_rsi=14, window_k=14, window_d=14).struct.field("k"),
    "sine_wave": lambda: sine_wave(CLOSE).struct.field("sine"),
    "sma": lambda: sma(CLOSE, 14),
    "standard_deviation_ewma": lambda: standard_deviation_ewma(CLOSE, 14),
    "standard_deviation_rolling": lambda: standard_deviation_rolling(CLOSE, 14),
    "stochastic_fast": lambda: stochastic_fast(HIGH, LOW, CLOSE, window_k=14, window_d=14).struct.field("k"),
    "stochastic_slow": lambda: stochastic_slow(
        HIGH, LOW, CLOSE, window_k=14, window_slowing=14, window_d=14
    ).struct.field("k"),
    "supertrend": lambda: supertrend(HIGH, LOW, CLOSE, 10).struct.field("line"),
    "t3": lambda: t3(CLOSE, 14),
    "tema": lambda: tema(CLOSE, 14),
    "time_series_forecast": lambda: time_series_forecast(CLOSE, 14),
    "trend_mode": lambda: trend_mode(CLOSE),
    "trima": lambda: trima(CLOSE, 14),
    "trix": lambda: trix(CLOSE, 14),
    "true_range": lambda: true_range(HIGH, LOW, CLOSE),
    "ultimate_oscillator": lambda: ultimate_oscillator(
        HIGH, LOW, CLOSE, window_short=7, window_medium=14, window_long=28
    ),
    "variance_ewma": lambda: variance_ewma(CLOSE, 14),
    "variance_rolling": lambda: variance_rolling(CLOSE, 14),
    "vortex": lambda: vortex(HIGH, LOW, CLOSE, 14).struct.field("plus"),
    "vwap": lambda: vwap(HIGH, LOW, CLOSE, VOLUME),
    "vwma": lambda: vwma(CLOSE, VOLUME, 14),
    "williams_r": lambda: williams_r(HIGH, LOW, CLOSE, 14),
    "wma": lambda: wma(CLOSE, 14),
}


@pytest.mark.benchmark
def test_cases_cover_public_surface() -> None:
    """
    Verifies that ``CASES`` covers exactly the public indicator surface, so a newly added indicator cannot slip the
    benchmark net (its complexity-scaling regression would otherwise go unmeasured).
    """
    assert set(CASES) == set(indicators.__all__)


@pytest.mark.benchmark
@pytest.mark.parametrize("name", sorted(CASES), ids=sorted(CASES))
def test_throughput(benchmark: BenchmarkFixture, name: str) -> None:
    """
    Records the evaluation time of each indicator over a fixed-size frame.
    """
    frame = _frame(100_000)
    expr = CASES[name]().alias("y")
    benchmark(lambda: frame.select(expr))


@pytest.mark.benchmark
@pytest.mark.parametrize("name", sorted(CASES), ids=sorted(CASES))
def test_scales_sub_quadratically(name: str) -> None:
    """
    Verifies that a 10x increase in rows costs well under 100x the time, guarding against an O(n^2) regression.

    The bound is multiplicative with a small additive floor: the cheapest indicators evaluate in well under a
    millisecond, where the time is dominated by fixed overhead rather than the row count, and the floor keeps that
    measurement noise from failing them. A genuine super-linear regression blows the absolute time up to seconds at a
    million rows, far past either term.
    """
    build = CASES[name]
    base = fastest_eval(_frame(100_000), build)
    large = fastest_eval(_frame(1_000_000), build)
    assert large < 25.0 * base + 0.02, f"{name}: {large:.4f}s vs {base:.4f}s base for 10x the rows (super-linear?)"
