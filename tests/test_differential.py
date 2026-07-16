"""
Differential tests: each indicator against TA-Lib (the C reference), at the reference tolerance.

This is the cross-cutting ``differential`` tier (non-gating). It needs the ``differential`` dependency group; the whole
module is skipped when TA-Lib is not installed, so the default gate is unaffected. TA-Lib is the de-facto industry
reference -- an independent C implementation, a genuinely different computation path -- so agreement is strong external
evidence of correctness.

Most indicators are compared over the **whole series, from the first defined value**, at the reference tolerance: with
the canonical SMA seeding they match TA-Lib bar for bar. A documented minority (``STEADY_STATE_ONLY``) is compared only
on the converged tail, each with the reason its warm-up differs -- always a case where TA-Lib deviates from the
indicator's author (Wilder's first true range, the independent MACD/Chaikin EMAs) or a long implementation-specific
warm-up (Ehlers' Hilbert pipeline, the Parabolic SAR cold start), never a steady-state disagreement.

Self-contained by design: the market frame IS the fixture being pinned, so this module names the OHLCV columns literally
and calls the factories directly rather than routing through the registry -- the comparison windows here are tuned to
TA-Lib's own calls (e.g. ``kama`` at 30, ``rsi_stochastic`` with ``window_d=3``) and deliberately differ from the
canonical spec parameters, so they cannot be derived from the spec registry without changing the values.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest
from tests.support import ABSOLUTE_TOLERANCE_REFERENCE, RELATIVE_TOLERANCE_REFERENCE
from tests.support.talib_coverage import DOCUMENTED_DIVERGENCES, NO_TALIB_EQUIVALENT

from pomata import indicators
from pomata.indicators import (
    absolute_price_oscillator,
    accumulation_distribution,
    accumulation_distribution_oscillator,
    adx,
    aroon,
    aroon_oscillator,
    atr,
    atr_normalized,
    balance_of_power,
    bollinger_bands,
    cci,
    dema,
    di_minus,
    di_plus,
    dm_minus,
    dm_plus,
    dominant_cycle_period,
    dominant_cycle_phase,
    dx,
    ema,
    hilbert_phasor,
    hilbert_trendline,
    kama,
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
    parabolic_sar,
    percentage_price_oscillator,
    price_average,
    price_median,
    price_typical,
    price_weighted_close,
    roc,
    rsi,
    rsi_stochastic,
    sine_wave,
    sma,
    standard_deviation_rolling,
    stochastic_fast,
    stochastic_slow,
    t3,
    tema,
    time_series_forecast,
    trend_mode,
    trima,
    trix,
    true_range,
    ultimate_oscillator,
    variance_rolling,
    williams_r,
    wma,
)

talib: Any = pytest.importorskip("talib")

# A long series across a few seeds. Full-series indicators are checked from their first defined value; the
# STEADY_STATE_ONLY minority is checked over the last TAIL rows, where any warm-up transient has fully decayed -- SIZE
# is large enough that even Wilder's slow (1 - 1/n) decay reaches the reference tolerance by then.
SIZE = 600
TAIL = 60
SEEDS = (1, 2, 3)


def _market(seed: int) -> dict[str, Any]:
    """
    A long, well-formed OHLCV series as both numpy arrays (for TA-Lib) and a Polars frame (for pomata).
    """
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0.0, 1.0, SIZE).cumsum()
    high = close + np.abs(rng.normal(0.0, 0.5, SIZE))
    low = close - np.abs(rng.normal(0.0, 0.5, SIZE))
    open_ = low + rng.uniform(0.0, 1.0, SIZE) * (high - low)
    volume = np.abs(rng.normal(1e4, 2e3, SIZE)) + 1.0
    frame = pl.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "frame": frame}


def _ours(frame: pl.DataFrame, expr: pl.Expr) -> list[float | None]:
    """
    Materialize a pomata expression over the market frame as a Python list.
    """
    return frame.select(expr.alias("y"))["y"].to_list()


# Each spec maps an indicator name to its (label, pomata-output, TA-Lib-output) triples on a given market. Multi-output
# indicators (e.g. macd) yield one triple per field.
Compare = Callable[[dict[str, Any]], list[tuple[str, list[float | None], Any]]]

SPECS: dict[str, Compare] = {
    "accumulation_distribution": lambda m: [
        (
            "accumulation_distribution",
            _ours(
                m["frame"], accumulation_distribution(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"))
            ),
            talib.AD(m["high"], m["low"], m["close"], m["volume"]),
        )
    ],
    "accumulation_distribution_oscillator": lambda m: [
        (
            "accumulation_distribution_oscillator",
            _ours(
                m["frame"],
                accumulation_distribution_oscillator(
                    pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), window_fast=3, window_slow=10
                ),
            ),
            talib.ADOSC(m["high"], m["low"], m["close"], m["volume"], 3, 10),
        )
    ],
    "adx": lambda m: [
        (
            "adx",
            _ours(m["frame"], adx(pl.col("high"), pl.col("low"), pl.col("close"), 14)),
            talib.ADX(m["high"], m["low"], m["close"], 14),
        )
    ],
    "absolute_price_oscillator": lambda m: [
        (
            "absolute_price_oscillator",
            _ours(m["frame"], absolute_price_oscillator(pl.col("close"), window_fast=12, window_slow=26)),
            talib.APO(m["close"], 12, 26, 1),
        )
    ],
    "aroon": lambda m: [
        (
            "up",
            _ours(m["frame"], aroon(pl.col("high"), pl.col("low"), 14).struct.field("up")),
            talib.AROON(m["high"], m["low"], 14)[1],
        ),
        (
            "down",
            _ours(m["frame"], aroon(pl.col("high"), pl.col("low"), 14).struct.field("down")),
            talib.AROON(m["high"], m["low"], 14)[0],
        ),
    ],
    "aroon_oscillator": lambda m: [
        (
            "aroon_oscillator",
            _ours(m["frame"], aroon_oscillator(pl.col("high"), pl.col("low"), 14)),
            talib.AROONOSC(m["high"], m["low"], 14),
        )
    ],
    "atr": lambda m: [
        (
            "atr",
            _ours(m["frame"], atr(pl.col("high"), pl.col("low"), pl.col("close"), 14)),
            talib.ATR(m["high"], m["low"], m["close"], 14),
        )
    ],
    "atr_normalized": lambda m: [
        (
            "atr_normalized",
            _ours(m["frame"], atr_normalized(pl.col("high"), pl.col("low"), pl.col("close"), 14)),
            talib.NATR(m["high"], m["low"], m["close"], 14),
        )
    ],
    "bollinger_bands": lambda m: [
        (
            "upper",
            _ours(m["frame"], bollinger_bands(pl.col("close"), 14).struct.field("upper")),
            talib.BBANDS(m["close"], 14, 2.0, 2.0, 0)[0],
        ),
        (
            "middle",
            _ours(m["frame"], bollinger_bands(pl.col("close"), 14).struct.field("middle")),
            talib.BBANDS(m["close"], 14, 2.0, 2.0, 0)[1],
        ),
        (
            "lower",
            _ours(m["frame"], bollinger_bands(pl.col("close"), 14).struct.field("lower")),
            talib.BBANDS(m["close"], 14, 2.0, 2.0, 0)[2],
        ),
    ],
    "balance_of_power": lambda m: [
        (
            "balance_of_power",
            _ours(m["frame"], balance_of_power(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))),
            talib.BOP(m["open"], m["high"], m["low"], m["close"]),
        )
    ],
    "cci": lambda m: [
        (
            "cci",
            _ours(m["frame"], cci(pl.col("high"), pl.col("low"), pl.col("close"), 14)),
            talib.CCI(m["high"], m["low"], m["close"], 14),
        )
    ],
    "dema": lambda m: [("dema", _ours(m["frame"], dema(pl.col("close"), 14)), talib.DEMA(m["close"], 14))],
    "di_minus": lambda m: [
        (
            "di_minus",
            _ours(m["frame"], di_minus(pl.col("high"), pl.col("low"), pl.col("close"), 14)),
            talib.MINUS_DI(m["high"], m["low"], m["close"], 14),
        )
    ],
    "di_plus": lambda m: [
        (
            "di_plus",
            _ours(m["frame"], di_plus(pl.col("high"), pl.col("low"), pl.col("close"), 14)),
            talib.PLUS_DI(m["high"], m["low"], m["close"], 14),
        )
    ],
    "dm_minus": lambda m: [
        (
            "dm_minus",
            _ours(m["frame"], dm_minus(pl.col("high"), pl.col("low"), 14) * 14),
            talib.MINUS_DM(m["high"], m["low"], 14),
        )
    ],
    "dm_plus": lambda m: [
        (
            "dm_plus",
            _ours(m["frame"], dm_plus(pl.col("high"), pl.col("low"), 14) * 14),
            talib.PLUS_DM(m["high"], m["low"], 14),
        )
    ],
    "dominant_cycle_period": lambda m: [
        (
            "dominant_cycle_period",
            _ours(m["frame"], dominant_cycle_period(pl.col("close"))),
            talib.HT_DCPERIOD(m["close"]),
        )
    ],
    "dominant_cycle_phase": lambda m: [
        (
            "dominant_cycle_phase",
            _ours(m["frame"], dominant_cycle_phase(pl.col("close"))),
            talib.HT_DCPHASE(m["close"]),
        ),
    ],
    "dx": lambda m: [
        (
            "dx",
            _ours(m["frame"], dx(pl.col("high"), pl.col("low"), pl.col("close"), 14)),
            talib.DX(m["high"], m["low"], m["close"], 14),
        )
    ],
    "ema": lambda m: [("ema", _ours(m["frame"], ema(pl.col("close"), 14)), talib.EMA(m["close"], 14))],
    "hilbert_phasor": lambda m: [
        (
            "in_phase",
            _ours(m["frame"], hilbert_phasor(pl.col("close")).struct.field("in_phase")),
            talib.HT_PHASOR(m["close"])[0],
        ),
        (
            "quadrature",
            _ours(m["frame"], hilbert_phasor(pl.col("close")).struct.field("quadrature")),
            talib.HT_PHASOR(m["close"])[1],
        ),
    ],
    "hilbert_trendline": lambda m: [
        ("hilbert_trendline", _ours(m["frame"], hilbert_trendline(pl.col("close"))), talib.HT_TRENDLINE(m["close"])),
    ],
    "kama": lambda m: [
        (
            "kama",
            _ours(m["frame"], kama(pl.col("close"), window=30, window_fast=2, window_slow=30)),
            talib.KAMA(m["close"], 30),
        )
    ],
    "linear_regression": lambda m: [
        (
            "linear_regression",
            _ours(m["frame"], linear_regression(pl.col("close"), 14)),
            talib.LINEARREG(m["close"], 14),
        )
    ],
    "linear_regression_angle": lambda m: [
        (
            "linear_regression_angle",
            _ours(m["frame"], linear_regression_angle(pl.col("close"), 14)),
            talib.LINEARREG_ANGLE(m["close"], 14),
        )
    ],
    "linear_regression_intercept": lambda m: [
        (
            "linear_regression_intercept",
            _ours(m["frame"], linear_regression_intercept(pl.col("close"), 14)),
            talib.LINEARREG_INTERCEPT(m["close"], 14),
        )
    ],
    "linear_regression_slope": lambda m: [
        (
            "linear_regression_slope",
            _ours(m["frame"], linear_regression_slope(pl.col("close"), 14)),
            talib.LINEARREG_SLOPE(m["close"], 14),
        )
    ],
    "macd": lambda m: [
        (
            "macd",
            _ours(
                m["frame"], macd(pl.col("close"), window_fast=12, window_slow=26, window_signal=9).struct.field("macd")
            ),
            talib.MACD(m["close"], 12, 26, 9)[0],
        ),
        (
            "signal",
            _ours(
                m["frame"],
                macd(pl.col("close"), window_fast=12, window_slow=26, window_signal=9).struct.field("signal"),
            ),
            talib.MACD(m["close"], 12, 26, 9)[1],
        ),
        (
            "histogram",
            _ours(
                m["frame"],
                macd(pl.col("close"), window_fast=12, window_slow=26, window_signal=9).struct.field("histogram"),
            ),
            talib.MACD(m["close"], 12, 26, 9)[2],
        ),
    ],
    "mama": lambda m: [
        ("mama", _ours(m["frame"], mama(pl.col("close")).struct.field("mama")), talib.MAMA(m["close"])[0]),
        ("fama", _ours(m["frame"], mama(pl.col("close")).struct.field("fama")), talib.MAMA(m["close"])[1]),
    ],
    "midpoint": lambda m: [
        ("midpoint", _ours(m["frame"], midpoint(pl.col("close"), 14)), talib.MIDPOINT(m["close"], 14))
    ],
    "midprice": lambda m: [
        (
            "midprice",
            _ours(m["frame"], midprice(pl.col("high"), pl.col("low"), 14)),
            talib.MIDPRICE(m["high"], m["low"], 14),
        )
    ],
    "mom": lambda m: [("mom", _ours(m["frame"], mom(pl.col("close"), 14)), talib.MOM(m["close"], 14))],
    "money_flow_index": lambda m: [
        (
            "money_flow_index",
            _ours(m["frame"], money_flow_index(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), 14)),
            talib.MFI(m["high"], m["low"], m["close"], m["volume"], 14),
        )
    ],
    "parabolic_sar": lambda m: [
        (
            "parabolic_sar",
            _ours(m["frame"], parabolic_sar(pl.col("high"), pl.col("low"))),
            talib.SAR(m["high"], m["low"]),
        )
    ],
    "percentage_price_oscillator": lambda m: [
        (
            "percentage_price_oscillator",
            _ours(m["frame"], percentage_price_oscillator(pl.col("close"), window_fast=12, window_slow=26)),
            talib.PPO(m["close"], 12, 26, 1),
        )
    ],
    "price_average": lambda m: [
        (
            "price_average",
            _ours(m["frame"], price_average(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))),
            talib.AVGPRICE(m["open"], m["high"], m["low"], m["close"]),
        )
    ],
    "price_median": lambda m: [
        (
            "price_median",
            _ours(m["frame"], price_median(pl.col("high"), pl.col("low"))),
            talib.MEDPRICE(m["high"], m["low"]),
        )
    ],
    "price_typical": lambda m: [
        (
            "price_typical",
            _ours(m["frame"], price_typical(pl.col("high"), pl.col("low"), pl.col("close"))),
            talib.TYPPRICE(m["high"], m["low"], m["close"]),
        )
    ],
    "price_weighted_close": lambda m: [
        (
            "price_weighted_close",
            _ours(m["frame"], price_weighted_close(pl.col("high"), pl.col("low"), pl.col("close"))),
            talib.WCLPRICE(m["high"], m["low"], m["close"]),
        )
    ],
    "roc": lambda m: [("roc", _ours(m["frame"], roc(pl.col("close"), 14)), talib.ROC(m["close"], 14))],
    "rsi": lambda m: [("rsi", _ours(m["frame"], rsi(pl.col("close"), 14)), talib.RSI(m["close"], 14))],
    "rsi_stochastic": lambda m: [
        (
            "k",
            _ours(
                m["frame"], rsi_stochastic(pl.col("close"), window_rsi=14, window_k=14, window_d=3).struct.field("k")
            ),
            talib.STOCHRSI(m["close"], 14, 14, 3)[0],
        ),
        (
            "d",
            _ours(
                m["frame"], rsi_stochastic(pl.col("close"), window_rsi=14, window_k=14, window_d=3).struct.field("d")
            ),
            talib.STOCHRSI(m["close"], 14, 14, 3)[1],
        ),
    ],
    "sine_wave": lambda m: [
        ("sine", _ours(m["frame"], sine_wave(pl.col("close")).struct.field("sine")), talib.HT_SINE(m["close"])[0]),
        (
            "lead_sine",
            _ours(m["frame"], sine_wave(pl.col("close")).struct.field("lead_sine")),
            talib.HT_SINE(m["close"])[1],
        ),
    ],
    "sma": lambda m: [("sma", _ours(m["frame"], sma(pl.col("close"), 14)), talib.SMA(m["close"], 14))],
    "standard_deviation_rolling": lambda m: [
        (
            "standard_deviation_rolling",
            _ours(m["frame"], standard_deviation_rolling(pl.col("close"), 14, ddof=0)),
            talib.STDDEV(m["close"], 14, 1),
        )
    ],
    "stochastic_fast": lambda m: [
        (
            "k",
            _ours(
                m["frame"],
                stochastic_fast(pl.col("high"), pl.col("low"), pl.col("close"), window_k=14, window_d=3).struct.field(
                    "k"
                ),
            ),
            talib.STOCHF(m["high"], m["low"], m["close"], 14, 3)[0],
        ),
        (
            "d",
            _ours(
                m["frame"],
                stochastic_fast(pl.col("high"), pl.col("low"), pl.col("close"), window_k=14, window_d=3).struct.field(
                    "d"
                ),
            ),
            talib.STOCHF(m["high"], m["low"], m["close"], 14, 3)[1],
        ),
    ],
    "stochastic_slow": lambda m: [
        (
            "k",
            _ours(
                m["frame"],
                stochastic_slow(
                    pl.col("high"), pl.col("low"), pl.col("close"), window_k=14, window_slowing=3, window_d=3
                ).struct.field("k"),
            ),
            talib.STOCH(m["high"], m["low"], m["close"], 14, 3, 0, 3, 0)[0],
        ),
        (
            "d",
            _ours(
                m["frame"],
                stochastic_slow(
                    pl.col("high"), pl.col("low"), pl.col("close"), window_k=14, window_slowing=3, window_d=3
                ).struct.field("d"),
            ),
            talib.STOCH(m["high"], m["low"], m["close"], 14, 3, 0, 3, 0)[1],
        ),
    ],
    "t3": lambda m: [("t3", _ours(m["frame"], t3(pl.col("close"), 14)), talib.T3(m["close"], 14, 0.7))],
    "tema": lambda m: [("tema", _ours(m["frame"], tema(pl.col("close"), 14)), talib.TEMA(m["close"], 14))],
    "time_series_forecast": lambda m: [
        (
            "time_series_forecast",
            _ours(m["frame"], time_series_forecast(pl.col("close"), 14)),
            talib.TSF(m["close"], 14),
        )
    ],
    "trend_mode": lambda m: [
        ("trend_mode", _ours(m["frame"], trend_mode(pl.col("close"))), talib.HT_TRENDMODE(m["close"])),
    ],
    "trima": lambda m: [("trima", _ours(m["frame"], trima(pl.col("close"), 14)), talib.TRIMA(m["close"], 14))],
    "trix": lambda m: [("trix", _ours(m["frame"], trix(pl.col("close"), 14)), talib.TRIX(m["close"], 14))],
    "true_range": lambda m: [
        (
            "true_range",
            _ours(m["frame"], true_range(pl.col("high"), pl.col("low"), pl.col("close"))),
            talib.TRANGE(m["high"], m["low"], m["close"]),
        )
    ],
    "ultimate_oscillator": lambda m: [
        (
            "ultimate_oscillator",
            _ours(
                m["frame"],
                ultimate_oscillator(
                    pl.col("high"), pl.col("low"), pl.col("close"), window_short=7, window_medium=14, window_long=28
                ),
            ),
            talib.ULTOSC(m["high"], m["low"], m["close"], 7, 14, 28),
        )
    ],
    "variance_rolling": lambda m: [
        (
            "variance_rolling",
            _ours(m["frame"], variance_rolling(pl.col("close"), 14, ddof=0)),
            talib.VAR(m["close"], 14, 1),
        )
    ],
    "williams_r": lambda m: [
        (
            "williams_r",
            _ours(m["frame"], williams_r(pl.col("high"), pl.col("low"), pl.col("close"), 14)),
            talib.WILLR(m["high"], m["low"], m["close"], 14),
        )
    ],
    "wma": lambda m: [("wma", _ours(m["frame"], wma(pl.col("close"), 14)), talib.WMA(m["close"], 14))],
}

# The no-twin / deliberate-divergence partition lives in tests.support.talib_coverage (pure data, no TA-Lib
# dependency), so the docstring and README guards can read it on every run while this module stays talib-gated.


# Indicators whose warm-up legitimately differs from TA-Lib, compared only on the converged steady-state tail. In every
# case pomata follows the indicator's author (or carries a long implementation-specific warm-up) and TA-Lib is the one
# that deviates over the lead-in; the steady state agrees to the reference tolerance. Reasons are shared where the cause
# is the same.
_WILDER_TRUE_RANGE = "Wilder's first true range is the bar's high-low; TA-Lib omits that first bar over the warm-up."
_INDEPENDENT_EMAS = "Gap of two independent EMAs (Appel / Chaikin); TA-Lib aligns the fast EMA to the slow EMA's start."
_HILBERT_PIPELINE = "Ehlers' Hilbert-transform pipeline carries a long, implementation-specific warm-up."
_SAR_COLD_START = "Wilder leaves the initial trend unspecified; pomata and TA-Lib pick different cold starts."
STEADY_STATE_ONLY: dict[str, str] = {
    "accumulation_distribution_oscillator": _INDEPENDENT_EMAS,
    "adx": _WILDER_TRUE_RANGE,
    "atr": _WILDER_TRUE_RANGE,
    "atr_normalized": _WILDER_TRUE_RANGE,
    "di_minus": _WILDER_TRUE_RANGE,
    "di_plus": _WILDER_TRUE_RANGE,
    "dominant_cycle_period": _HILBERT_PIPELINE,
    "dominant_cycle_phase": _HILBERT_PIPELINE,
    "hilbert_phasor": _HILBERT_PIPELINE,
    "hilbert_trendline": _HILBERT_PIPELINE,
    "macd": _INDEPENDENT_EMAS,
    "mama": _HILBERT_PIPELINE,
    "parabolic_sar": _SAR_COLD_START,
    "sine_wave": _HILBERT_PIPELINE,
    "trend_mode": _HILBERT_PIPELINE,
}


@pytest.mark.differential
@pytest.mark.parametrize("name", sorted(SPECS), ids=sorted(SPECS))
def test_agrees_with_talib(name: str) -> None:
    """
    Verifies agreement with TA-Lib at the reference tolerance: over the whole series from the first defined value, or --
    for the documented ``STEADY_STATE_ONLY`` minority -- over the converged tail.
    """
    tail_only = name in STEADY_STATE_ONLY
    for seed in SEEDS:
        market = _market(seed)
        for label, ours, theirs in SPECS[name](market):
            assert len(ours) == len(theirs)
            window = list(zip(ours, theirs, strict=True))
            if tail_only:
                window = window[-TAIL:]
            compared = 0
            for our_value, their_value in window:
                if our_value is None or (isinstance(their_value, float) and math.isnan(their_value)):
                    continue
                assert math.isclose(
                    our_value,
                    float(their_value),
                    rel_tol=RELATIVE_TOLERANCE_REFERENCE,
                    abs_tol=ABSOLUTE_TOLERANCE_REFERENCE,
                ), f"{label}: pomata={our_value} vs TA-Lib={their_value} (seed {seed})"
                compared += 1
            assert compared > 0, f"{label}: no overlapping defined values"


@pytest.mark.differential
def test_every_indicator_is_accounted_for() -> None:
    """
    Verifies that every public indicator is either compared against TA-Lib or explicitly documented as not comparable.
    """
    covered = set(SPECS) | set(NO_TALIB_EQUIVALENT) | set(DOCUMENTED_DIVERGENCES)
    assert covered == set(indicators.__all__)
    assert set(STEADY_STATE_ONLY) <= set(SPECS)
