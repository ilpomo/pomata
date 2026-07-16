"""
The TA-Lib coverage registry: which indicators have no TA-Lib twin and which deliberately diverge from it.

Pure data with no TA-Lib dependency, so the docstring and README guards can read the partition on every run —
``tests/test_differential.py`` (whose module import skips wherever the optional ``talib`` package is absent)
consumes the same two mappings to prove the partition covers ``pomata.indicators.__all__``.
"""

__all__ = ("DOCUMENTED_DIVERGENCES", "NO_TALIB_EQUIVALENT")

# Indicators with no TA-Lib twin: the differential tier cannot cover them. Listed so the gap is explicit, not silent.
NO_TALIB_EQUIVALENT: dict[str, str] = {
    "awesome_oscillator": "TA-Lib has no Awesome Oscillator.",
    "chaikin_money_flow": "TA-Lib has the A/D line and Chaikin oscillator, but not the volume-normalized CMF.",
    "donchian_channels": "TA-Lib has no Donchian channels.",
    "fisher_transform": "TA-Lib has no Fisher Transform.",
    "hma": "TA-Lib has no Hull moving average.",
    "ichimoku": "TA-Lib has no Ichimoku Kinko Hyo.",
    "keltner_channels": "TA-Lib has no Keltner channels.",
    "rma": "TA-Lib has no Wilder smoothing (SMMA) as a standalone function.",
    "standard_deviation_ewma": "TA-Lib STDDEV is windowed; there is no exponentially-weighted standard deviation.",
    "supertrend": "TA-Lib has no SuperTrend.",
    "variance_ewma": "TA-Lib VAR is windowed; there is no exponentially-weighted variance.",
    "vortex": "TA-Lib has no Vortex indicator.",
    "vwap": "TA-Lib has no VWAP.",
    "vwma": "TA-Lib has no volume-weighted moving average.",
}

# Indicators that map to a TA-Lib function but follow a different (deliberate) definitional convention, so the
# steady-state tail does not agree. Documented with the exact difference; see each indicator's own docstring.
DOCUMENTED_DIVERGENCES: dict[str, str] = {
    "adxr": "ADXR averages ADX with its lagged self; pomata lags by `window`, TA-Lib by `window - 1`.",
    "chande_momentum_oscillator": (
        "pomata uses Chande's original fixed-window sums; TA-Lib uses Wilder smoothing (CMO == 2*RSI - 100)."
    ),
    "obv": "OBV is a cumulative sum with an arbitrary origin; pomata seeds OBV[0] = 0, TA-Lib uses volume[0].",
}
