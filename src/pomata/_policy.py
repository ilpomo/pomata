"""
The declared null / NaN policy of every public function — the package's one policy registry.

These are the two facts about a function that a test cannot observe and so must be *stated*: the declaration encodes
intent, and :mod:`tests.test_policies` proves it against the code on every run. The registry lives in the package —
not in the test suite — so there is a single source of truth for every consumer: the suite imports it back through
``tests/support/policies.py``, and the docstrings state each function's pair in prose. See ``tests/README.md`` for
the full definition of each state.
"""

from enum import Enum


class NullPolicy(Enum):
    """What an interior ``null`` does to the output (see ``tests/README.md`` for the full definition of each)."""

    SKIPPED = "skipped"  # excluded from the reduction; the result is as if the null were absent
    ABSORBED = "absorbed"  # the pointwise computation skips the null candidate entirely; no output row is nulled
    PROPAGATES = "propagates"  # nulls at most its own output row and a one-bar lag (a pointwise map)
    IN_WINDOW_IS_NULL = "in_window_is_null"  # nulls every window that overlaps it, then recovers
    BRIDGED = "bridged"  # a recursion steps over it (state carries), so later rows recover
    LATCHES = "latches"  # contaminates every subsequent row


class NanPolicy(Enum):
    """What an interior ``NaN`` does to the output (see ``tests/README.md`` for the full definition of each)."""

    POISONS = "poisons"  # a reduction goes ``NaN``
    PROPAGATES = "propagates"  # nans the rows it reaches, then recovers
    LATCHES = "latches"  # a recursion carries it forward forever


# One row per public function: its declared ``(null_policy, nan_policy)``. Proven against actual behaviour, and kept in
# exact bijection with the public surface, by :mod:`tests.test_policies`.
POLICIES: dict[str, tuple[NullPolicy, NanPolicy]] = {
    # indicators
    "absolute_price_oscillator": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "accumulation_distribution": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "accumulation_distribution_oscillator": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "adx": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "adxr": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "aroon": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "aroon_oscillator": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "atr": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "atr_normalized": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "awesome_oscillator": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "balance_of_power": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "bollinger_bands": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "cci": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "chaikin_money_flow": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "chande_momentum_oscillator": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "dema": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "di_minus": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "di_plus": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "dm_minus": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "dm_plus": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "dominant_cycle_period": (NullPolicy.LATCHES, NanPolicy.LATCHES),
    "dominant_cycle_phase": (NullPolicy.LATCHES, NanPolicy.LATCHES),
    "donchian_channels": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "dx": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "ema": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "fisher_transform": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "hilbert_phasor": (NullPolicy.LATCHES, NanPolicy.LATCHES),
    "hilbert_trendline": (NullPolicy.LATCHES, NanPolicy.LATCHES),
    "hma": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "ichimoku": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "kama": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "keltner_channels": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "linear_regression": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "linear_regression_angle": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "linear_regression_intercept": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "linear_regression_slope": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "macd": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "mama": (NullPolicy.LATCHES, NanPolicy.LATCHES),
    "midpoint": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "midprice": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "mom": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "money_flow_index": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "obv": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "parabolic_sar": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "percentage_price_oscillator": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "price_average": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "price_median": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "price_typical": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "price_weighted_close": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "rma": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "roc": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "rsi": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "rsi_stochastic": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "sine_wave": (NullPolicy.LATCHES, NanPolicy.LATCHES),
    "sma": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "standard_deviation_ewma": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "standard_deviation_rolling": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "stochastic_fast": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "stochastic_slow": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "supertrend": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "t3": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "tema": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "time_series_forecast": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "trend_mode": (NullPolicy.LATCHES, NanPolicy.LATCHES),
    "trima": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "trix": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "true_range": (NullPolicy.ABSORBED, NanPolicy.PROPAGATES),
    "ultimate_oscillator": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "variance_ewma": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "variance_rolling": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "vortex": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "vwap": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "vwma": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "williams_r": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "wma": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    # pnl
    "cost_borrow": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "cost_fixed": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "cost_funding": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "cost_notional": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "cost_per_share": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "cost_proportional": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "cost_slippage": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "cumulative_pnl": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "dividend": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "equity_curve": (NullPolicy.BRIDGED, NanPolicy.LATCHES),
    "pnl_gross": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "pnl_gross_inverse": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "pnl_net": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "returns_gross": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "returns_log": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "returns_net": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "returns_simple": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "turnover": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    # metrics
    "adjusted_sharpe_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "alpha": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "alpha_rolling": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "beta": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "beta_rolling": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "burke_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "cagr": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "cagr_rolling": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "calmar_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "capture_downside_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "capture_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "capture_upside_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "common_sense_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "conditional_drawdown_at_risk": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "conditional_value_at_risk": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "downside_deviation": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "downside_deviation_rolling": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "drawdown": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "drawdown_rolling": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "gain_to_pain_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "information_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "information_ratio_rolling": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "kelly_criterion": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "kurtosis": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "kurtosis_rolling": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "max_drawdown": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "max_drawdown_duration": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "modigliani_risk_adjusted_performance": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "omega_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "omega_ratio_rolling": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "pain_index": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "pain_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "payoff_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "probabilistic_sharpe_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "profit_factor": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "recovery_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "risk_of_ruin": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "sharpe_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "sharpe_ratio_rolling": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "skewness": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "skewness_rolling": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "sortino_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "sortino_ratio_rolling": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "stability": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "sterling_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "tail_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "tail_ratio_rolling": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "total_return": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "total_return_rolling": (NullPolicy.PROPAGATES, NanPolicy.PROPAGATES),
    "treynor_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "treynor_ratio_rolling": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "ulcer_index": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "ulcer_performance_ratio": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "value_at_risk": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "value_at_risk_modified": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "value_at_risk_parametric": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "value_at_risk_rolling": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "volatility": (NullPolicy.SKIPPED, NanPolicy.POISONS),
    "volatility_rolling": (NullPolicy.IN_WINDOW_IS_NULL, NanPolicy.PROPAGATES),
    "win_rate": (NullPolicy.SKIPPED, NanPolicy.POISONS),
}

# Functions whose correctness is pinned by a golden master rather than an independent ``*_reference`` oracle.
NO_ORACLE: frozenset[str] = frozenset(
    {
        "dominant_cycle_period",
        "dominant_cycle_phase",
        "hilbert_phasor",
        "hilbert_trendline",
        "mama",
        "sine_wave",
        "trend_mode",
    }
)
