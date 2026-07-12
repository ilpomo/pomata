# DESIGN — the contract framework and the migration map

**Status:** framework only — no per-function contract has been migrated yet. The declaration surface below is
provisional until it has been proven on the four structurally hardest functions (`ichimoku`, `mama`,
`sharpe_ratio`, `equity_curve`); it freezes then, and the rollout proceeds family by family.

## The axes

The public surface varies along exactly three axes (probed from HEAD, 11 observed combinations across 153
functions), and the framework composes along them:

- **shape** — `ReducingContract` (43) | `SeriesContract` (90) | `StructContract` (15+5): what one probe row
  observes; struct contracts declare their `fields` and every rung reads **all** of them.
- **windowed** — `WindowedContract` (74): declares the exact `warmup` (an `int`, or a per-field mapping for
  structs whose lines warm up differently, e.g. `ichimoku`).
- **null/NaN policy** — **not a mixin**: derived from `pomata._policy` by function name and dispatched inside
  the shared flow rungs, so a contract cannot pair the wrong behavior with the wrong declaration.

## The declaration surface (what a child states — nothing else)

| Declaration | Required by | Meaning |
|---|---|---|
| `factory` | all | the `pl.Expr` factory under test (`staticmethod(...)`) |
| `inputs` | all | ordered input column roles (drawn from the probe-frame vocabulary) |
| `params` | all | the canonical scalar kwargs used by probes and goldens |
| `warmup` | windowed | exact leading-null count under `params` (`int` or per-field mapping) |
| `fields` | struct | the struct's field names, in order |
| `lands_on` | optional | landing column when it is not the first input (11 known exceptions) |
| `override_ok` | optional | visible consent to redefine an inherited rung (empty by default) |

Derived, never declared: `name` (from the factory), `family` (from `__all__`), `null_policy`/`nan_policy`
(from the registry). The Correctness/Properties declarations (`oracle`, `golden`, strategy/scale specs) are
added once the hardest-four proof fixes their exact shape.

## The three locks (all born red — see `support/tests/test_machinery.py`)

1. **Completeness**: a missing declaration dies at import, naming every gap.
2. **Honesty**: overriding an inherited rung without `override_ok` dies at import.
3. **Bijection**: `test_surface.py` holds the contract registry in exact two-way correspondence with the
   migrated surface (`MIGRATED`, extended as each family lands; replaced by `__all__` at cutover).

## Before any rollout

The remaining rung surface is implemented and proven on the four hardest functions first: `.over()` partition
independence, lazy/eager parity, the policy-dispatched null/NaN flow rungs, empty/single-row/all-null edges,
validation `*_raises` derivation from declared params, `matches_reference`/`golden_master`, and the Properties
tier (fuzz, missing-data, scale) reusing `tests.support` strategies/asserts/tolerances unchanged.

## The migration map (153 functions)

| function | family | mixins | fields | policy (derived) |
|---|---|---|---|---|
| `absolute_price_oscillator` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `accumulation_distribution` | indicators | SeriesContract | — | BRIDGED / LATCHES |
| `accumulation_distribution_oscillator` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `adx` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `adxr` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `aroon` | indicators | WindowedContract + StructContract | up, down | IN_WINDOW_IS_NULL / PROPAGATES |
| `aroon_oscillator` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `atr` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `atr_normalized` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `awesome_oscillator` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `balance_of_power` | indicators | SeriesContract | — | PROPAGATES / PROPAGATES |
| `bollinger_bands` | indicators | WindowedContract + StructContract | lower, middle, upper | IN_WINDOW_IS_NULL / PROPAGATES |
| `cci` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `chaikin_money_flow` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `chande_momentum_oscillator` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `dema` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `di_minus` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `di_plus` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `dm_minus` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `dm_plus` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `dominant_cycle_period` | indicators | SeriesContract | — | LATCHES / LATCHES · golden-only |
| `dominant_cycle_phase` | indicators | SeriesContract | — | LATCHES / LATCHES · golden-only |
| `donchian_channels` | indicators | WindowedContract + StructContract | lower, middle, upper | IN_WINDOW_IS_NULL / PROPAGATES |
| `dx` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `ema` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `fisher_transform` | indicators | WindowedContract + StructContract | fisher, signal | IN_WINDOW_IS_NULL / PROPAGATES |
| `hilbert_phasor` | indicators | StructContract | in_phase, quadrature | LATCHES / LATCHES · golden-only |
| `hilbert_trendline` | indicators | SeriesContract | — | LATCHES / LATCHES · golden-only |
| `hma` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `ichimoku` | indicators | WindowedContract + StructContract | tenkan, kijun, senkou_a, senkou_b | IN_WINDOW_IS_NULL / PROPAGATES |
| `kama` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `keltner_channels` | indicators | WindowedContract + StructContract | lower, middle, upper | BRIDGED / LATCHES |
| `linear_regression` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `linear_regression_angle` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `linear_regression_intercept` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `linear_regression_slope` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `macd` | indicators | WindowedContract + StructContract | macd, signal, histogram | BRIDGED / LATCHES |
| `mama` | indicators | StructContract | mama, fama | LATCHES / LATCHES · golden-only |
| `midpoint` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `midprice` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `mom` | indicators | WindowedContract + SeriesContract | — | PROPAGATES / PROPAGATES |
| `money_flow_index` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `obv` | indicators | SeriesContract | — | BRIDGED / LATCHES |
| `parabolic_sar` | indicators | SeriesContract | — | PROPAGATES / PROPAGATES |
| `percentage_price_oscillator` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `price_average` | indicators | SeriesContract | — | PROPAGATES / PROPAGATES |
| `price_median` | indicators | SeriesContract | — | PROPAGATES / PROPAGATES |
| `price_typical` | indicators | SeriesContract | — | PROPAGATES / PROPAGATES |
| `price_weighted_close` | indicators | SeriesContract | — | PROPAGATES / PROPAGATES |
| `rma` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `roc` | indicators | WindowedContract + SeriesContract | — | PROPAGATES / PROPAGATES |
| `rsi` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `rsi_stochastic` | indicators | WindowedContract + StructContract | k, d | BRIDGED / LATCHES |
| `sine_wave` | indicators | StructContract | sine, lead_sine | LATCHES / LATCHES · golden-only |
| `sma` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `standard_deviation_ewma` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `standard_deviation_rolling` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `stochastic_fast` | indicators | WindowedContract + StructContract | k, d | IN_WINDOW_IS_NULL / PROPAGATES |
| `stochastic_slow` | indicators | WindowedContract + StructContract | k, d | IN_WINDOW_IS_NULL / PROPAGATES |
| `supertrend` | indicators | WindowedContract + StructContract | line, direction | BRIDGED / LATCHES |
| `t3` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `tema` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `time_series_forecast` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `trend_mode` | indicators | SeriesContract | — | LATCHES / LATCHES · golden-only |
| `trima` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `trix` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `true_range` | indicators | SeriesContract | — | ABSORBED / PROPAGATES |
| `ultimate_oscillator` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `variance_ewma` | indicators | WindowedContract + SeriesContract | — | BRIDGED / LATCHES |
| `variance_rolling` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `vortex` | indicators | WindowedContract + StructContract | plus, minus | IN_WINDOW_IS_NULL / PROPAGATES |
| `vwap` | indicators | SeriesContract | — | BRIDGED / LATCHES |
| `vwma` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `williams_r` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `wma` | indicators | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `adjusted_sharpe_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `alpha` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `alpha_rolling` | metrics | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `beta` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `beta_rolling` | metrics | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `burke_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `cagr` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `cagr_rolling` | metrics | WindowedContract + SeriesContract | — | PROPAGATES / PROPAGATES |
| `calmar_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `capture_downside_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `capture_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `capture_upside_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `common_sense_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `conditional_drawdown_at_risk` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `conditional_value_at_risk` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `downside_deviation` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `downside_deviation_rolling` | metrics | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `drawdown` | metrics | SeriesContract | — | PROPAGATES / PROPAGATES |
| `drawdown_rolling` | metrics | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `gain_to_pain_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `information_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `information_ratio_rolling` | metrics | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `kelly_criterion` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `kurtosis` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `kurtosis_rolling` | metrics | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `max_drawdown` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `max_drawdown_duration` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `modigliani_risk_adjusted_performance` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `omega_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `omega_ratio_rolling` | metrics | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `pain_index` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `pain_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `payoff_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `probabilistic_sharpe_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `profit_factor` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `recovery_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `risk_of_ruin` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `sharpe_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `sharpe_ratio_rolling` | metrics | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `skewness` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `skewness_rolling` | metrics | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `sortino_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `sortino_ratio_rolling` | metrics | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `stability` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `sterling_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `tail_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `tail_ratio_rolling` | metrics | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `total_return` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `total_return_rolling` | metrics | WindowedContract + SeriesContract | — | PROPAGATES / PROPAGATES |
| `treynor_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `treynor_ratio_rolling` | metrics | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `ulcer_index` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `ulcer_performance_ratio` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `value_at_risk` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `value_at_risk_modified` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `value_at_risk_parametric` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `value_at_risk_rolling` | metrics | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `volatility` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `volatility_rolling` | metrics | WindowedContract + SeriesContract | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `win_rate` | metrics | ReducingContract | — | SKIPPED / POISONS |
| `cost_borrow` | pnl | SeriesContract | — | PROPAGATES / PROPAGATES |
| `cost_fixed` | pnl | SeriesContract | — | PROPAGATES / PROPAGATES |
| `cost_funding` | pnl | SeriesContract | — | PROPAGATES / PROPAGATES |
| `cost_notional` | pnl | SeriesContract | — | PROPAGATES / PROPAGATES |
| `cost_per_share` | pnl | SeriesContract | — | PROPAGATES / PROPAGATES |
| `cost_proportional` | pnl | SeriesContract | — | PROPAGATES / PROPAGATES |
| `cost_slippage` | pnl | SeriesContract | — | PROPAGATES / PROPAGATES |
| `cumulative_pnl` | pnl | SeriesContract | — | BRIDGED / LATCHES |
| `dividend` | pnl | SeriesContract | — | PROPAGATES / PROPAGATES |
| `equity_curve` | pnl | SeriesContract | — | BRIDGED / LATCHES |
| `pnl_gross` | pnl | SeriesContract | — | PROPAGATES / PROPAGATES |
| `pnl_gross_inverse` | pnl | SeriesContract | — | PROPAGATES / PROPAGATES |
| `pnl_net` | pnl | SeriesContract | — | PROPAGATES / PROPAGATES |
| `returns_gross` | pnl | SeriesContract | — | PROPAGATES / PROPAGATES |
| `returns_log` | pnl | SeriesContract | — | PROPAGATES / PROPAGATES |
| `returns_net` | pnl | SeriesContract | — | PROPAGATES / PROPAGATES |
| `returns_simple` | pnl | SeriesContract | — | PROPAGATES / PROPAGATES |
| `turnover` | pnl | SeriesContract | — | PROPAGATES / PROPAGATES |
