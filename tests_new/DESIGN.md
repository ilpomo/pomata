# DESIGN — the contract framework and the migration map

**Status:** the declaration surface is **frozen**: it has been proven end to end on the four structurally
hardest functions (`ichimoku`, `mama`, `sharpe_ratio`, `equity_curve` — see their contracts). The rollout
proceeds family by family; `MIGRATED` in `test_surface.py` tracks the landed set.

## The axes

The public surface varies along exactly three axes (probed from HEAD, 11 observed combinations across 153
functions), and the framework composes along them:

- **shape** — `ContractReducing` (43) | `ContractSeries` (90) | `ContractStruct` (15+5): what one probe row
  observes; struct contracts declare their `fields` and every rung reads **all** of them.
- **windowed** — `ContractWindowed` (74): declares the exact `warmup` (an `int`, or a per-field mapping for
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
| `raises` | with `params` | validation counterexamples: (kwargs override, ValueError match) — never empty when `params` is |
| `oracle` | correctness | the naive reference; signature-mirror call by default, `_reference` hook for deviants |
| `golden_input` / `golden_output` | correctness | the frozen golden master (per field for structs), rounded via `golden_round` |
| `golden_params` | optional | the golden's own parameters where they differ from the canonical `params` |
| `lands_on` | optional | landing column when it is not the first input (11 known exceptions) |
| `flow_horizon` | optional | override of the null/NaN-flow recovery horizon (displaced outputs) |
| `override_ok` | optional | visible consent to redefine an inherited rung (empty by default) |

Derived, never declared: `name` (from the factory), `family` (from `__all__`), `null_policy`/`nan_policy`
(from the registry). Hypothesis rungs are stamped per concrete class by the machinery (one shared `@given`
function object would trip `differing_executors` and cross-contaminate the example database). The scale and
large-magnitude property rungs land with the family rollout, reusing `tests.support` asserts unchanged.

A warm-up-bearing function without a window parameter (the cycle cluster: constant settling warm-up) composes
`ContractWindowed` all the same — the declaration is the warm-up, not the window.

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
| `absolute_price_oscillator` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `accumulation_distribution` | indicators | ContractSeries | — | BRIDGED / LATCHES |
| `accumulation_distribution_oscillator` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `adx` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `adxr` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `aroon` | indicators | ContractWindowed + ContractStruct | up, down | IN_WINDOW_IS_NULL / PROPAGATES |
| `aroon_oscillator` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `atr` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `atr_normalized` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `awesome_oscillator` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `balance_of_power` | indicators | ContractSeries | — | PROPAGATES / PROPAGATES |
| `bollinger_bands` | indicators | ContractWindowed + ContractStruct | lower, middle, upper | IN_WINDOW_IS_NULL / PROPAGATES |
| `cci` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `chaikin_money_flow` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `chande_momentum_oscillator` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `dema` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `di_minus` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `di_plus` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `dm_minus` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `dm_plus` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `dominant_cycle_period` | indicators | ContractWindowed + ContractSeries | — | LATCHES / LATCHES · golden-only |
| `dominant_cycle_phase` | indicators | ContractWindowed + ContractSeries | — | LATCHES / LATCHES · golden-only |
| `donchian_channels` | indicators | ContractWindowed + ContractStruct | lower, middle, upper | IN_WINDOW_IS_NULL / PROPAGATES |
| `dx` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `ema` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `fisher_transform` | indicators | ContractWindowed + ContractStruct | fisher, signal | IN_WINDOW_IS_NULL / PROPAGATES |
| `hilbert_phasor` | indicators | ContractWindowed + ContractStruct | in_phase, quadrature | LATCHES / LATCHES · golden-only |
| `hilbert_trendline` | indicators | ContractWindowed + ContractSeries | — | LATCHES / LATCHES · golden-only |
| `hma` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `ichimoku` | indicators | ContractWindowed + ContractStruct | tenkan, kijun, senkou_a, senkou_b | IN_WINDOW_IS_NULL / PROPAGATES |
| `kama` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `keltner_channels` | indicators | ContractWindowed + ContractStruct | lower, middle, upper | BRIDGED / LATCHES |
| `linear_regression` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `linear_regression_angle` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `linear_regression_intercept` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `linear_regression_slope` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `macd` | indicators | ContractWindowed + ContractStruct | macd, signal, histogram | BRIDGED / LATCHES |
| `mama` | indicators | ContractWindowed + ContractStruct | mama, fama | LATCHES / LATCHES · golden-only |
| `midpoint` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `midprice` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `mom` | indicators | ContractWindowed + ContractSeries | — | PROPAGATES / PROPAGATES |
| `money_flow_index` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `obv` | indicators | ContractSeries | — | BRIDGED / LATCHES |
| `parabolic_sar` | indicators | ContractSeries | — | PROPAGATES / PROPAGATES |
| `percentage_price_oscillator` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `price_average` | indicators | ContractSeries | — | PROPAGATES / PROPAGATES |
| `price_median` | indicators | ContractSeries | — | PROPAGATES / PROPAGATES |
| `price_typical` | indicators | ContractSeries | — | PROPAGATES / PROPAGATES |
| `price_weighted_close` | indicators | ContractSeries | — | PROPAGATES / PROPAGATES |
| `rma` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `roc` | indicators | ContractWindowed + ContractSeries | — | PROPAGATES / PROPAGATES |
| `rsi` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `rsi_stochastic` | indicators | ContractWindowed + ContractStruct | k, d | BRIDGED / LATCHES |
| `sine_wave` | indicators | ContractWindowed + ContractStruct | sine, lead_sine | LATCHES / LATCHES · golden-only |
| `sma` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `standard_deviation_ewma` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `standard_deviation_rolling` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `stochastic_fast` | indicators | ContractWindowed + ContractStruct | k, d | IN_WINDOW_IS_NULL / PROPAGATES |
| `stochastic_slow` | indicators | ContractWindowed + ContractStruct | k, d | IN_WINDOW_IS_NULL / PROPAGATES |
| `supertrend` | indicators | ContractWindowed + ContractStruct | line, direction | BRIDGED / LATCHES |
| `t3` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `tema` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `time_series_forecast` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `trend_mode` | indicators | ContractWindowed + ContractSeries | — | LATCHES / LATCHES · golden-only |
| `trima` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `trix` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `true_range` | indicators | ContractSeries | — | ABSORBED / PROPAGATES |
| `ultimate_oscillator` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `variance_ewma` | indicators | ContractWindowed + ContractSeries | — | BRIDGED / LATCHES |
| `variance_rolling` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `vortex` | indicators | ContractWindowed + ContractStruct | plus, minus | IN_WINDOW_IS_NULL / PROPAGATES |
| `vwap` | indicators | ContractSeries | — | BRIDGED / LATCHES |
| `vwma` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `williams_r` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `wma` | indicators | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `adjusted_sharpe_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `alpha` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `alpha_rolling` | metrics | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `beta` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `beta_rolling` | metrics | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `burke_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `cagr` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `cagr_rolling` | metrics | ContractWindowed + ContractSeries | — | PROPAGATES / PROPAGATES |
| `calmar_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `capture_downside_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `capture_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `capture_upside_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `common_sense_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `conditional_drawdown_at_risk` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `conditional_value_at_risk` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `downside_deviation` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `downside_deviation_rolling` | metrics | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `drawdown` | metrics | ContractSeries | — | PROPAGATES / PROPAGATES |
| `drawdown_rolling` | metrics | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `gain_to_pain_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `information_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `information_ratio_rolling` | metrics | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `kelly_criterion` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `kurtosis` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `kurtosis_rolling` | metrics | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `max_drawdown` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `max_drawdown_duration` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `modigliani_risk_adjusted_performance` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `omega_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `omega_ratio_rolling` | metrics | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `pain_index` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `pain_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `payoff_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `probabilistic_sharpe_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `profit_factor` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `recovery_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `risk_of_ruin` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `sharpe_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `sharpe_ratio_rolling` | metrics | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `skewness` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `skewness_rolling` | metrics | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `sortino_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `sortino_ratio_rolling` | metrics | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `stability` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `sterling_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `tail_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `tail_ratio_rolling` | metrics | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `total_return` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `total_return_rolling` | metrics | ContractWindowed + ContractSeries | — | PROPAGATES / PROPAGATES |
| `treynor_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `treynor_ratio_rolling` | metrics | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `ulcer_index` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `ulcer_performance_ratio` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `value_at_risk` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `value_at_risk_modified` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `value_at_risk_parametric` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `value_at_risk_rolling` | metrics | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `volatility` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `volatility_rolling` | metrics | ContractWindowed + ContractSeries | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `win_rate` | metrics | ContractReducing | — | SKIPPED / POISONS |
| `cost_borrow` | pnl | ContractSeries | — | PROPAGATES / PROPAGATES |
| `cost_fixed` | pnl | ContractSeries | — | PROPAGATES / PROPAGATES |
| `cost_funding` | pnl | ContractSeries | — | PROPAGATES / PROPAGATES |
| `cost_notional` | pnl | ContractSeries | — | PROPAGATES / PROPAGATES |
| `cost_per_share` | pnl | ContractSeries | — | PROPAGATES / PROPAGATES |
| `cost_proportional` | pnl | ContractSeries | — | PROPAGATES / PROPAGATES |
| `cost_slippage` | pnl | ContractSeries | — | PROPAGATES / PROPAGATES |
| `cumulative_pnl` | pnl | ContractSeries | — | BRIDGED / LATCHES |
| `dividend` | pnl | ContractSeries | — | PROPAGATES / PROPAGATES |
| `equity_curve` | pnl | ContractSeries | — | BRIDGED / LATCHES |
| `pnl_gross` | pnl | ContractSeries | — | PROPAGATES / PROPAGATES |
| `pnl_gross_inverse` | pnl | ContractSeries | — | PROPAGATES / PROPAGATES |
| `pnl_net` | pnl | ContractSeries | — | PROPAGATES / PROPAGATES |
| `returns_gross` | pnl | ContractSeries | — | PROPAGATES / PROPAGATES |
| `returns_log` | pnl | ContractSeries | — | PROPAGATES / PROPAGATES |
| `returns_net` | pnl | ContractSeries | — | PROPAGATES / PROPAGATES |
| `returns_simple` | pnl | ContractSeries | — | PROPAGATES / PROPAGATES |
| `turnover` | pnl | ContractSeries | — | PROPAGATES / PROPAGATES |
