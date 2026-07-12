# DESIGN — the spec ladder and the migration map

**Status:** the spec framework and its rungs are proven on the four structurally hardest functions (`ichimoku`,
`mama`, `sharpe_ratio`, `equity_curve`); the rollout proceeds family by family. The declaration surface below is
fixed by that proof.

## The shape of the framework

A per-function contract is a **frozen dataclass of pure data** — a `Spec` — and the rungs are **module-level
functions** parametrized over the specs they apply to. There is no metaprogramming: no metaclass, no
`__init_subclass__`, no runtime stamping of test functions. A declaration cannot lie by omission because the
`Spec` fields it would omit are either required by the language (no default) or made mandatory by a plain
`__post_init__`.

- **Spec per function** — one file per function, `tests_new/<family>/<name>.py`, data only (~20-35 lines),
  aggregated by explicit imports in `tests_new/all_specs.py`. A forgotten import is a red build (the bijection).
- **Rungs** — `tests_new/test_ladder.py`, each rung written once, `@pytest.mark.parametrize` over the applicable
  subset (a comprehension on declared fields), sub-parametrized where it reads better (per struct field, per
  validation counterexample, per scale axis).
- **The engine** — `tests_new/support/spec.py`: the frozen data types and the small engine the rungs delegate to
  (the deterministic probe frame, the expression builder, the lane readers, the oracle bridge, the fuzz strategies,
  the sizing helpers). It is the one module with the `disallow_any_explicit` deroga.

## The axes are declared fields, not a class hierarchy

The public surface varies along exactly three axes (11 observed combinations across 153 functions). In the spec
ladder each axis is a **declared field** or a **derived fact**, and a rung gates its own applicability by reading it:

- **shape** — `Shape.REDUCING` (43) · `Shape.SERIES` (95) · `Shape.STRUCT` (15): what one probe row observes. A
  struct declares its ordered `fields`, and every struct-aware rung reads **all** of them (never only the first).
- **windowed** — `warmup is not None` (75): the exact leading-null count under `params` — an `int`, or a per-field
  mapping for a struct whose lines warm up differently (e.g. `ichimoku`). A reduction and an unwindowed transform
  declare `warmup=None`.
- **null / NaN policy** — **not declared**: derived from `pomata._policy` by function name and dispatched inside
  the shared flow rungs, so a spec cannot pair the wrong behavior with the wrong declaration.

## The declaration surface (what a spec states — nothing else)

| Field | Required | Meaning |
|---|---|---|
| `factory` | yes | the `pl.Expr` factory under test |
| `inputs` | yes | ordered input column roles (drawn from the probe-frame vocabulary) |
| `params` | yes | the canonical scalar kwargs used by probes and goldens |
| `shape` | yes | `REDUCING` / `SERIES` / `STRUCT` — the observed output shape |
| `scale` | yes | a non-empty tuple of `ScaleAxis`, or a `ScaleExempt(reason)` — never an empty tuple |
| `oracle` | yes | the naive reference oracle |
| `golden_input` / `golden_output` | yes | the frozen golden master (per field for a struct) |
| `warmup` | optional | exact leading-null count under `params` (`int` / per-field mapping / `None`) |
| `fields` | struct | the struct's field names, in order |
| `raises` | params ⇒ yes | validation counterexamples: `(overrides, ValueError match)` |
| `golden_params` / `golden_round` | optional | the golden's own params and its rounding |
| `lands_on` | optional | landing column when it is not the first input |
| `flow_horizon` | optional | rows past a missing bar the flow must have played out by |
| `oracle_adapter` | optional | a frame->result callable when the oracle is not the factory's signature-mirror |
| `conditioning` | optional | a Hypothesis `assume` filter for the property tier |
| `all_null` | optional | a `Deviant(expected, reason)` when the all-null answer is not all-null |

Derived, never declared: `name` (from the factory), `family` (from `__all__`), `null_policy` / `nan_policy`
(from the registry), `spec_id` (the pytest id).

## The guarantees, all by construction

1. **Completeness of the language** — the required fields have no default, so a spec *cannot* be built without
   each one; no rung is ever silently skipped for want of a declaration.
2. **`__post_init__`** — the conditional requirements, checked loudly at construction (import time): a struct names
   its `fields`; a reduction has no `warmup`; declared `params` imply `raises` (else the validation rung is a
   no-op); `scale` is never an empty tuple (an exemption is a reasoned `ScaleExempt`); every scale axis and input
   role is real; the derived name has a declared policy and a public `__all__`.
3. **Two-way bijection** — `tests_new/all_specs.py` holds the per-family tuples in exact correspondence with
   `MIGRATED`, requires each migrated name to be in its family's `__all__`, and forbids duplicate names. It runs at
   import (born red), so any collection enforces it.
4. **Shape coverage guard** — one rung observes the output shape from the probe and asserts it is exactly the
   declared `shape`. (Windowedness is *not* observed: a seed-null that is not a warm-up would make that a false
   positive, so completeness rests on the required fields, not on inference.)

## The ladder (one function per rung, canonical order)

Contract — `returns_expr`, `output_lands_on_declared_column`, `shape_matches_declaration`, `lazy_eager_parity`,
`over_partitions_independently` (shape-aware: a reduction broadcasts across its group's rows, an elementwise output
concatenates; `assert_matches`, never a bit-equality), `bare_string_raises_type_error`.
Edge — `invalid_params_raise` (per counterexample), `all_null_input` (honoring an `all_null` `Deviant`),
`single_row`, `empty`, `interior_null_flow` / `interior_nan_flow` (the policy-dispatched flow, tail guards
included), `warmup_null_count` (windowed subset, per field for a struct), `no_lookahead` (non-reducing subset: a
prefix of the frame gives the prefix of the full output).
Correctness — `matches_reference`, `golden_master` (rounded expression-side).
Properties — `scale` (per `ScaleAxis`: scale only that axis's roles by a power of two, degree as declared),
`matches_reference_for_any_input` and `matches_reference_under_missing_data` (`@given(st.data())` inside
`@parametrize`, honoring a spec's `conditioning`).

## Sub-parametrized ids

A struct field, a validation counterexample, and a scale axis each get their own case with a readable id:
`ichimoku-senkou_b` (per-field warm-up), `sharpe_ratio-0` (per counterexample), `ichimoku-high+low` (per scale
axis). To read a failure: find the rung by the name in the id, read its few lines, then read the spec row the id
names.

## The migration map (153 functions)

Derived from the public surface (`shape` and `windowed` observed from HEAD, `policy` from the registry); `fields`
lists a struct's ordered lines.

| function | family | shape | windowed | fields | policy (derived) |
|---|---|---|---|---|---|
| `absolute_price_oscillator` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `accumulation_distribution` | indicators | SERIES | no | — | BRIDGED / LATCHES |
| `accumulation_distribution_oscillator` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `adx` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `adxr` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `aroon` | indicators | STRUCT | yes | up, down | IN_WINDOW_IS_NULL / PROPAGATES |
| `aroon_oscillator` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `atr` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `atr_normalized` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `awesome_oscillator` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `balance_of_power` | indicators | SERIES | no | — | PROPAGATES / PROPAGATES |
| `bollinger_bands` | indicators | STRUCT | yes | lower, middle, upper | IN_WINDOW_IS_NULL / PROPAGATES |
| `cci` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `chaikin_money_flow` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `chande_momentum_oscillator` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `dema` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `di_minus` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `di_plus` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `dm_minus` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `dm_plus` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `dominant_cycle_period` | indicators | SERIES | no | — | LATCHES / LATCHES · golden-only |
| `dominant_cycle_phase` | indicators | SERIES | no | — | LATCHES / LATCHES · golden-only |
| `donchian_channels` | indicators | STRUCT | yes | lower, middle, upper | IN_WINDOW_IS_NULL / PROPAGATES |
| `dx` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `ema` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `fisher_transform` | indicators | STRUCT | yes | fisher, signal | IN_WINDOW_IS_NULL / PROPAGATES |
| `hilbert_phasor` | indicators | STRUCT | no | in_phase, quadrature | LATCHES / LATCHES · golden-only |
| `hilbert_trendline` | indicators | SERIES | no | — | LATCHES / LATCHES · golden-only |
| `hma` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `ichimoku` | indicators | STRUCT | yes | tenkan, kijun, senkou_a, senkou_b | IN_WINDOW_IS_NULL / PROPAGATES |
| `kama` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `keltner_channels` | indicators | STRUCT | yes | lower, middle, upper | BRIDGED / LATCHES |
| `linear_regression` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `linear_regression_angle` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `linear_regression_intercept` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `linear_regression_slope` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `macd` | indicators | STRUCT | yes | macd, signal, histogram | BRIDGED / LATCHES |
| `mama` | indicators | STRUCT | yes | mama, fama | LATCHES / LATCHES · golden-only |
| `midpoint` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `midprice` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `mom` | indicators | SERIES | yes | — | PROPAGATES / PROPAGATES |
| `money_flow_index` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `obv` | indicators | SERIES | no | — | BRIDGED / LATCHES |
| `parabolic_sar` | indicators | SERIES | no | — | PROPAGATES / PROPAGATES |
| `percentage_price_oscillator` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `price_average` | indicators | SERIES | no | — | PROPAGATES / PROPAGATES |
| `price_median` | indicators | SERIES | no | — | PROPAGATES / PROPAGATES |
| `price_typical` | indicators | SERIES | no | — | PROPAGATES / PROPAGATES |
| `price_weighted_close` | indicators | SERIES | no | — | PROPAGATES / PROPAGATES |
| `rma` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `roc` | indicators | SERIES | yes | — | PROPAGATES / PROPAGATES |
| `rsi` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `rsi_stochastic` | indicators | STRUCT | yes | k, d | BRIDGED / LATCHES |
| `sine_wave` | indicators | STRUCT | no | sine, lead_sine | LATCHES / LATCHES · golden-only |
| `sma` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `standard_deviation_ewma` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `standard_deviation_rolling` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `stochastic_fast` | indicators | STRUCT | yes | k, d | IN_WINDOW_IS_NULL / PROPAGATES |
| `stochastic_slow` | indicators | STRUCT | yes | k, d | IN_WINDOW_IS_NULL / PROPAGATES |
| `supertrend` | indicators | STRUCT | yes | line, direction | BRIDGED / LATCHES |
| `t3` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `tema` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `time_series_forecast` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `trend_mode` | indicators | SERIES | no | — | LATCHES / LATCHES · golden-only |
| `trima` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `trix` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `true_range` | indicators | SERIES | no | — | ABSORBED / PROPAGATES |
| `ultimate_oscillator` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `variance_ewma` | indicators | SERIES | yes | — | BRIDGED / LATCHES |
| `variance_rolling` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `vortex` | indicators | STRUCT | yes | plus, minus | IN_WINDOW_IS_NULL / PROPAGATES |
| `vwap` | indicators | SERIES | no | — | BRIDGED / LATCHES |
| `vwma` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `williams_r` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `wma` | indicators | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `adjusted_sharpe_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `alpha` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `alpha_rolling` | metrics | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `beta` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `beta_rolling` | metrics | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `burke_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `cagr` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `cagr_rolling` | metrics | SERIES | yes | — | PROPAGATES / PROPAGATES |
| `calmar_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `capture_downside_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `capture_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `capture_upside_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `common_sense_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `conditional_drawdown_at_risk` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `conditional_value_at_risk` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `downside_deviation` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `downside_deviation_rolling` | metrics | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `drawdown` | metrics | SERIES | no | — | PROPAGATES / PROPAGATES |
| `drawdown_rolling` | metrics | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `gain_to_pain_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `information_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `information_ratio_rolling` | metrics | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `kelly_criterion` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `kurtosis` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `kurtosis_rolling` | metrics | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `max_drawdown` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `max_drawdown_duration` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `modigliani_risk_adjusted_performance` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `omega_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `omega_ratio_rolling` | metrics | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `pain_index` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `pain_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `payoff_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `probabilistic_sharpe_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `profit_factor` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `recovery_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `risk_of_ruin` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `sharpe_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `sharpe_ratio_rolling` | metrics | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `skewness` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `skewness_rolling` | metrics | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `sortino_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `sortino_ratio_rolling` | metrics | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `stability` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `sterling_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `tail_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `tail_ratio_rolling` | metrics | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `total_return` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `total_return_rolling` | metrics | SERIES | yes | — | PROPAGATES / PROPAGATES |
| `treynor_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `treynor_ratio_rolling` | metrics | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `ulcer_index` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `ulcer_performance_ratio` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `value_at_risk` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `value_at_risk_modified` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `value_at_risk_parametric` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `value_at_risk_rolling` | metrics | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `volatility` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `volatility_rolling` | metrics | SERIES | yes | — | IN_WINDOW_IS_NULL / PROPAGATES |
| `win_rate` | metrics | REDUCING | no | — | SKIPPED / POISONS |
| `cost_borrow` | pnl | SERIES | no | — | PROPAGATES / PROPAGATES |
| `cost_fixed` | pnl | SERIES | no | — | PROPAGATES / PROPAGATES |
| `cost_funding` | pnl | SERIES | no | — | PROPAGATES / PROPAGATES |
| `cost_notional` | pnl | SERIES | no | — | PROPAGATES / PROPAGATES |
| `cost_per_share` | pnl | SERIES | no | — | PROPAGATES / PROPAGATES |
| `cost_proportional` | pnl | SERIES | no | — | PROPAGATES / PROPAGATES |
| `cost_slippage` | pnl | SERIES | no | — | PROPAGATES / PROPAGATES |
| `cumulative_pnl` | pnl | SERIES | no | — | BRIDGED / LATCHES |
| `dividend` | pnl | SERIES | no | — | PROPAGATES / PROPAGATES |
| `equity_curve` | pnl | SERIES | no | — | BRIDGED / LATCHES |
| `pnl_gross` | pnl | SERIES | no | — | PROPAGATES / PROPAGATES |
| `pnl_gross_inverse` | pnl | SERIES | no | — | PROPAGATES / PROPAGATES |
| `pnl_net` | pnl | SERIES | no | — | PROPAGATES / PROPAGATES |
| `returns_gross` | pnl | SERIES | no | — | PROPAGATES / PROPAGATES |
| `returns_log` | pnl | SERIES | no | — | PROPAGATES / PROPAGATES |
| `returns_net` | pnl | SERIES | no | — | PROPAGATES / PROPAGATES |
| `returns_simple` | pnl | SERIES | no | — | PROPAGATES / PROPAGATES |
| `turnover` | pnl | SERIES | no | — | PROPAGATES / PROPAGATES |
