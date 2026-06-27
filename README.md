# pomata

**A Polars-native quant toolkit — technical indicators, PnL accounting, and performance & risk metrics.** Each is a
composable `pl.Expr`, so an entire study is one lazy Polars pipeline, from price to performance.

And it doesn't ask you to trust its numbers — it **proves** them: every function is verified to the `float64` floor
against an independent reference, under 100% branch coverage.

[![CI](https://img.shields.io/github/actions/workflow/status/ilpomo/pomata/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/ilpomo/pomata/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/codecov/c/github/ilpomo/pomata?style=flat-square&label=coverage)](https://codecov.io/gh/ilpomo/pomata)
[![ruff](https://img.shields.io/badge/ruff-261230?style=flat-square&logo=ruff&logoColor=D7FF64)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/badge/ty-261230?style=flat-square&logo=ty&logoColor=D7FF64)](https://github.com/astral-sh/ty)
[![mypy](https://img.shields.io/badge/mypy-4B6BFB?style=flat-square)](https://www.mypy-lang.org)
[![pyright](https://img.shields.io/badge/pyright-4B6BFB?style=flat-square)](https://github.com/microsoft/pyright)
[![pyrefly](https://img.shields.io/badge/pyrefly-4B6BFB?style=flat-square)](https://pyrefly.org)

![Linux](https://img.shields.io/badge/Linux-505050?style=flat-square&logo=linux&logoColor=white)
![macOS](https://img.shields.io/badge/macOS-505050?style=flat-square&logo=apple&logoColor=white)
![Windows](https://custom-icon-badges.demolab.com/badge/Windows-505050.svg?style=flat-square&logo=windows11&logoColor=white)
[![python](https://img.shields.io/badge/python-3.12%20|%203.13%20|%203.14-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Polars](https://img.shields.io/badge/Polars-%E2%89%A51.40-CD792C?style=flat-square)](https://pola.rs)
[![license](https://img.shields.io/badge/license-MIT-750014?style=flat-square)](LICENSE)

> **Alpha.** The API is not frozen until `1.0`; expect refinement. The correctness bar holds at every commit regardless.

## From price to performance, in one query

Signal, PnL, and metrics are all plain `pl.Expr`, so an entire study is a single Polars pipeline — no glue code, no
DataFrame ping-pong, no second dependency between the steps:

```python
import polars as pl
from pomata.indicators import rsi
from pomata.pnl import returns_simple, returns_gross, returns_net, cost_proportional, equity_curve
from pomata.metrics import sharpe_ratio, max_drawdown

report = (
    frame  # a DataFrame (or LazyFrame) with a "close" column
    .with_columns(
        weight=(rsi(pl.col("close"), 14) < 30).cast(pl.Float64).shift(1),  # go long when oversold, act next bar
        asset_returns=returns_simple(pl.col("close")),
    )
    .with_columns(
        net=returns_net(
            returns_gross(pl.col("weight"), pl.col("asset_returns")),
            cost_proportional(pl.col("weight"), rate=0.001),
        ),
    )
    .select(
        sharpe=sharpe_ratio(pl.col("net"), periods_per_year=252),
        max_drawdown=max_drawdown(equity_curve(pl.col("net"))),
    )
)
```

The indicator feeds the signal, the signal feeds the PnL, the PnL feeds the metrics — every arrow is a `pl.Expr`, so it
all fuses into one Polars query (eager or lazy, single series or a multi-asset panel via `.over`). The `.shift(1)` is
the whole no-look-ahead story: a signal computed at the close acts on the next bar, by construction.

## Install

The only runtime dependency is **Polars**; Python **3.12+**.

```bash
# from source (today)
git clone https://github.com/ilpomo/pomata
cd pomata && uv sync
```

Once published to PyPI, the install will be `pip install pomata` (or `uv add pomata`).

Every function is a free-standing `pl.Expr` factory — name it, compose it, run it in any Polars context. Warm-up rows
are `null` until the window fills, never a fabricated value:

```python
import polars as pl
from pomata.indicators import rsi

frame = pl.DataFrame({"close": [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
                                45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 46.21, 46.25, 45.71, 46.45]})
frame.select(rsi(pl.col("close"), 14).round(2).alias("rsi"))["rsi"].to_list()
# [None, None, ..., 57.92, 62.88, 63.21, 56.01, 62.34]
```

## What's inside

Three families, one package. They share a grammar (pure `pl.Expr` factories, one canonical name per concept) and a
handoff: `pnl` emits exactly the return and equity series `metrics` consumes.

### indicators — 75 functions

The technical-analysis layer, each indicator a `pl.Expr` checked against TA-Lib to the `float64` floor — most bar-for-bar
from the first emitted value, a documented minority only on the converged tail (the differential tier is non-gating).
Multi-output indicators (`bollinger_bands`, `macd`, `stochastic_slow`, …) return a single `pl.Struct` —
pick a line with `.struct.field(...)` or expand with `.struct.unnest()`.

```python
from pomata.indicators import bollinger_bands
frame.select(bollinger_bands(pl.col("close"), 20).alias("bb")).unnest("bb")
```

<details><summary>All 75 indicators, by category</summary>

- **channel** (5) — `donchian_channels`, `ichimoku`, `keltner_channels`, `midpoint`, `midprice`
- **cycle** (7) — `dominant_cycle_period`, `dominant_cycle_phase`, `hilbert_phasor`, `hilbert_trendline`, `mama`, `sine_wave`, `trend_mode`
- **directional movement** (8) — `adx`, `adxr`, `di_minus`, `di_plus`, `dm_minus`, `dm_plus`, `dx`, `vortex`
- **momentum** (17) — `absolute_price_oscillator`, `aroon`, `aroon_oscillator`, `awesome_oscillator`, `balance_of_power`, `cci`, `chande_momentum_oscillator`, `fisher_transform`, `macd`, `mom`, `percentage_price_oscillator`, `roc`, `rsi`, `rsi_stochastic`, `trix`, `ultimate_oscillator`, `williams_r`
- **moving average** (11) — `dema`, `ema`, `hma`, `kama`, `rma`, `sma`, `t3`, `tema`, `trima`, `vwma`, `wma`
- **price transform** (4) — `price_average`, `price_median`, `price_typical`, `price_weighted_close`
- **statistic** (9) — `linear_regression`, `linear_regression_angle`, `linear_regression_intercept`, `linear_regression_slope`, `standard_deviation_ewma`, `standard_deviation_rolling`, `time_series_forecast`, `variance_ewma`, `variance_rolling`
- **stochastic** (2) — `stochastic_fast`, `stochastic_slow`
- **trend** (2) — `parabolic_sar`, `supertrend`
- **volatility** (4) — `atr`, `atr_normalized`, `bollinger_bands`, `true_range`
- **volume** (6) — `accumulation_distribution`, `accumulation_distribution_oscillator`, `chaikin_money_flow`, `money_flow_index`, `obv`, `vwap`

</details>

### pnl — 18 functions

Profit-and-loss accounting in two flows: **returns** (a signed `weight` of capital and asset returns) and **cash**
(a `quantity` of units and a price), with composable transaction-cost models, dividends, and inverse contracts. Every
degenerate input (`null` / `NaN` / `0` / `±inf` / warm-up) has a defined, documented, tested behavior.

```python
from pomata.pnl import returns_net, returns_gross, cost_proportional, equity_curve
gross = returns_gross(pl.col("weight"), pl.col("asset_returns"))
frame.select(equity_curve(returns_net(gross, cost_proportional(pl.col("weight"), rate=0.001))))
```

<details><summary>All 18 PnL functions</summary>

- **cash flow** — `cost_borrow`, `cost_fixed`, `cost_funding`, `cost_notional`, `cost_per_share`, `cumulative_pnl`, `dividend`, `pnl_gross`, `pnl_gross_inverse`, `pnl_net`
- **returns flow** — `cost_proportional`, `cost_slippage`, `equity_curve`, `returns_gross`, `returns_log`, `returns_net`, `returns_simple`, `turnover`

</details>

### metrics — 60 functions

Performance & risk statistics as reducing `pl.Expr`: point one at a return series (e.g. `pomata.pnl.returns_net`) or an
equity curve (e.g. `pomata.pnl.equity_curve`). Sharpe, Sortino, Calmar, drawdown, VaR/CVaR, capture, benchmark-relative
(alpha/beta/Treynor/information ratio), and a rolling twin for every windowed form.

```python
from pomata.metrics import sharpe_ratio, max_drawdown
frame.select(sharpe_ratio(pl.col("returns"), periods_per_year=252))
```

<details><summary>All 60 metrics</summary>

- **drawdown** — `conditional_drawdown_at_risk`, `drawdown`, `drawdown_rolling`, `max_drawdown`, `max_drawdown_duration`, `pain_index`, `ulcer_index`
- **performance** — `cagr`, `cagr_rolling`, `stability`, `total_return`, `total_return_rolling`
- **ratio** — `adjusted_sharpe_ratio`, `burke_ratio`, `calmar_ratio`, `common_sense_ratio`, `gain_to_pain_ratio`, `omega_ratio`, `omega_ratio_rolling`, `pain_ratio`, `probabilistic_sharpe_ratio`, `recovery_ratio`, `sharpe_ratio`, `sharpe_ratio_rolling`, `sortino_ratio`, `sortino_ratio_rolling`, `sterling_ratio`, `ulcer_performance_ratio`
- **relative** — `alpha`, `alpha_rolling`, `beta`, `beta_rolling`, `capture_downside_ratio`, `capture_ratio`, `capture_upside_ratio`, `information_ratio`, `information_ratio_rolling`, `modigliani_risk_adjusted_performance`, `treynor_ratio`, `treynor_ratio_rolling`
- **risk** — `conditional_value_at_risk`, `downside_deviation`, `downside_deviation_rolling`, `kelly_criterion`, `kurtosis`, `kurtosis_rolling`, `payoff_ratio`, `profit_ratio`, `risk_of_ruin`, `skewness`, `skewness_rolling`, `tail_ratio`, `tail_ratio_rolling`, `value_at_risk`, `value_at_risk_modified`, `value_at_risk_parametric`, `value_at_risk_rolling`, `volatility`, `volatility_rolling`, `win_rate`

</details>

## Correctness

**Verified, not asserted.** Every function is checked against an *independent* reference — a second code path that shares
nothing with the implementation — plus frozen golden-master values and property-based invariants, under **100% branch
coverage**. A function ships only when that suite is green.

For indicators there is also a public reference to meet: TA-Lib. Here is one figure to every digit a `float64` holds —
`rsi(14)`, the last value of a deterministic 400-bar series:

```text
pomata      85.20908701341023
reference   85.20908701341023   ← independent reimplementation: identical, to the last bit
TA-Lib      85.20908701341024   ← fifteen figures identical; differs only at the float64 floor
```

The same five indicators on the same series — most reproduce the reference *exactly*, the rest land at the noise floor:

| indicator | pomata | vs reimplementation | vs TA-Lib |
| --- | --- | :-: | :-: |
| `sma(20)` | `105.15146076264764` | exact | `1e-13` |
| `ema(20)` | `107.7299930892346` | `1e-13` | `1e-14` |
| `rsi(14)` | `85.20908701341023` | exact | `1e-14` |
| `atr(14)` | `1.904174462198776` | `9e-16` | `4e-15` |
| `macd(12,26,9)` | `2.523444380829531` | `1e-13` | `1e-14` |

The `pomata` and reference columns are pinned in the test suite; regenerate the full table — including the TA-Lib column
(which needs the optional `differential` dependency) — from a fresh clone with
`uv run --group differential python scripts/precision_table.py`.

`pnl` and `metrics` are proven on a different axis — every degenerate input has a defined behavior, matched against an
independent reference oracle — because their math is simple and their correctness lives at the edges, not in the digits.
The full method (the precision guarantee, the test-sizing derivations, exactly what is and is not claimed) is in
**[CORRECTNESS.md](CORRECTNESS.md)**.

## Where pomata fits

pomata is for the quant already working in Polars. Each function is a free-standing `pl.Expr` with `polars` as the only
runtime dependency, composable across eager, lazy, single-series, and grouped (`.over`) contexts — so the everyday
primitives live in one coherent toolkit instead of a wired-together stack.

It is vectorized analytics and accounting: indicators, total mark-to-market PnL, and metrics. It is **not** an execution
engine — no order fills, no event loop, no lot accounting.

## Project

- **Requirements** — Python ≥ 3.12, Polars ≥ 1.40.
- **Contributing** — see [CONTRIBUTING.md](CONTRIBUTING.md); the full gate (lint, three gating type checkers plus an
  advisory fourth, doctests, 100% branch coverage) runs on every commit.
- **License** — MIT, see [LICENSE](LICENSE).
- **Citation** — [CITATION.cff](CITATION.cff).
