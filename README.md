# pomata

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20980041.svg)](https://doi.org/10.5281/zenodo.20980041)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/13392/badge)](https://www.bestpractices.dev/projects/13392)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/ilpomo/pomata/badge)](https://securityscorecards.dev/viewer/?uri=github.com/ilpomo/pomata)

[![CI](https://img.shields.io/github/actions/workflow/status/ilpomo/pomata/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/ilpomo/pomata/actions/workflows/ci.yml)
[![codecov](https://img.shields.io/codecov/c/github/ilpomo/pomata?style=flat-square&label=codecov)](https://codecov.io/gh/ilpomo/pomata)
[![ruff](https://img.shields.io/badge/ruff-261230?style=flat-square&logo=ruff&logoColor=D7FF64)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/badge/ty-261230?style=flat-square&logo=ty&logoColor=D7FF64)](https://github.com/astral-sh/ty)
[![mypy](https://img.shields.io/badge/mypy-505050?style=flat-square)](https://www.mypy-lang.org)
[![pyright](https://img.shields.io/badge/pyright-505050?style=flat-square)](https://github.com/microsoft/pyright)
[![pyrefly](https://img.shields.io/badge/pyrefly-505050?style=flat-square)](https://pyrefly.org)

![Linux](https://img.shields.io/badge/Linux-505050?style=flat-square&logo=linux&logoColor=white)
![macOS](https://img.shields.io/badge/macOS-505050?style=flat-square&logo=apple&logoColor=white)
![Windows](https://custom-icon-badges.demolab.com/badge/Windows-505050.svg?style=flat-square&logo=windows11&logoColor=white)
[![python](https://img.shields.io/badge/python-3.12%20|%203.13%20|%203.14-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Polars](https://img.shields.io/badge/Polars-%E2%89%A51.40-CD792C?style=flat-square)](https://pola.rs)

**A Polars-native quant toolkit — technical indicators, PnL accounting, and performance & risk metrics.** Each is a
composable `pl.Expr`, so an entire study is one lazy Polars pipeline, from price to performance.

And it doesn't ask you to trust its numbers — it **proves** them: every function is verified to the `float64` floor
against an independent reference, under 100% branch coverage.

## Install

```bash
pip install pomata
# or
uv add pomata
```

From source:

```bash
git clone https://github.com/ilpomo/pomata
cd pomata && uv sync
```

## Dependencies

- **Runtime** — `polars` only (`>= 1.40`). Nothing else is pulled into your environment.
- **Python** — 3.12, 3.13, 3.14.
- **Optional** — the `differential` group (TA-Lib) powers the cross-reference parity tier; the other groups are the
  contributor gate. See [CONTRIBUTING.md](CONTRIBUTING.md).

## The data

Every snippet below runs on the same sample: a quarter of real daily bars for `AAPL`, `GOOG`, and `NVDA`, shipped with
these docs. Load it once — or point `read_parquet` at your own OHLCV frame:

```python
import polars as pl

ohlcv = pl.read_parquet("docs/_static/ohlcv_sample.parquet")
ohlcv.head(9)
```

```text
shape: (9, 7)
┌────────────────────────────────┬────────┬────────┬────────┬────────┬────────┬───────────┐
│ datetime                       ┆ ticker ┆ open   ┆ high   ┆ low    ┆ close  ┆ volume    │
│ ---                            ┆ ---    ┆ ---    ┆ ---    ┆ ---    ┆ ---    ┆ ---       │
│ datetime[μs, America/New_York] ┆ str    ┆ f64    ┆ f64    ┆ f64    ┆ f64    ┆ i64       │
╞════════════════════════════════╪════════╪════════╪════════╪════════╪════════╪═══════════╡
│ 2024-01-02 17:00:00 EST        ┆ AAPL   ┆ 185.06 ┆ 186.33 ┆ 181.83 ┆ 183.56 ┆ 82488700  │
│ 2024-01-02 17:00:00 EST        ┆ GOOG   ┆ 138.38 ┆ 139.39 ┆ 136.54 ┆ 138.34 ┆ 20071900  │
│ 2024-01-02 17:00:00 EST        ┆ NVDA   ┆ 49.16  ┆ 49.21  ┆ 47.51  ┆ 48.08  ┆ 411254000 │
│ 2024-01-03 17:00:00 EST        ┆ AAPL   ┆ 182.16 ┆ 183.8  ┆ 181.38 ┆ 182.19 ┆ 58414500  │
│ 2024-01-03 17:00:00 EST        ┆ GOOG   ┆ 137.39 ┆ 139.86 ┆ 137.22 ┆ 139.13 ┆ 18974300  │
│ 2024-01-03 17:00:00 EST        ┆ NVDA   ┆ 47.4   ┆ 48.1   ┆ 47.24  ┆ 47.48  ┆ 320896000 │
│ 2024-01-04 17:00:00 EST        ┆ AAPL   ┆ 180.11 ┆ 181.04 ┆ 178.86 ┆ 179.87 ┆ 71983600  │
│ 2024-01-04 17:00:00 EST        ┆ GOOG   ┆ 138.63 ┆ 139.41 ┆ 136.8  ┆ 136.83 ┆ 18253300  │
│ 2024-01-04 17:00:00 EST        ┆ NVDA   ┆ 47.68  ┆ 48.41  ┆ 47.42  ┆ 47.91  ┆ 306535000 │
└────────────────────────────────┴────────┴────────┴────────┴────────┴────────┴───────────┘
```

## Technical Indicators

75 indicators, each a `pl.Expr` you compute straight on your price frame — checked against **TA-Lib** to the `float64`
floor. On a multi-ticker panel, wrap the call in `.over("ticker")` so each symbol warms up on its own history (`null`
until the window fills, never a fabricated value):

```python
from pomata.indicators import rsi

ohlcv.with_columns(
    rsi=rsi(pl.col("close"), 14).over("ticker").round(2),
).select("datetime", "ticker", "close", "rsi").tail(9)
```

```text
shape: (9, 4)
┌────────────────────────────────┬────────┬────────┬───────┐
│ datetime                       ┆ ticker ┆ close  ┆ rsi   │
│ ---                            ┆ ---    ┆ ---    ┆ ---   │
│ datetime[μs, America/New_York] ┆ str    ┆ f64    ┆ f64   │
╞════════════════════════════════╪════════╪════════╪═══════╡
│ 2024-03-26 17:00:00 EDT        ┆ AAPL   ┆ 168.02 ┆ 37.64 │
│ 2024-03-26 17:00:00 EDT        ┆ GOOG   ┆ 150.37 ┆ 64.6  │
│ 2024-03-26 17:00:00 EDT        ┆ NVDA   ┆ 92.4   ┆ 65.66 │
│ 2024-03-27 17:00:00 EDT        ┆ AAPL   ┆ 171.59 ┆ 45.62 │
│ 2024-03-27 17:00:00 EDT        ┆ GOOG   ┆ 150.61 ┆ 64.95 │
│ 2024-03-27 17:00:00 EDT        ┆ NVDA   ┆ 90.09  ┆ 60.06 │
│ 2024-03-28 17:00:00 EDT        ┆ AAPL   ┆ 169.78 ┆ 42.64 │
│ 2024-03-28 17:00:00 EDT        ┆ GOOG   ┆ 150.93 ┆ 65.43 │
│ 2024-03-28 17:00:00 EDT        ┆ NVDA   ┆ 90.2   ┆ 60.24 │
└────────────────────────────────┴────────┴────────┴───────┘
```

Multi-output indicators (`bollinger_bands`, `macd`, `stochastic_slow`, …) return a single `pl.Struct` — pick a line
with `.struct.field(...)` or expand every line with `.struct.unnest()`.

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

## PnL Accounting

18 functions that turn a signal into money. An indicator becomes a signed `weight`; `returns_gross` /
`cost_proportional` / `returns_net` turn that into a costed return, and the `.shift(1)` on the signal is the whole
no-look-ahead story — a decision at the close acts on the next bar. Every degenerate input (`null` / `NaN` / `0` /
`±inf` / warm-up) has a defined, documented, tested behavior:

```python
from pomata.pnl import returns_simple, returns_gross, returns_net, cost_proportional

pnl = (
    ohlcv
    .with_columns(
        weight=(rsi(pl.col("close"), 14) > 50).cast(pl.Float64).shift(1).over("ticker"),
        asset_returns=returns_simple(pl.col("close")).over("ticker"),
    )
    .with_columns(
        net=returns_net(
            returns_gross(pl.col("weight"), pl.col("asset_returns")),
            cost_proportional(pl.col("weight"), rate=0.001).over("ticker"),
        ),
    )
)

pnl.select("datetime", "ticker", "weight", pl.col("net").round(4)).tail(9)
```

```text
shape: (9, 4)
┌────────────────────────────────┬────────┬────────┬─────────┐
│ datetime                       ┆ ticker ┆ weight ┆ net     │
│ ---                            ┆ ---    ┆ ---    ┆ ---     │
│ datetime[μs, America/New_York] ┆ str    ┆ f64    ┆ f64     │
╞════════════════════════════════╪════════╪════════╪═════════╡
│ 2024-03-26 17:00:00 EDT        ┆ AAPL   ┆ 0.0    ┆ -0.0    │
│ 2024-03-26 17:00:00 EDT        ┆ GOOG   ┆ 1.0    ┆ 0.0036  │
│ 2024-03-26 17:00:00 EDT        ┆ NVDA   ┆ 1.0    ┆ -0.0257 │
│ 2024-03-27 17:00:00 EDT        ┆ AAPL   ┆ 0.0    ┆ 0.0     │
│ 2024-03-27 17:00:00 EDT        ┆ GOOG   ┆ 1.0    ┆ 0.0016  │
│ 2024-03-27 17:00:00 EDT        ┆ NVDA   ┆ 1.0    ┆ -0.025  │
│ 2024-03-28 17:00:00 EDT        ┆ AAPL   ┆ 0.0    ┆ -0.0    │
│ 2024-03-28 17:00:00 EDT        ┆ GOOG   ┆ 1.0    ┆ 0.0021  │
│ 2024-03-28 17:00:00 EDT        ┆ NVDA   ┆ 1.0    ┆ 0.0012  │
└────────────────────────────────┴────────┴────────┴─────────┘
```

<details><summary>All 18 PnL functions</summary>

- **cash flow** — `cost_borrow`, `cost_fixed`, `cost_funding`, `cost_notional`, `cost_per_share`, `cumulative_pnl`, `dividend`, `pnl_gross`, `pnl_gross_inverse`, `pnl_net`
- **returns flow** — `cost_proportional`, `cost_slippage`, `equity_curve`, `returns_gross`, `returns_log`, `returns_net`, `returns_simple`, `turnover`

</details>

## Performance & Risk Metrics

60 reducing `pl.Expr` — point one at the net returns and it folds the whole history into the figure you report:
Sharpe, Sortino, Calmar, drawdown, VaR/CVaR, capture, benchmark-relative, and a rolling twin for every windowed form. A
`null` is skipped; a non-null `NaN` poisons the result loudly, rather than passing a plausible lie downstream:

```python
from pomata.pnl import equity_curve
from pomata.metrics import sharpe_ratio, total_return, max_drawdown

report = (
    pnl
    .group_by("ticker", maintain_order=True)
    .agg(
        sharpe=sharpe_ratio(pl.col("net"), periods_per_year=252).round(2),
        total_return=total_return(equity_curve(pl.col("net"))).round(4),
        max_drawdown=max_drawdown(equity_curve(pl.col("net"))).round(4),
    )
)

report
```

```text
shape: (3, 4)
┌────────┬────────┬──────────────┬──────────────┐
│ ticker ┆ sharpe ┆ total_return ┆ max_drawdown │
│ ---    ┆ ---    ┆ ---          ┆ ---          │
│ str    ┆ f64    ┆ f64          ┆ f64          │
╞════════╪════════╪══════════════╪══════════════╡
│ AAPL   ┆ -3.97  ┆ -0.0788      ┆ -0.0773      │
│ GOOG   ┆ -0.67  ┆ -0.0384      ┆ -0.1359      │
│ NVDA   ┆ 4.16   ┆ 0.4727       ┆ -0.087       │
└────────┴────────┴──────────────┴──────────────┘
```

<details><summary>All 60 metrics</summary>

- **drawdown** — `conditional_drawdown_at_risk`, `drawdown`, `drawdown_rolling`, `max_drawdown`, `max_drawdown_duration`, `pain_index`, `ulcer_index`
- **performance** — `cagr`, `cagr_rolling`, `stability`, `total_return`, `total_return_rolling`
- **ratio** — `adjusted_sharpe_ratio`, `burke_ratio`, `calmar_ratio`, `common_sense_ratio`, `gain_to_pain_ratio`, `omega_ratio`, `omega_ratio_rolling`, `pain_ratio`, `probabilistic_sharpe_ratio`, `recovery_ratio`, `sharpe_ratio`, `sharpe_ratio_rolling`, `sortino_ratio`, `sortino_ratio_rolling`, `sterling_ratio`, `ulcer_performance_ratio`
- **relative** — `alpha`, `alpha_rolling`, `beta`, `beta_rolling`, `capture_downside_ratio`, `capture_ratio`, `capture_upside_ratio`, `information_ratio`, `information_ratio_rolling`, `modigliani_risk_adjusted_performance`, `treynor_ratio`, `treynor_ratio_rolling`
- **risk** — `conditional_value_at_risk`, `downside_deviation`, `downside_deviation_rolling`, `kelly_criterion`, `kurtosis`, `kurtosis_rolling`, `payoff_ratio`, `profit_ratio`, `risk_of_ruin`, `skewness`, `skewness_rolling`, `tail_ratio`, `tail_ratio_rolling`, `value_at_risk`, `value_at_risk_modified`, `value_at_risk_parametric`, `value_at_risk_rolling`, `volatility`, `volatility_rolling`, `win_rate`

</details>

## The whole study, one lazy query

Every step above is a `pl.Expr`, so the four of them fuse into a single lazy pipeline — no intermediate frames, no glue,
no second dependency between the steps. Run it on `.lazy()` and `.collect()` the same three-row verdict:

```python
report = (
    ohlcv.lazy()
    .with_columns(
        weight=(rsi(pl.col("close"), 14) > 50).cast(pl.Float64).shift(1).over("ticker"),
        asset_returns=returns_simple(pl.col("close")).over("ticker"),
    )
    .with_columns(
        net=returns_net(
            returns_gross(pl.col("weight"), pl.col("asset_returns")),
            cost_proportional(pl.col("weight"), rate=0.001).over("ticker"),
        ),
    )
    .group_by("ticker", maintain_order=True)
    .agg(
        sharpe=sharpe_ratio(pl.col("net"), periods_per_year=252).round(2),
        total_return=total_return(equity_curve(pl.col("net"))).round(4),
        max_drawdown=max_drawdown(equity_curve(pl.col("net"))).round(4),
    )
    .collect()
)
```

```text
shape: (3, 4)
┌────────┬────────┬──────────────┬──────────────┐
│ ticker ┆ sharpe ┆ total_return ┆ max_drawdown │
│ ---    ┆ ---    ┆ ---          ┆ ---          │
│ str    ┆ f64    ┆ f64          ┆ f64          │
╞════════╪════════╪══════════════╪══════════════╡
│ AAPL   ┆ -3.97  ┆ -0.0788      ┆ -0.0773      │
│ GOOG   ┆ -0.67  ┆ -0.0384      ┆ -0.1359      │
│ NVDA   ┆ 4.16   ┆ 0.4727       ┆ -0.087       │
└────────┴────────┴──────────────┴──────────────┘
```

Same three numbers, arrived at as one query: the indicator fed the signal, the signal fed the PnL, the PnL fed the
metrics, and the optimizer fused the lot. Momentum paid in a quarter NVDA ran and cost a little where AAPL slid — the
`.over("ticker")` keeps a three-name panel, or a five-hundred-name one, equally separate.

## Correctness

**Verified, not asserted.** Every function is written twice: the shipped `pl.Expr`, and a second, independent *oracle*
that shares no code with it. The two must agree — on fixed series, frozen golden masters, and thousands of fuzzed
inputs, under **100% branch coverage** — or the build is red.

Each family is then held to the yardstick that catches its bugs: **indicators to the digit**, against the public TA-Lib 
reference; **PnL and metrics at the edges**, where every degenerate input has a defined, tested behavior.

The full account — the precision guarantee, the receipts, and exactly what is and is not claimed — is on the 
[trust page](https://ilpomo.github.io/pomata/trust.html) and in [CORRECTNESS.md](CORRECTNESS.md).

## Where pomata fits

pomata is for the quant already working in Polars. Each function is a free-standing `pl.Expr` with `polars` as the only
runtime dependency, composable across eager, lazy, single-series, and grouped (`.over`) contexts — so the everyday
primitives live in one coherent toolkit instead of a wired-together stack.

It is vectorized analytics and accounting: indicators, total mark-to-market PnL, and metrics. It is **not** an execution
engine — no order fills, no event loop, no lot accounting.

## Project

- **Contributing** — see [CONTRIBUTING.md](CONTRIBUTING.md); the full gate (lint, three gating type checkers plus an
  advisory fourth, doctests, 100% branch coverage) runs on every commit.
- **License** — MIT, see [LICENSE](LICENSE).
- **Citation** — [CITATION.cff](CITATION.cff).