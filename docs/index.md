# pomata

**A Polars-native quant toolkit: technical indicators, PnL accounting, and performance & risk metrics.** Every public
function is a composable `pl.Expr`, so an entire study — indicator, position, costs, verdict — fuses into a single
Polars query, eager or lazy, single series or multi-asset panel.

Here is the whole flow on a quarter of real daily bars for `AAPL`, `GOOG`, and `NVDA` — price to signal to net
returns to a per-ticker verdict, in one lazy pipeline with no intermediate frames:

```{doctest}
>>> import polars as pl
>>> from pomata.indicators import rsi
>>> from pomata.pnl import returns_simple, returns_gross, returns_net, cost_proportional, equity_curve
>>> from pomata.metrics import sharpe_ratio, total_return, max_drawdown
>>>
>>> ohlcv = pl.read_parquet("docs/_static/ohlcv_sample.parquet")  # the sample checked into the repo
>>> (
...     ohlcv.lazy()
...     .with_columns(
...         weight=(rsi(pl.col("close"), 14) > 50).cast(pl.Float64).shift(1).over("ticker"),
...         asset_returns=returns_simple(pl.col("close")).over("ticker"),
...     )
...     .with_columns(
...         net=returns_net(
...             returns_gross(pl.col("weight"), pl.col("asset_returns")),
...             cost_proportional(pl.col("weight"), rate=0.001).over("ticker"),
...         ),
...     )
...     .group_by("ticker", maintain_order=True)
...     .agg(
...         sharpe=sharpe_ratio(pl.col("net"), periods_per_year=252).round(2),
...         total_return=total_return(equity_curve(pl.col("net"))).round(4),
...         max_drawdown=max_drawdown(equity_curve(pl.col("net"))).round(4),
...     )
...     .collect()
... )
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

```bash
uv add pomata          # or: pip install pomata
```

The table was not typed in: every example on this site runs in CI, and the output you read is the output that ran.

Two claims, both proven rather than asserted:

- **{doc}`Correct, verifiably <correctness>`** — every function is written twice: the shipped Polars form and its
  oracle (an independent, deliberately naive reference) must agree on fixed series, frozen golden masters, and
  thousands of fuzzed inputs, under 100% branch coverage, or the build is red.
- **{doc}`Fast, measurably <benchmarks/index>`** — every function is benchmarked against a native `rolling_mean`
  anchor and its own oracle: at 100,000 rows the median function costs `1.9×` the anchor, and a nightly guard holds
  the whole surface to its **O(n)** scaling contract.

## Three families, one grammar

| Family | n | What it covers |
| --- | --- | --- |
| **`pomata.indicators`** | 75 | moving averages, momentum, volatility, channels, cycles, directional movement, … |
| **`pomata.pnl`** | 18 | returns & cash-flow accounting, transaction costs, dividends, equity curves |
| **`pomata.metrics`** | 60 | risk-adjusted ratios, drawdown, performance, benchmark-relative, tail risk, … |

They share a grammar — pure `pl.Expr` factories, one canonical name per concept — and a handoff: `pnl` emits exactly
the return and equity series that `metrics` consumes.

## Where pomata fits

`pomata` is for the quant already working in Polars: vectorized analytics and accounting — indicators, total
mark-to-market PnL, and metrics — with `polars` as the only runtime dependency.

It is **not** an execution engine: no order fills, no event loop, no lot accounting. Nor is it a charting or
reporting layer: no candlestick patterns, no plots, no tearsheets — `pomata` computes the numbers; presentation
stays in your stack.

## Start here

- **{doc}`installation`** — one runtime dependency (Polars), Python 3.12+. Assumes nothing.
- **{doc}`quickstart`** — one function, then the smallest complete study on a single series. Assumes you know Polars basics.
- **{doc}`design`** — the six ideas that make everything compose, each proven with a runnable demo.
- **{doc}`tutorial`** — the three-ticker walk, every decision explained: data shapes, the input contract, costs.
- **The families** — {doc}`indicators <families/indicators>` · {doc}`pnl <families/pnl>` ·
  {doc}`metrics <families/metrics>`: each family's catalog and its conventions, for choosing the right function.
- **{doc}`correctness`** and **{doc}`benchmarks <benchmarks/index>`** — the proof pages: how every number is
  verified, and how fast it runs. Read these to decide whether to trust `pomata`.
- **{doc}`API Reference <api/index>`** — every function's exact contract, generated from the docstrings. Assumes you
  know which function you need — the families pages route you there.

```{toctree}
:hidden:
:caption: GET STARTED

installation
quickstart
design
tutorial
```

```{toctree}
:hidden:
:caption: FAMILIES

families/indicators
families/pnl
families/metrics
```

```{toctree}
:hidden:
:caption: VERIFICATION

correctness
benchmarks/index
```

```{toctree}
:hidden:
:caption: API REFERENCE

api/indicators
api/pnl
api/metrics
```

```{toctree}
:hidden:
:caption: PROJECT

project
```
