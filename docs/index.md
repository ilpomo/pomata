# pomata

**A Polars-native quant toolkit: technical indicators, PnL accounting, and performance & risk metrics.** Every public
function is a composable `pl.Expr`, so an entire study is one lazy Polars pipeline, from price to performance.

And it doesn't ask you to trust its numbers — it **proves** them: each function is verified to the `float64` floor
against an independent reference, under 100% branch coverage.

:::{admonition} Alpha
:class: note
The API is not frozen until `1.0` — expect refinement. The correctness bar holds at every commit regardless.
:::

## Three families, one grammar

| Family | n | What it covers |
| --- | --- | --- |
| **`pomata.indicators`** | 75 | moving averages, momentum, volatility, channels, cycles, directional movement, … |
| **`pomata.pnl`** | 18 | returns & cash-flow accounting, transaction costs, dividends, equity curves |
| **`pomata.metrics`** | 60 | Sharpe Ratio/Sortino Ratio/Calmar Ratio, drawdown, VaR/CVaR, capture, benchmark-relative, … |

They share a grammar — pure `pl.Expr` factories, one canonical name per concept — and a handoff: `pnl` emits exactly
the return and equity series that `metrics` consumes.

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
        weight=(rsi(pl.col("close"), 14) < 30).cast(pl.Float64).shift(1),  # long when oversold, act next bar
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

The indicator feeds the signal, the signal feeds the PnL, the PnL feeds the metrics — every arrow is a `pl.Expr`, so
it all fuses into one Polars query (eager or lazy, a single series or a multi-asset panel via `.over`). The `.shift(1)`
is the whole no-look-ahead story: a signal computed at the close acts on the next bar, by construction.

## Start here

- **{doc}`installation`** — one runtime dependency (Polars), Python 3.12+.
- **{doc}`concepts`** — the five ideas that make everything compose.
- **{doc}`tutorial`** — raw bars to a costed, look-ahead-free report, end to end.
- **{doc}`trust`** — what pomata tests, and the precision it is confident to guarantee.
- **The families** — {doc}`indicators <families/indicators>` · {doc}`pnl <families/pnl>` · {doc}`metrics <families/metrics>`: the catalogs, and how `pomata` solves the classic pains.
- **{doc}`API reference <api/index>`** — every function, generated from the docstrings.
- **Guides** — {doc}`faq` · {doc}`troubleshooting` · {doc}`versioning`.

```{toctree}
:hidden:
:caption: Getting started

installation
concepts
tutorial
```

```{toctree}
:hidden:
:caption: Trust & families

trust
families/indicators
families/pnl
families/metrics
```

```{toctree}
:hidden:
:caption: Guides

faq
troubleshooting
versioning
```

```{toctree}
:hidden:
:caption: Reference

api/index
project
```
