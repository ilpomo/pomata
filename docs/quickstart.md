# Quickstart

One function first, then the smallest complete study — signal, costs, verdict — on a single price series. The
{doc}`tutorial` runs a full study on a three-ticker panel and explains each step.

## Install

```bash
uv add pomata          # or: pip install pomata
```

Python 3.12+, one runtime dependency (Polars). Details and options in {doc}`installation`.

## A first indicator

Every `pomata` function returns a Polars expression — you run it wherever an expression runs. The bars are the
sample shipped in the repository (clone it and run from its root, or point `read_parquet` at your own frame):

```{doctest}
>>> import polars as pl
>>> from pomata.indicators import rsi
>>>
>>> ohlcv = pl.read_parquet("docs/_static/ohlcv_sample.parquet").filter(pl.col("ticker") == "AAPL")
>>> ohlcv.select(rsi=rsi(pl.col("close"), 14).round(2)).tail(3)
shape: (3, 1)
┌───────┐
│ rsi   │
│ ---   │
│ f64   │
╞═══════╡
│ 37.64 │
│ 45.62 │
│ 42.64 │
└───────┘
```

With fewer bars than the window, every output is `null` — the warm-up. No seed is fabricated for an unfilled
window; {doc}`design` (idea 4) is the canonical treatment:

```{doctest}
>>> pl.DataFrame({"close": [10.0, 10.5, 10.2]}).select(rsi=rsi(pl.col("close"), 14))
shape: (3, 1)
┌──────┐
│ rsi  │
│ ---  │
│ f64  │
╞══════╡
│ null │
│ null │
│ null │
└──────┘
```

One more contract fact before you feed it your own data: rows must already be sorted oldest-first — nothing
re-sorts for you. The {doc}`tutorial` shows that failing, silently.

## From price to performance, in one query

Signal, PnL, and metrics are all plain `pl.Expr`, so the entire study fits in one Polars query:

```{doctest}
>>> from pomata.pnl import returns_simple, returns_gross, returns_net, cost_proportional, equity_curve
>>> from pomata.metrics import sharpe_ratio, max_drawdown
>>>
>>> (
...     ohlcv  # a DataFrame (or LazyFrame) with a "close" column
...     .with_columns(
...         weight=(rsi(pl.col("close"), 14) < 30).cast(pl.Float64).shift(1),  # long when oversold, act next bar
...         asset_returns=returns_simple(pl.col("close")),
...     )
...     .with_columns(
...         net=returns_net(
...             returns_gross(pl.col("weight"), pl.col("asset_returns")),
...             cost_proportional(pl.col("weight"), rate=0.001),
...         ),
...     )
...     .select(
...         sharpe=sharpe_ratio(pl.col("net"), periods_per_year=252).round(4),
...         max_drawdown=max_drawdown(equity_curve(pl.col("net"))).round(4),
...     )
... )
shape: (1, 2)
┌─────────┬──────────────┐
│ sharpe  ┆ max_drawdown │
│ ---     ┆ ---          │
│ f64     ┆ f64          │
╞═════════╪══════════════╡
│ -1.0372 ┆ -0.0358      │
└─────────┴──────────────┘
```

Every arrow in the chain you just ran is a `pl.Expr` feeding the next:

```text
┌────────┐  rsi(14) < 30  ┌────────┐  gross − costs  ┌───────┐  sharpe · drawdown  ┌─────────┐
│ prices │ ─────────────► │ signal │ ──────────────► │  PnL  │ ──────────────────► │ verdict │
└────────┘   indicators   └────────┘       pnl       └───────┘       metrics       └─────────┘
```

One family per arrow: {py:func}`~pomata.indicators.rsi` builds the signal, {py:func}`~pomata.pnl.returns_net`
with {py:func}`~pomata.pnl.cost_proportional` prices it, {py:func}`~pomata.metrics.sharpe_ratio` and
{py:func}`~pomata.metrics.max_drawdown` judge it.

This page used three of the six ideas in {doc}`design` — the shared expression grammar, the warm-up `null`, and the
explicit `.shift(1)` timing; a multi-ticker frame adds a fourth, `.over` partitioning. That page states and proves
all six.

## Where next

- {doc}`design` — the six ideas, each stated and proven.
- {doc}`tutorial` — the full three-ticker walkthrough: data shapes, the input contract, the anatomy of costs.
- {doc}`correctness` and {doc}`benchmarks/index` — what is proven, and how fast it runs.
- {doc}`API Reference <api/index>` — every function, generated from the docstrings.
