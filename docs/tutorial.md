# Tutorial

You have a table of prices and a strategy. You want a single number that says whether the strategy would have made
money — honestly, after costs, without cheating on timing. This walks the whole way there: signal, position, costs,
equity, metrics, on three real tickers, in one Polars query.

Every block below runs against a small sample of daily bars shipped **in the repository** — clone it and run from
its root (or point `read_parquet` at any OHLCV frame of your own, adjusting the column names). Paste them in order and you get
exactly the numbers you see.

## The data, in whatever shape it arrives

A quarter of daily bars for `AAPL`, `GOOG`, and `NVDA` — the first three months of 2024, split-adjusted, the ordinary
OHLCV any vendor would hand you. In your own code this is just your `pl.DataFrame`; here we read the bundled sample so
the page stays reproducible.

```{doctest}
>>> import polars as pl
>>>
>>> wide = pl.read_parquet("docs/_static/ohlcv_sample.parquet")
>>> wide.head(6)
shape: (6, 7)
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
└────────────────────────────────┴────────┴────────┴────────┴────────┴────────┴───────────┘
```

The `datetime` is tz-aware, stamped at the 17:00 New York close — `EST` here, `EDT` once March's clock change
lands — so the zone travels with the price. One row per ticker per day, one column per field: the shape every
`pomata` factory expects, because `pl.col("close")` has to point at a column.

Prices do not always arrive that way. The other common layout is long — every field stacked into a single `value`
column, named by a `field` column — which is what a tidy database or a melted export gives you.

```{doctest}
>>> long = wide.unpivot(index=["datetime", "ticker"], variable_name="field", value_name="value")
>>> long.head(6)
shape: (6, 4)
┌────────────────────────────────┬────────┬───────┬────────┐
│ datetime                       ┆ ticker ┆ field ┆ value  │
│ ---                            ┆ ---    ┆ ---   ┆ ---    │
│ datetime[μs, America/New_York] ┆ str    ┆ str   ┆ f64    │
╞════════════════════════════════╪════════╪═══════╪════════╡
│ 2024-01-02 17:00:00 EST        ┆ AAPL   ┆ open  ┆ 185.06 │
│ 2024-01-02 17:00:00 EST        ┆ GOOG   ┆ open  ┆ 138.38 │
│ 2024-01-02 17:00:00 EST        ┆ NVDA   ┆ open  ┆ 49.16  │
│ 2024-01-03 17:00:00 EST        ┆ AAPL   ┆ open  ┆ 182.16 │
│ 2024-01-03 17:00:00 EST        ┆ GOOG   ┆ open  ┆ 137.39 │
│ 2024-01-03 17:00:00 EST        ┆ NVDA   ┆ open  ┆ 47.4   │
└────────────────────────────────┴────────┴───────┴────────┘
```

Folding the integer volume in with the float prices makes `value` a single `f64`. Nothing is wrong with it; it just
means the field you want is picked out by a filter rather than named by a column.

We run the whole backtest from each shape below — first from `wide`, then from `long` — and the two land on the
same verdict to the last digit.

## A signal, and the warm-up it owes you

The idea is the oldest one in the book: be long while a fast average sits above a slow one. Two `ema` calls and a
comparison. The `.over("ticker")` is not decoration — it stops `NVDA`'s recursion from bleeding into the next ticker.

```{doctest}
>>> from pomata.indicators import ema
>>>
>>> signal = (
...     wide
...     .with_columns(
...         fast=ema(pl.col("close"), 5).over("ticker"),
...         slow=ema(pl.col("close"), 10).over("ticker"),
...     )
...     .with_columns(
...         position=(pl.col("fast") > pl.col("slow")).cast(pl.Float64).shift(1).over("ticker"),
...     )
... )
>>> (
...     signal.filter(pl.col("ticker") == "NVDA")
...     .select("datetime", "close", "fast", "slow", "position")
...     .slice(7, 7)
...     .with_columns(pl.col("fast", "slow").round(2))
... )
shape: (7, 5)
┌────────────────────────────────┬───────┬───────┬───────┬──────────┐
│ datetime                       ┆ close ┆ fast  ┆ slow  ┆ position │
│ ---                            ┆ ---   ┆ ---   ┆ ---   ┆ ---      │
│ datetime[μs, America/New_York] ┆ f64   ┆ f64   ┆ f64   ┆ f64      │
╞════════════════════════════════╪═══════╪═══════╪═══════╪══════════╡
│ 2024-01-11 17:00:00 EST        ┆ 54.72 ┆ 52.65 ┆ null  ┆ null     │
│ 2024-01-12 17:00:00 EST        ┆ 54.61 ┆ 53.3  ┆ null  ┆ null     │
│ 2024-01-16 17:00:00 EST        ┆ 56.28 ┆ 54.3  ┆ 51.76 ┆ null     │
│ 2024-01-17 17:00:00 EST        ┆ 55.95 ┆ 54.85 ┆ 52.52 ┆ 1.0      │
│ 2024-01-18 17:00:00 EST        ┆ 57.01 ┆ 55.57 ┆ 53.33 ┆ 1.0      │
│ 2024-01-19 17:00:00 EST        ┆ 59.39 ┆ 56.84 ┆ 54.44 ┆ 1.0      │
│ 2024-01-22 17:00:00 EST        ┆ 59.55 ┆ 57.74 ┆ 55.37 ┆ 1.0      │
└────────────────────────────────┴───────┴───────┴───────┴──────────┘
```

Two things to read. `slow` is `null` until its tenth bar: the window has not filled, so `pomata` leaves the cell empty
instead of seeding it with a zero that would quietly wreck everything downstream. And `position` is the comparison
`.shift(1)`-ed — the cross you can see at the 16th's close is acted on the next bar, the 17th, never the same day.
That one shift is the whole no-look-ahead story, and the `.over("ticker")` on it stops the shift at the ticker
boundary instead of carrying one ticker's first position back onto another's last row.

## From a signal to a verdict

A position is not yet a return. Take the per-bar return of the close, scale it by the position you were actually
holding, then pay for the trading — a backtest that skips the third step is telling you a comforting lie.

`returns_gross` applies the weight, `cost_proportional` charges a fee on the traded notional, `returns_net` subtracts
one from the other, and `equity_curve` compounds what is left. Every lagging piece carries `.over("ticker")` — the
averages, the position shift, the asset return, and the turnover inside the cost — so nothing reaches across the seam
between one ticker and the next. Group by ticker and reduce each to the few numbers you'd actually compare — one row
per ticker, because the grouping was there all along.

```{doctest}
>>> from pomata.pnl import returns_simple, returns_gross, returns_net, cost_proportional, equity_curve
>>> from pomata.metrics import sharpe_ratio, cagr, max_drawdown
>>>
>>> report_wide = (
...     wide
...     .with_columns(
...         position=(ema(pl.col("close"), 5).over("ticker") > ema(pl.col("close"), 10).over("ticker"))
...         .cast(pl.Float64)
...         .shift(1)
...         .over("ticker"),
...         asset_returns=returns_simple(pl.col("close")).over("ticker"),
...     )
...     .with_columns(
...         net=returns_net(
...             returns_gross(pl.col("position"), pl.col("asset_returns")),
...             cost_proportional(pl.col("position"), rate=0.001).over("ticker"),
...         ),
...     )
...     .group_by("ticker", maintain_order=True)
...     .agg(
...         sharpe=sharpe_ratio(pl.col("net"), periods_per_year=252).round(2),
...         cagr=cagr(equity_curve(pl.col("net")), periods_per_year=252).round(4),
...         max_drawdown=max_drawdown(equity_curve(pl.col("net"))).round(4),
...     )
... )
>>> report_wide
shape: (3, 4)
┌────────┬────────┬─────────┬──────────────┐
│ ticker ┆ sharpe ┆ cagr    ┆ max_drawdown │
│ ---    ┆ ---    ┆ ---     ┆ ---          │
│ str    ┆ f64    ┆ f64     ┆ f64          │
╞════════╪════════╪═════════╪══════════════╡
│ AAPL   ┆ -2.13  ┆ -0.2411 ┆ -0.0832      │
│ GOOG   ┆ 0.76   ┆ 0.166   ┆ -0.1196      │
│ NVDA   ┆ 4.76   ┆ 10.1001 ┆ -0.087       │
└────────┴────────┴─────────┴──────────────┘
```

Same rule, three verdicts. `NVDA` trended almost the whole quarter, the cross caught it, and the Sharpe shows it — the
CAGR reads like a fantasy because annualizing a single parabolic quarter always does, which is the standing reminder
that sixty bars are a sample, not a strategy. `GOOG` drifted up and the rule clipped a modest, real gain from it. `AAPL`
fell and chopped, the cross was never on the right side for long, and it bled. That is the honest outcome: a
moving-average cross follows trends, and `AAPL` had none to follow.

## The same query, from long

Long data hides the close inside `value`, on the rows where `field == "close"`. Filter to those, swap `pl.col("close")`
for `pl.col("value")`, and the pipeline is otherwise character-for-character the one above — `pomata`'s factories take
a column expression, and neither know nor care which shape it was sliced from.

```{doctest}
>>> prices = long.filter(pl.col("field") == "close")
>>> report_long = (
...     prices
...     .with_columns(
...         position=(ema(pl.col("value"), 5).over("ticker") > ema(pl.col("value"), 10).over("ticker"))
...         .cast(pl.Float64)
...         .shift(1)
...         .over("ticker"),
...         asset_returns=returns_simple(pl.col("value")).over("ticker"),
...     )
...     .with_columns(
...         net=returns_net(
...             returns_gross(pl.col("position"), pl.col("asset_returns")),
...             cost_proportional(pl.col("position"), rate=0.001).over("ticker"),
...         ),
...     )
...     .group_by("ticker", maintain_order=True)
...     .agg(
...         sharpe=sharpe_ratio(pl.col("net"), periods_per_year=252).round(2),
...         cagr=cagr(equity_curve(pl.col("net")), periods_per_year=252).round(4),
...         max_drawdown=max_drawdown(equity_curve(pl.col("net"))).round(4),
...     )
... )
>>> report_long
shape: (3, 4)
┌────────┬────────┬─────────┬──────────────┐
│ ticker ┆ sharpe ┆ cagr    ┆ max_drawdown │
│ ---    ┆ ---    ┆ ---     ┆ ---          │
│ str    ┆ f64    ┆ f64     ┆ f64          │
╞════════╪════════╪═════════╪══════════════╡
│ AAPL   ┆ -2.13  ┆ -0.2411 ┆ -0.0832      │
│ GOOG   ┆ 0.76   ┆ 0.166   ┆ -0.1196      │
│ NVDA   ┆ 4.76   ┆ 10.1001 ┆ -0.087       │
└────────┴────────┴─────────┴──────────────┘
>>> report_wide.equals(report_long)
True
```

Same numbers, by construction.

The lesson is not the strategy — it is that you went from raw bars to a costed, look-ahead-free, ticker-by-ticker 
number, twice and from two different layouts, without leaving Polars and without a second dependency in the stack.

## Where to go next

Swap `ema` for anything in the [indicator catalog](families/indicators.md), change the fee model from the
[pnl family](families/pnl.md), or pull more figures from the [metrics catalog](families/metrics.md) — the shape of the
query does not move. Nor does it move when the panel grows: put five hundred tickers in `wide` and the same
`.over("ticker")` that kept three of them apart keeps five hundred apart.
