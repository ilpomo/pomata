# Tutorial

You have a table of prices and a hunch. You want a single number that says whether the hunch would have made money —
honestly, after costs, without cheating on timing. This walks the whole way there: signal, position, costs, equity,
metrics, on a two-ticker panel, in one Polars query. Every block below runs. Paste them in order and you get the same
output.

## A panel to work with

No data loader and no download — just a deterministic toy panel so the numbers stay reproducible. Two tickers,
twenty-five daily bars each, ordinary OHLCV. Skim the helper; the point is the frame it returns.

```{doctest}
>>> import polars as pl
>>> from math import sin
>>>
>>> def panel(ticker, base, drift, amplitude, period):
...     close = [round(base + drift * i + amplitude * sin(i / period), 2) for i in range(25)]
...     opens = [float(base), *close[:-1]]
...     high = [round(max(o, c) + 0.6, 2) for o, c in zip(opens, close)]
...     low = [round(min(o, c) - 0.6, 2) for o, c in zip(opens, close)]
...     volume = [1000.0 + 100 * (i % 5) for i in range(25)]
...     return pl.DataFrame({"ticker": ticker, "open": opens, "high": high, "low": low, "close": close, "volume": volume})
...
>>> frame = pl.concat([panel("AAA", 100, 0.6, 4.0, 6.0), panel("BBB", 50, -0.15, 3.0, 5.0)])
>>> frame.head(6)
shape: (6, 6)
┌────────┬────────┬────────┬────────┬────────┬────────┐
│ ticker ┆ open   ┆ high   ┆ low    ┆ close  ┆ volume │
│ ---    ┆ ---    ┆ ---    ┆ ---    ┆ ---    ┆ ---    │
│ str    ┆ f64    ┆ f64    ┆ f64    ┆ f64    ┆ f64    │
╞════════╪════════╪════════╪════════╪════════╪════════╡
│ AAA    ┆ 100.0  ┆ 100.6  ┆ 99.4   ┆ 100.0  ┆ 1000.0 │
│ AAA    ┆ 100.0  ┆ 101.86 ┆ 99.4   ┆ 101.26 ┆ 1100.0 │
│ AAA    ┆ 101.26 ┆ 103.11 ┆ 100.66 ┆ 102.51 ┆ 1200.0 │
│ AAA    ┆ 102.51 ┆ 104.32 ┆ 101.91 ┆ 103.72 ┆ 1300.0 │
│ AAA    ┆ 103.72 ┆ 105.47 ┆ 103.12 ┆ 104.87 ┆ 1400.0 │
│ AAA    ┆ 104.87 ┆ 106.56 ┆ 104.27 ┆ 105.96 ┆ 1000.0 │
└────────┴────────┴────────┴────────┴────────┴────────┘
```

AAA drifts up, BBB drifts down, and both wobble on the way. Keep that asymmetry in mind — it decides the verdict.

## A signal, and the warm-up it owes you

The idea is the oldest one in the book: be long while a fast average sits above a slow one. Two `ema` calls and a
comparison. The `.over("ticker")` is not decoration — it stops AAA's recursion from bleeding into BBB.

```{doctest}
>>> from pomata.indicators import ema
>>>
>>> signal = frame.with_columns(
...     fast=ema(pl.col("close"), 5).over("ticker"),
...     slow=ema(pl.col("close"), 10).over("ticker"),
... ).with_columns(
...     position=(pl.col("fast") > pl.col("slow")).cast(pl.Float64).shift(1).over("ticker"),
... )
>>> signal.filter(pl.col("ticker") == "AAA").select("close", "fast", "slow", "position").slice(8, 7).with_columns(pl.col("fast", "slow").round(2))
shape: (7, 4)
┌────────┬────────┬────────┬──────────┐
│ close  ┆ fast   ┆ slow   ┆ position │
│ ---    ┆ ---    ┆ ---    ┆ ---      │
│ f64    ┆ f64    ┆ f64    ┆ f64      │
╞════════╪════════╪════════╪══════════╡
│ 108.69 ┆ 106.76 ┆ null   ┆ null     │
│ 109.39 ┆ 107.63 ┆ 105.13 ┆ null     │
│ 109.98 ┆ 108.42 ┆ 106.01 ┆ 1.0      │
│ 110.46 ┆ 109.1  ┆ 106.82 ┆ 1.0      │
│ 110.84 ┆ 109.68 ┆ 107.55 ┆ 1.0      │
│ 111.11 ┆ 110.16 ┆ 108.2  ┆ 1.0      │
│ 111.29 ┆ 110.53 ┆ 108.76 ┆ 1.0      │
└────────┴────────┴────────┴──────────┘
```

Two things to read. `slow` is `null` until its tenth bar: the window has not filled, so `pomata` leaves the cell empty
instead of seeding it with a zero that would quietly wreck everything downstream. And `position` is the comparison
`.shift(1)`-ed — the cross you can see at today's close is acted on *tomorrow's* bar, never today's. That one shift is
the whole no-look-ahead story, and the `.over("ticker")` on it means the shift stops at the ticker boundary rather
than carrying BBB's first position back onto AAA's last row.

## From a signal to money

A position is not yet a return. Take the per-bar return of the close, scale it by the position you were actually
holding, then pay for the trading — a backtest that skips the third step is telling you a comforting lie.
`returns_gross` applies the weight, `cost_proportional` charges a fee on the traded notional, and `returns_net`
subtracts the cost from the gross. Those three feed straight into the metrics, so rather than print the intermediate
columns, here is the whole thing reduced to the only rows you compare.

## The verdict

Compound the net returns into an equity curve, then collapse each ticker to the figures you would actually put side by
side. It is one row per ticker because the grouping has been there all along.

```{doctest}
>>> from pomata.pnl import returns_simple, returns_gross, returns_net, cost_proportional, equity_curve
>>> from pomata.metrics import sharpe_ratio, cagr, max_drawdown
>>>
>>> report = (
...     frame.with_columns(
...         position=(ema(pl.col("close"), 5).over("ticker") > ema(pl.col("close"), 10).over("ticker"))
...         .cast(pl.Float64)
...         .shift(1)
...         .over("ticker"),
...         asset_returns=returns_simple(pl.col("close")).over("ticker"),
...     )
...     .with_columns(
...         net=returns_net(
...             returns_gross(pl.col("position"), pl.col("asset_returns")),
...             cost_proportional(pl.col("position"), rate=0.001),
...         ),
...     )
...     .group_by("ticker", maintain_order=True)
...     .agg(
...         sharpe=sharpe_ratio(pl.col("net"), periods_per_year=252).round(2),
...         cagr=cagr(equity_curve(pl.col("net")), periods_per_year=252).round(4),
...         max_drawdown=max_drawdown(equity_curve(pl.col("net"))).round(4),
...     )
... )
>>> report
shape: (2, 4)
┌────────┬────────┬─────────┬──────────────┐
│ ticker ┆ sharpe ┆ cagr    ┆ max_drawdown │
│ ---    ┆ ---    ┆ ---     ┆ ---          │
│ str    ┆ f64    ┆ f64     ┆ f64          │
╞════════╪════════╪═════════╪══════════════╡
│ AAA    ┆ 9.17   ┆ 0.2537  ┆ -0.0022      │
│ BBB    ┆ -6.56  ┆ -0.3112 ┆ -0.0118      │
└────────┴────────┴─────────┴──────────────┘
```

Same rule, opposite verdicts. On AAA the cross caught the drift and gave almost nothing back — a worst drawdown of a
fifth of a percent. On BBB it stayed long into a market that kept falling, and bled for it. That is the honest outcome:
a moving-average cross follows trends, and BBB had none to follow. The lesson is not the strategy — it is that you went
from raw bars to a costed, look-ahead-free, ticker-by-ticker number without leaving Polars and without a second
dependency in the stack.

## Where to go next

Swap `ema` for anything in the [indicator catalog](families/indicators.md), change the fee model from the
[pnl family](families/pnl.md), or pull more figures from the [metrics catalog](families/metrics.md) — the shape of the
query does not move. Nor does it move when the panel grows: put five hundred tickers in `frame` and the same
`.over("ticker")` that kept two of them apart keeps five hundred apart.
```
