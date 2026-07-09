# PnL

`pomata.pnl` turns the positions you hold into per-bar profit and loss, in two flows — **returns** (a signed weight of
capital times an asset return) and **cash** (a quantity of units times a price change) — with transaction costs you
compose, never bake in. Pick the flow that matches your data; either one ends in exactly the return or equity series the
{doc}`metrics <metrics>` family consumes.

## What you get

Eighteen primitives, split by the unit you hold — which is also the flow.

### Returns flow

A signed weight and an asset return, from the price transform through to the compounded capital curve:

{py:func}`~pomata.pnl.returns_simple` · {py:func}`~pomata.pnl.returns_log` · {py:func}`~pomata.pnl.returns_gross` ·
{py:func}`~pomata.pnl.returns_net` · {py:func}`~pomata.pnl.equity_curve` · {py:func}`~pomata.pnl.turnover` ·
{py:func}`~pomata.pnl.cost_proportional` · {py:func}`~pomata.pnl.cost_slippage`

### Cash flow

A quantity of units and a price, for instrument-level booking with contract multipliers, dividends, and funding:

{py:func}`~pomata.pnl.pnl_gross` · {py:func}`~pomata.pnl.pnl_gross_inverse` · {py:func}`~pomata.pnl.pnl_net` ·
{py:func}`~pomata.pnl.cumulative_pnl` · {py:func}`~pomata.pnl.dividend` · {py:func}`~pomata.pnl.cost_fixed` ·
{py:func}`~pomata.pnl.cost_notional` · {py:func}`~pomata.pnl.cost_per_share` · {py:func}`~pomata.pnl.cost_borrow` ·
{py:func}`~pomata.pnl.cost_funding`

The whole flow, end to end — a position built, paid a dividend, and unwound, with per-share and notional fees booked
on every trade. `quantity` at a row is the position held **over the price change into that row** (decide at the
close, lag by one — the no-look-ahead note on {py:func}`~pomata.pnl.pnl_gross`), so the entry sits on the second row
and the entry fees land where the first defined P&L does:

```{doctest}
>>> import polars as pl
>>> from pomata.pnl import pnl_gross, dividend, cost_per_share, cost_notional, pnl_net, cumulative_pnl
>>>
>>> frame = pl.DataFrame(
...     {
...         "quantity": [0.0, 100.0, 150.0, 150.0, 0.0],
...         "price": [50.0, 51.5, 51.0, 52.5, 52.0],
...         "dividend_per_share": [0.0, 0.0, 0.4, 0.0, 0.0],
...     }
... )
>>> gross = pnl_gross(pl.col("quantity"), pl.col("price"))
>>> income = dividend(pl.col("quantity"), pl.col("dividend_per_share"))
>>> fees = cost_per_share(pl.col("quantity"), fee=0.01) + cost_notional(
...     pl.col("quantity"), pl.col("price"), rate=0.0005
... )
>>> net = pnl_net(gross + income, fees)
>>> frame.with_columns(net=net.round(2), total=cumulative_pnl(net).round(2))
shape: (5, 5)
┌──────────┬───────┬────────────────────┬────────┬────────┐
│ quantity ┆ price ┆ dividend_per_share ┆ net    ┆ total  │
│ ---      ┆ ---   ┆ ---                ┆ ---    ┆ ---    │
│ f64      ┆ f64   ┆ f64                ┆ f64    ┆ f64    │
╞══════════╪═══════╪════════════════════╪════════╪════════╡
│ 0.0      ┆ 50.0  ┆ 0.0                ┆ null   ┆ null   │
│ 100.0    ┆ 51.5  ┆ 0.0                ┆ 146.43 ┆ 146.43 │
│ 150.0    ┆ 51.0  ┆ 0.4                ┆ -16.77 ┆ 129.65 │
│ 150.0    ┆ 52.5  ┆ 0.0                ┆ 225.0  ┆ 354.65 │
│ 0.0      ┆ 52.0  ┆ 0.0                ┆ -5.4   ┆ 349.25 │
└──────────┴───────┴────────────────────┴────────┴────────┘
```

Row by row: the entry earns `100 x 1.5 = 150` gross less `1.00` per-share and `2.575` notional fees; the add-on bar
loses `150 x (-0.5) = -75` but collects the `150 x 0.4 = 60` dividend, less the fees on the 50-share trade; the flat
bar rides `150 x 1.5` cost-free; the unwind pays only its exit fees on a flat price move. The running ``total`` is
additive — currency P&L sums, it does not compound.

## Common pains, solved

### Gross vs net: the costs that actually bite

A backtest that forgets transaction costs is a fiction — the alpha you keep is the gross return minus what the broker
takes. Compose the cost and net it; on every bar you traded, net sits strictly below gross.

```{doctest}
>>> import polars as pl
>>> from datetime import datetime
>>> from pomata.pnl import returns_gross, returns_net, cost_proportional
>>>
>>> frame = pl.DataFrame(
...     {
...         "datetime": [datetime(2024, 1, d, 17) for d in (2, 3, 4, 5, 8, 9)],
...         "ticker": "AAPL",
...         "weight": [1.0, 1.0, -1.0, -1.0, 0.5, 0.5],
...         "asset_returns": [0.02, -0.01, 0.03, -0.02, 0.01, 0.04],
...     }
... ).with_columns(pl.col("datetime").dt.replace_time_zone("America/New_York"))
>>> gross = returns_gross(pl.col("weight"), pl.col("asset_returns"))
>>> net = returns_net(gross, cost_proportional(pl.col("weight"), rate=0.001))
>>> frame.with_columns(gross=gross.round(4), net=net.round(4))
shape: (6, 6)
┌────────────────────────────────┬────────┬────────┬───────────────┬───────┬────────┐
│ datetime                       ┆ ticker ┆ weight ┆ asset_returns ┆ gross ┆ net    │
│ ---                            ┆ ---    ┆ ---    ┆ ---           ┆ ---   ┆ ---    │
│ datetime[μs, America/New_York] ┆ str    ┆ f64    ┆ f64           ┆ f64   ┆ f64    │
╞════════════════════════════════╪════════╪════════╪═══════════════╪═══════╪════════╡
│ 2024-01-02 17:00:00 EST        ┆ AAPL   ┆ 1.0    ┆ 0.02          ┆ 0.02  ┆ 0.019  │
│ 2024-01-03 17:00:00 EST        ┆ AAPL   ┆ 1.0    ┆ -0.01         ┆ -0.01 ┆ -0.01  │
│ 2024-01-04 17:00:00 EST        ┆ AAPL   ┆ -1.0   ┆ 0.03          ┆ -0.03 ┆ -0.032 │
│ 2024-01-05 17:00:00 EST        ┆ AAPL   ┆ -1.0   ┆ -0.02         ┆ 0.02  ┆ 0.02   │
│ 2024-01-08 17:00:00 EST        ┆ AAPL   ┆ 0.5    ┆ 0.01          ┆ 0.005 ┆ 0.0035 │
│ 2024-01-09 17:00:00 EST        ┆ AAPL   ┆ 0.5    ┆ 0.04          ┆ 0.02  ┆ 0.02   │
└────────────────────────────────┴────────┴────────┴───────────────┴───────┴────────┘
```

The drag lands only where the weight moved (rows 0, 2, 4 — the entry, the flip, the resize); a held bar pays nothing,
so its net equals its gross.

### Signal → position → P&L, with no look-ahead

A signal computed at a bar's close cannot trade that same bar — only the next one. One `.shift(1)` on the weight is the
whole story: the position lags the signal by exactly one bar, so the P&L can never peek at the return that produced the
signal.

```{doctest}
>>> from pomata.pnl import returns_simple
>>>
>>> frame = pl.DataFrame(
...     {
...         "datetime": [datetime(2024, 1, d, 17) for d in (2, 3, 4, 5, 8, 9, 10)],
...         "ticker": "AAPL",
...         "close": [100.0, 102.0, 101.0, 104.0, 103.0, 106.0, 108.0],
...     }
... ).with_columns(pl.col("datetime").dt.replace_time_zone("America/New_York"))
>>> asset_returns = returns_simple(pl.col("close"))
>>> signal = (asset_returns > 0).cast(pl.Float64)   # decide at close t
>>> weight = signal.shift(1)                         # act at t+1
>>> frame.with_columns(signal=signal, weight=weight, gross=returns_gross(weight, asset_returns).round(4))
shape: (7, 6)
┌────────────────────────────────┬────────┬───────┬────────┬────────┬─────────┐
│ datetime                       ┆ ticker ┆ close ┆ signal ┆ weight ┆ gross   │
│ ---                            ┆ ---    ┆ ---   ┆ ---    ┆ ---    ┆ ---     │
│ datetime[μs, America/New_York] ┆ str    ┆ f64   ┆ f64    ┆ f64    ┆ f64     │
╞════════════════════════════════╪════════╪═══════╪════════╪════════╪═════════╡
│ 2024-01-02 17:00:00 EST        ┆ AAPL   ┆ 100.0 ┆ null   ┆ null   ┆ null    │
│ 2024-01-03 17:00:00 EST        ┆ AAPL   ┆ 102.0 ┆ 1.0    ┆ null   ┆ null    │
│ 2024-01-04 17:00:00 EST        ┆ AAPL   ┆ 101.0 ┆ 0.0    ┆ 1.0    ┆ -0.0098 │
│ 2024-01-05 17:00:00 EST        ┆ AAPL   ┆ 104.0 ┆ 1.0    ┆ 0.0    ┆ 0.0     │
│ 2024-01-08 17:00:00 EST        ┆ AAPL   ┆ 103.0 ┆ 0.0    ┆ 1.0    ┆ -0.0096 │
│ 2024-01-09 17:00:00 EST        ┆ AAPL   ┆ 106.0 ┆ 1.0    ┆ 0.0    ┆ 0.0     │
│ 2024-01-10 17:00:00 EST        ┆ AAPL   ┆ 108.0 ┆ 1.0    ┆ 1.0    ┆ 0.0189  │
└────────────────────────────────┴────────┴───────┴────────┴────────┴─────────┘
```

The `signal` and the `weight` are the same series, offset by one row: the up-bar at index 1 only earns its position at
index 2, where it eats the next bar's `-0.0098`. Nothing is shifted for you, so an already-aligned weight is never
double-lagged.

### Equity curve vs cumulative P&L

Two ways to total a return series, and confusing them quietly corrupts every downstream metric.
{py:func}`~pomata.pnl.equity_curve` *compounds* — a multiplicative growth factor for capital that reinvests its P&L;
{py:func}`~pomata.pnl.cumulative_pnl` *sums* — an additive currency total for a fixed notional. Same inputs, different
cumulation.

```{doctest}
>>> from pomata.pnl import equity_curve, cumulative_pnl
>>>
>>> frame = pl.DataFrame(
...     {
...         "datetime": [datetime(2024, 1, d, 17) for d in (2, 3, 4, 5, 8)],
...         "ticker": "AAPL",
...         "returns": [0.1, 0.1, 0.1, -0.1, 0.1],
...     }
... ).with_columns(pl.col("datetime").dt.replace_time_zone("America/New_York"))
>>> frame.with_columns(
...     equity=equity_curve(pl.col("returns")).round(4),
...     cum_pnl=cumulative_pnl(pl.col("returns")).round(4),
... )
shape: (5, 5)
┌────────────────────────────────┬────────┬─────────┬────────┬─────────┐
│ datetime                       ┆ ticker ┆ returns ┆ equity ┆ cum_pnl │
│ ---                            ┆ ---    ┆ ---     ┆ ---    ┆ ---     │
│ datetime[μs, America/New_York] ┆ str    ┆ f64     ┆ f64    ┆ f64     │
╞════════════════════════════════╪════════╪═════════╪════════╪═════════╡
│ 2024-01-02 17:00:00 EST        ┆ AAPL   ┆ 0.1     ┆ 1.1    ┆ 0.1     │
│ 2024-01-03 17:00:00 EST        ┆ AAPL   ┆ 0.1     ┆ 1.21   ┆ 0.2     │
│ 2024-01-04 17:00:00 EST        ┆ AAPL   ┆ 0.1     ┆ 1.331  ┆ 0.3     │
│ 2024-01-05 17:00:00 EST        ┆ AAPL   ┆ -0.1    ┆ 1.1979 ┆ 0.2     │
│ 2024-01-08 17:00:00 EST        ┆ AAPL   ┆ 0.1     ┆ 1.3177 ┆ 0.3     │
└────────────────────────────────┴────────┴─────────┴────────┴─────────┘
```

Three `+10%` bars compound to `1.331` but sum to `0.3`; the `-10%` bar then takes the curve to `1.1979` (a tenth *of
the grown capital*) while the sum drops by a flat `0.1`. Feed the equity curve to a drawdown, the cumulative total to a
fixed-notional ledger — never the reverse.

### Per-asset P&L on a panel, via `.over`

A long panel of many tickers is one DataFrame — but a naked return reaches across the ticker boundary and books a
phantom move where one symbol's last close meets the next's first bar. Wrap the call in `.over(...)` and each ticker
restarts its own warm-up.

```{doctest}
>>> frame = pl.DataFrame(
...     {
...         "datetime": [datetime(2024, 1, d, 17) for d in (2, 3, 4)] * 3,
...         "ticker": ["AAPL"] * 3 + ["GOOG"] * 3 + ["NVDA"] * 3,
...         "close": [183.56, 182.19, 179.87, 138.34, 139.13, 136.83, 48.08, 47.48, 47.91],
...     }
... ).with_columns(pl.col("datetime").dt.replace_time_zone("America/New_York"))
>>> (
...     frame
...     .sort("ticker", "datetime")
...     .with_columns(
...         leaky=returns_simple(pl.col("close")).round(4),
...         clean=returns_simple(pl.col("close")).over("ticker").round(4),
...     )
... )
shape: (9, 5)
┌────────────────────────────────┬────────┬────────┬─────────┬─────────┐
│ datetime                       ┆ ticker ┆ close  ┆ leaky   ┆ clean   │
│ ---                            ┆ ---    ┆ ---    ┆ ---     ┆ ---     │
│ datetime[μs, America/New_York] ┆ str    ┆ f64    ┆ f64     ┆ f64     │
╞════════════════════════════════╪════════╪════════╪═════════╪═════════╡
│ 2024-01-02 17:00:00 EST        ┆ AAPL   ┆ 183.56 ┆ null    ┆ null    │
│ 2024-01-03 17:00:00 EST        ┆ AAPL   ┆ 182.19 ┆ -0.0075 ┆ -0.0075 │
│ 2024-01-04 17:00:00 EST        ┆ AAPL   ┆ 179.87 ┆ -0.0127 ┆ -0.0127 │
│ 2024-01-02 17:00:00 EST        ┆ GOOG   ┆ 138.34 ┆ -0.2309 ┆ null    │
│ 2024-01-03 17:00:00 EST        ┆ GOOG   ┆ 139.13 ┆ 0.0057  ┆ 0.0057  │
│ 2024-01-04 17:00:00 EST        ┆ GOOG   ┆ 136.83 ┆ -0.0165 ┆ -0.0165 │
│ 2024-01-02 17:00:00 EST        ┆ NVDA   ┆ 48.08  ┆ -0.6486 ┆ null    │
│ 2024-01-03 17:00:00 EST        ┆ NVDA   ┆ 47.48  ┆ -0.0125 ┆ -0.0125 │
│ 2024-01-04 17:00:00 EST        ┆ NVDA   ┆ 47.91  ┆ 0.0091  ┆ 0.0091  │
└────────────────────────────────┴────────┴────────┴─────────┴─────────┘
```

Without `.over`, ticker `AAPL`'s `179.87` close bleeds into ticker `GOOG`'s `138.34` open and fabricates a `-23.09%`
return at the boundary. With it, `GOOG` begins at `null` — its own warm-up — exactly as a fresh series should.
