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

## Common pains, solved

### Gross vs net: the costs that actually bite

A backtest that forgets transaction costs is a fiction — the alpha you keep is the gross return minus what the broker
takes. Compose the cost and net it; on every bar you traded, net sits strictly below gross.

```{doctest}
>>> import polars as pl
>>> from pomata.pnl import returns_gross, returns_net, cost_proportional
>>>
>>> frame = pl.DataFrame(
...     {
...         "weight": [1.0, 1.0, -1.0, -1.0, 0.5, 0.5],
...         "asset_returns": [0.02, -0.01, 0.03, -0.02, 0.01, 0.04],
...     }
... )
>>> gross = returns_gross(pl.col("weight"), pl.col("asset_returns"))
>>> net = returns_net(gross, cost_proportional(pl.col("weight"), rate=0.001))
>>> frame.with_columns(gross=gross.round(4), net=net.round(4))
shape: (6, 4)
┌────────┬───────────────┬───────┬────────┐
│ weight ┆ asset_returns ┆ gross ┆ net    │
│ ---    ┆ ---           ┆ ---   ┆ ---    │
│ f64    ┆ f64           ┆ f64   ┆ f64    │
╞════════╪═══════════════╪═══════╪════════╡
│ 1.0    ┆ 0.02          ┆ 0.02  ┆ 0.019  │
│ 1.0    ┆ -0.01         ┆ -0.01 ┆ -0.01  │
│ -1.0   ┆ 0.03          ┆ -0.03 ┆ -0.032 │
│ -1.0   ┆ -0.02         ┆ 0.02  ┆ 0.02   │
│ 0.5    ┆ 0.01          ┆ 0.005 ┆ 0.0035 │
│ 0.5    ┆ 0.04          ┆ 0.02  ┆ 0.02   │
└────────┴───────────────┴───────┴────────┘
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
>>> frame = pl.DataFrame({"close": [100.0, 102.0, 101.0, 104.0, 103.0, 106.0, 108.0]})
>>> asset_returns = returns_simple(pl.col("close"))
>>> signal = (asset_returns > 0).cast(pl.Float64)   # decide at close t
>>> weight = signal.shift(1)                         # act at t+1
>>> frame.with_columns(signal=signal, weight=weight, gross=returns_gross(weight, asset_returns).round(4))
shape: (7, 4)
┌───────┬────────┬────────┬─────────┐
│ close ┆ signal ┆ weight ┆ gross   │
│ ---   ┆ ---    ┆ ---    ┆ ---     │
│ f64   ┆ f64    ┆ f64    ┆ f64     │
╞═══════╪════════╪════════╪═════════╡
│ 100.0 ┆ null   ┆ null   ┆ null    │
│ 102.0 ┆ 1.0    ┆ null   ┆ null    │
│ 101.0 ┆ 0.0    ┆ 1.0    ┆ -0.0098 │
│ 104.0 ┆ 1.0    ┆ 0.0    ┆ 0.0     │
│ 103.0 ┆ 0.0    ┆ 1.0    ┆ -0.0096 │
│ 106.0 ┆ 1.0    ┆ 0.0    ┆ 0.0     │
│ 108.0 ┆ 1.0    ┆ 1.0    ┆ 0.0189  │
└───────┴────────┴────────┴─────────┘
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
>>> frame = pl.DataFrame({"returns": [0.1, 0.1, 0.1, -0.1, 0.1]})
>>> frame.with_columns(equity=equity_curve(pl.col("returns")).round(4), cum_pnl=cumulative_pnl(pl.col("returns")).round(4))
shape: (5, 3)
┌─────────┬────────┬─────────┐
│ returns ┆ equity ┆ cum_pnl │
│ ---     ┆ ---    ┆ ---     │
│ f64     ┆ f64    ┆ f64     │
╞═════════╪════════╪═════════╡
│ 0.1     ┆ 1.1    ┆ 0.1     │
│ 0.1     ┆ 1.21   ┆ 0.2     │
│ 0.1     ┆ 1.331  ┆ 0.3     │
│ -0.1    ┆ 1.1979 ┆ 0.2     │
│ 0.1     ┆ 1.3177 ┆ 0.3     │
└─────────┴────────┴─────────┘
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
...         "ticker": ["A", "A", "A", "B", "B", "B"],
...         "close": [100.0, 110.0, 121.0, 50.0, 55.0, 60.5],
...     }
... )
>>> frame.with_columns(
...     leaky=returns_simple(pl.col("close")).round(4),
...     clean=returns_simple(pl.col("close")).over("ticker").round(4),
... )
shape: (6, 4)
┌────────┬───────┬─────────┬───────┐
│ ticker ┆ close ┆ leaky   ┆ clean │
│ ---    ┆ ---   ┆ ---     ┆ ---   │
│ str    ┆ f64   ┆ f64     ┆ f64   │
╞════════╪═══════╪═════════╪═══════╡
│ A      ┆ 100.0 ┆ null    ┆ null  │
│ A      ┆ 110.0 ┆ 0.1     ┆ 0.1   │
│ A      ┆ 121.0 ┆ 0.1     ┆ 0.1   │
│ B      ┆ 50.0  ┆ -0.5868 ┆ null  │
│ B      ┆ 55.0  ┆ 0.1     ┆ 0.1   │
│ B      ┆ 60.5  ┆ 0.1     ┆ 0.1   │
└────────┴───────┴─────────┴───────┘
```

Without `.over`, ticker `A`'s `121` close bleeds into ticker `B`'s `50` open and fabricates a `-58.68%` return at the
boundary. With it, `B` begins at `null` — its own warm-up — exactly as a fresh series should.
