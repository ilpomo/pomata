# Concepts

Five ideas do all the work. Internalize them once, and every function in `pomata` composes the same way.

## 1. Everything is a `pl.Expr` factory

Every public function *returns a Polars expression* — it never touches your data. Name it, compose it, pass it
around, and run it in any context: a `select`, a `with_columns`, eager or lazy. Nothing forces a DataFrame shape on
you.

```python
from pomata.indicators import ema

macd_line = ema(pl.col("close"), 12) - ema(pl.col("close"), 26)   # just an expression
frame.with_columns(macd=macd_line)                                # run it wherever you like
```

One naming rule follows from this: **the output column keeps the root name of its leading input** (`rsi(pl.col("close"), 14)`
lands on `close` — in a `with_columns` it *replaces* that column unless you name it). To name an output, alias the
**returned expression**, never the input: `rsi(pl.col("close"), 14).alias("rsi_14")` — an alias on the input is
deliberately ignored (`name.keep` restores the root), so it can never silently land the result on an unexpected column.

## 2. `.over` for multi-asset panels

A panel of many tickers is one DataFrame and one query: wrap the call in `.over(...)` and each group is reduced
independently — windows and recursions never bleed across a boundary.

```{doctest}
>>> import polars as pl
>>> from pomata.indicators import sma
>>>
>>> frame = pl.DataFrame(
...     {
...         "ticker": ["AAPL", "AAPL", "AAPL", "GOOG", "GOOG", "GOOG", "NVDA", "NVDA", "NVDA"],
...         "close": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
...     }
... )
>>> frame.with_columns(sma=sma(pl.col("close"), 2).over("ticker"))
shape: (9, 3)
┌────────┬───────┬───────┐
│ ticker ┆ close ┆ sma   │
│ ---    ┆ ---   ┆ ---   │
│ str    ┆ f64   ┆ f64   │
╞════════╪═══════╪═══════╡
│ AAPL   ┆ 1.0   ┆ null  │
│ AAPL   ┆ 2.0   ┆ 1.5   │
│ AAPL   ┆ 3.0   ┆ 2.5   │
│ GOOG   ┆ 10.0  ┆ null  │
│ GOOG   ┆ 20.0  ┆ 15.0  │
│ GOOG   ┆ 30.0  ┆ 25.0  │
│ NVDA   ┆ 100.0 ┆ null  │
│ NVDA   ┆ 200.0 ┆ 150.0 │
│ NVDA   ┆ 300.0 ┆ 250.0 │
└────────┴───────┴───────┘
```

Ticker `GOOG` starts its own warm-up (`None`) instead of inheriting `AAPL`'s tail.

One performance note: a handful of sequential functions run a Python kernel per evaluation — among the indicators
the Ehlers cycle family, KAMA, the parabolic SAR, SuperTrend, the Fisher Transform, and the seeded EMA family; among
the metrics the rolling skewness / kurtosis and `probabilistic_sharpe_ratio`'s normal CDF — and reusing such an
expression textually re-runs it: Polars' subexpression caching does not reach inside `.over(...)`. Assign the
expression to a column once and derive from the column when composing several outputs from the same expensive input.

## 3. Warm-up is `null`, never fabricated

Until a window fills, the output is `null` — never a zero, never a forward-filled guess:

```{doctest}
>>> frame = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]})
>>> frame.with_columns(sma=sma(pl.col("close"), 3))
shape: (5, 2)
┌───────┬──────┐
│ close ┆ sma  │
│ ---   ┆ ---  │
│ f64   ┆ f64  │
╞═══════╪══════╡
│ 1.0   ┆ null │
│ 2.0   ┆ null │
│ 3.0   ┆ 2.0  │
│ 4.0   ┆ 3.0  │
│ 5.0   ┆ 4.0  │
└───────┴──────┘
```

A fabricated warm-up value is a silent look-ahead. `pomata` refuses to invent one, so a `null` always means *not
enough data yet* — exactly what you want before you trade on it.

## 4. No look-ahead, by construction

A signal computed at the close can only act on the *next* bar. That is a single `.shift(1)`, and it is the whole
story — no hidden alignment, no off-by-one:

```python
from pomata.indicators import rsi

weight = (rsi(pl.col("close"), 14) < 30).cast(pl.Float64).shift(1)   # decide at close t, act at t+1
```

## 5. Multi-output indicators return a `pl.Struct`

Anything with several lines — Bollinger Bands, MACD, Stochastic Oscillator, Ichimoku Cloud — returns one struct column. Pick a line
with `.struct.field(...)`, or expand them all with `.struct.unnest()`:

```{doctest}
>>> from pomata.indicators import bollinger_bands
>>>
>>> frame = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
>>> frame.select(bollinger_bands(pl.col("close"), 3).alias("bb")).unnest("bb").with_columns(pl.all().round(4))
shape: (6, 3)
┌───────┬────────┬───────┐
│ lower ┆ middle ┆ upper │
│ ---   ┆ ---    ┆ ---   │
│ f64   ┆ f64    ┆ f64   │
╞═══════╪════════╪═══════╡
│ null  ┆ null   ┆ null  │
│ null  ┆ null   ┆ null  │
│ 0.367 ┆ 2.0    ┆ 3.633 │
│ 1.367 ┆ 3.0    ┆ 4.633 │
│ 2.367 ┆ 4.0    ┆ 5.633 │
│ 3.367 ┆ 5.0    ┆ 6.633 │
└───────┴────────┴───────┘
```
