# Design

Six ideas underlie every function in `pomata`. This page states each one and proves it with a runnable demo.

## 1. Everything is a `pl.Expr` factory

Every public function *returns a Polars expression* — it never touches your data. Name it, compose it, pass it
around, and run it in any context: a `select`, a `with_columns`, eager or lazy. Nothing forces a DataFrame shape on
you.

With an array-in, array-out library, computing a moving average, an oscillator, and a volatility band means three
separate calls and three intermediate results to glue back onto the frame by hand. Here, they are three expressions
in one lazy query: Polars applies its
[query optimizations](https://docs.pola.rs/user-guide/lazy/optimizations/) to the single plan, runs
[independent expressions in parallel](https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/), and one
`.collect()` materializes the finished frame:

```{doctest}
>>> import polars as pl
>>> from pomata.indicators import rsi, sma, atr
>>>
>>> prices = pl.LazyFrame(
...     {
...         "high":  [10.0, 11.0, 12.0, 11.5, 13.0, 14.0, 13.5, 15.0],
...         "low":   [ 9.0,  9.5, 10.5, 10.0, 11.0, 12.5, 12.0, 13.5],
...         "close": [ 9.5, 10.5, 11.5, 11.0, 12.5, 13.5, 13.0, 14.5],
...     }
... )
>>> signals = (
...     prices
...     .with_columns(
...         fast=sma(pl.col("close"), 2),
...         slow=sma(pl.col("close"), 4),
...         vol=atr(pl.col("high"), pl.col("low"), pl.col("close"), 3),
...     )
...     .with_columns(long=(pl.col("fast") > pl.col("slow")) & (rsi(pl.col("close"), 3) > 50.0))
...     .collect()
... )
>>> signals.select("close", pl.col("vol").round(4), "long").tail(4)
shape: (4, 3)
┌───────┬────────┬──────┐
│ close ┆ vol    ┆ long │
│ ---   ┆ ---    ┆ ---  │
│ f64   ┆ f64    ┆ bool │
╞═══════╪════════╪══════╡
│ 12.5  ┆ 1.5926 ┆ true │
│ 13.5  ┆ 1.5617 ┆ true │
│ 13.0  ┆ 1.5412 ┆ true │
│ 14.5  ┆ 1.6941 ┆ true │
└───────┴────────┴──────┘
```

The regime filter (`fast > slow`) and the momentum confirmation (`rsi > 50`) compose as ordinary boolean
expressions; nothing leaves Polars until you ask for it.

## 2. Where the result lands: one naming rule

An expression needs a column name to land on, and `pomata` resolves it with one rule: **the output keeps the root
name of its input column, and only an alias on the returned expression renames it**.

The rule has three consequences worth seeing once.

**A single-input function lands on its input's column.** `sma(pl.col("close"), 3)` is named `close`, so in a
`with_columns` it *replaces* the prices — until you alias the result:

```{doctest}
>>> prices = pl.DataFrame({"close": [10.0, 11.0, 12.0, 11.5, 13.0, 14.0]})
>>> prices.with_columns(sma(pl.col("close"), 3))  # lands on "close": the prices are gone
shape: (6, 1)
┌───────────┐
│ close     │
│ ---       │
│ f64       │
╞═══════════╡
│ null      │
│ null      │
│ 11.0      │
│ 11.5      │
│ 12.166667 │
│ 12.833333 │
└───────────┘
>>> prices.with_columns(sma(pl.col("close"), 3).alias("sma_3"))
shape: (6, 2)
┌───────┬───────────┐
│ close ┆ sma_3     │
│ ---   ┆ ---       │
│ f64   ┆ f64       │
╞═══════╪═══════════╡
│ 10.0  ┆ null      │
│ 11.0  ┆ null      │
│ 12.0  ┆ 11.0      │
│ 11.5  ┆ 11.5      │
│ 13.0  ┆ 12.166667 │
│ 14.0  ┆ 12.833333 │
└───────┴───────────┘
```

**A multi-input function lands on one fixed role column — not always the first argument.**
{py:func}`~pomata.indicators.balance_of_power` reads `open`, `high`, `low`, `close` and lands on `close`;
{py:func}`~pomata.indicators.donchian_channels` reads `high`, `low` and lands on `low`. Each function's landing column 
is deterministic, but it is a role column that says nothing about what the result *is* — which is why a multi-input
result wants an alias:

```{doctest}
>>> from pomata.indicators import balance_of_power
>>>
>>> bars = pl.DataFrame(
...     {
...         "open":  [10.0, 11.2, 11.8, 12.4],
...         "high":  [11.5, 12.0, 12.5, 12.6],
...         "low":   [ 9.5, 10.8, 11.4, 11.2],
...         "close": [11.0, 11.6, 12.4, 11.5],
...     }
... )
>>> bop = balance_of_power(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close"))
>>> bars.select(bop.round(4))  # lands on "close" — a role name, not a description
shape: (4, 1)
┌─────────┐
│ close   │
│ ---     │
│ f64     │
╞═════════╡
│ 0.5     │
│ 0.3333  │
│ 0.5455  │
│ -0.6429 │
└─────────┘
>>> bars.select(bop.round(4).alias("bop"))
shape: (4, 1)
┌─────────┐
│ bop     │
│ ---     │
│ f64     │
╞═════════╡
│ 0.5     │
│ 0.3333  │
│ 0.5455  │
│ -0.6429 │
└─────────┘
```

**An alias on the input is deliberately ignored.** Every factory closes with `name.keep`, which restores the input's
root name — so an aliased input can never silently land the result on an unexpected column, and the only spelling that
renames an output is the alias on the returned expression:

```{doctest}
>>> prices.select(rsi(pl.col("close").alias("momentum"), 3).round(4))  # the input alias does not survive
shape: (6, 1)
┌─────────┐
│ close   │
│ ---     │
│ f64     │
╞═════════╡
│ null    │
│ null    │
│ null    │
│ 80.0    │
│ 89.4737 │
│ 92.8571 │
└─────────┘
>>> prices.select(rsi(pl.col("close"), 3).round(4).alias("momentum"))  # the output alias does
shape: (6, 1)
┌──────────┐
│ momentum │
│ ---      │
│ f64      │
╞══════════╡
│ null     │
│ null     │
│ null     │
│ 80.0     │
│ 89.4737  │
│ 92.8571  │
└──────────┘
```

None of this rests on convention alone: every factory ends by restoring the input's root name with `name.keep`, and
every landing shown above is a CI-executed doctest.

## 3. `.over` for multi-asset panels

A stacked multi-ticker frame is the classic trap: a window that spills from one symbol's tail into the next fabricates
signals that never existed. Wrap the call in `.over("ticker")` and each group is computed on its own — windows and
recursions restart at every boundary.

Here is the leak itself:

```{doctest}
>>> from pomata.indicators import ema
>>>
>>> panel = pl.DataFrame(
...     {
...         "ticker": ["AAPL"] * 4 + ["GOOG"] * 4,
...         "close": [10.0, 11.0, 12.0, 13.0, 100.0, 90.0, 95.0, 105.0],
...     }
... )
>>> panel.with_columns(
...     clean=ema(pl.col("close"), 3).over("ticker").round(4),
...     leaky=ema(pl.col("close"), 3).round(4),
... )
shape: (8, 4)
┌────────┬───────┬───────┬───────┐
│ ticker ┆ close ┆ clean ┆ leaky │
│ ---    ┆ ---   ┆ ---   ┆ ---   │
│ str    ┆ f64   ┆ f64   ┆ f64   │
╞════════╪═══════╪═══════╪═══════╡
│ AAPL   ┆ 10.0  ┆ null  ┆ null  │
│ AAPL   ┆ 11.0  ┆ null  ┆ null  │
│ AAPL   ┆ 12.0  ┆ 11.0  ┆ 11.0  │
│ AAPL   ┆ 13.0  ┆ 12.0  ┆ 12.0  │
│ GOOG   ┆ 100.0 ┆ null  ┆ 56.0  │
│ GOOG   ┆ 90.0  ┆ null  ┆ 73.0  │
│ GOOG   ┆ 95.0  ┆ 95.0  ┆ 84.0  │
│ GOOG   ┆ 105.0 ┆ 100.0 ┆ 94.5  │
└────────┴───────┴───────┴───────┘
```

With `.over`, ticker `GOOG` opens its own warm-up (`null, null`) and its first average, `95.0`, is built from `GOOG`
bars alone. Without it, `GOOG`'s first value is `56.0` — contaminated by `AAPL`'s tail: the cross-asset leak, made
visible.

:::{admonition} Reuse the column, not the expression
:class: tip
A handful of sequential functions run a hand-written Python kernel on every evaluation, because their recursion has
no native Polars form: among the indicators the Ehlers cycle family, {py:func}`~pomata.indicators.kama`,
{py:func}`~pomata.indicators.parabolic_sar`, {py:func}`~pomata.indicators.supertrend`,
{py:func}`~pomata.indicators.fisher_transform`, and the seeded EMA family — with everything built on it:
{py:func}`~pomata.indicators.rsi`, {py:func}`~pomata.indicators.atr`, {py:func}`~pomata.indicators.macd`, the ADX
family, {py:func}`~pomata.indicators.trix`; among the metrics {py:func}`~pomata.metrics.skewness_rolling`,
{py:func}`~pomata.metrics.kurtosis_rolling`, and {py:func}`~pomata.metrics.probabilistic_sharpe_ratio`'s normal CDF.

What that costs you: writing the same expression twice runs the kernel twice — a Python kernel is opaque to Polars'
subexpression caching, which in any case does not reach inside `.over(...)` — so each textual copy pays the full
price again.

What to do instead: materialize the expensive expression **once**, as a column, and derive every downstream output
from that column — reusing a column is free, reusing the expression re-runs the kernel:

```python
kama_ = kama(pl.col("close"), window=10, window_fast=2, window_slow=30).over("ticker")
frame.with_columns(k=kama_).with_columns(          # one evaluation, landed as a column
    above=pl.col("close") > pl.col("k"),           # derived from the column: free
    below=pl.col("close") < pl.col("k"),
)
```
:::

## 4. Warm-up is `null`, never fabricated

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

A seeded warm-up value is a guess wearing the dtype of a measurement. The `null` means *not enough data yet* — and
`null` propagates, so a signal built on top stays `null` too instead of firing off an invented seed.

## 5. No hidden shifts — your `.shift(1)` is the only timing step

A signal computed at the close of bar *t* must not be acted on until bar *t+1*; getting that alignment wrong is
look-ahead that flatters every backtest.

To be clear about who does what: `pomata` does not prevent look-ahead for you — preventing it is the `.shift(1)` *you*
write on the signal, and nothing else is needed.

What `pomata` guarantees is the other half: **no function shifts its output internally**, so there is no hidden
alignment to reason about, and the one shift you wrote is the whole story — no off-by-one stacking up behind the scenes:

```{doctest}
>>> bars = pl.DataFrame({"close": [10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.5, 12.0]})
>>> signal = (rsi(pl.col("close"), 3) > 50.0).cast(pl.Int8)
>>> bars.with_columns(
...     decided_at_close=signal,
...     acted_next_bar=signal.shift(1),
... ).tail(5)
shape: (5, 3)
┌───────┬──────────────────┬────────────────┐
│ close ┆ decided_at_close ┆ acted_next_bar │
│ ---   ┆ ---              ┆ ---            │
│ f64   ┆ i8               ┆ i8             │
╞═══════╪══════════════════╪════════════════╡
│ 11.0  ┆ 1                ┆ null           │
│ 10.0  ┆ 0                ┆ 1              │
│ 9.0   ┆ 0                ┆ 0              │
│ 10.5  ┆ 1                ┆ 0              │
│ 12.0  ┆ 1                ┆ 1              │
└───────┴──────────────────┴────────────────┘
```

`acted_next_bar` is `decided_at_close` slid one bar forward — the decision lands where it can actually be filled, and
the warm-up `null` never becomes a phantom position.

## 6. Multi-output indicators return a `pl.Struct`

Anything with several lines — {py:func}`~pomata.indicators.bollinger_bands`, {py:func}`~pomata.indicators.macd`,
{py:func}`~pomata.indicators.stochastic_slow`, {py:func}`~pomata.indicators.ichimoku` — returns one struct column.
Pick a line with `.struct.field(...)`, or expand them all with `.struct.unnest()`:

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

Where next: the ideas applied end to end on real data — {doc}`tutorial`; the proof they hold — {doc}`correctness`.
