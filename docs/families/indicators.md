# Indicators

`pomata.indicators` is the technical-analysis layer — 75 classic studies, from simple moving averages to Ehlers'
cycle measures. Each is a pure `pl.Expr` factory over your OHLCV columns, verified to the `float64` floor against an
independent oracle, so any number of them fuse into a single Polars query with no glue code between the steps.

## What you get

Seventy-five studies across eleven categories. Every name links to its full signature and formula in the API reference.

### Moving average

The trend line under everything — from the simple mean to adaptive and lag-reduced variants.

{py:func}`~pomata.indicators.sma` ·
{py:func}`~pomata.indicators.ema` ·
{py:func}`~pomata.indicators.wma` ·
{py:func}`~pomata.indicators.hma` ·
{py:func}`~pomata.indicators.dema` ·
{py:func}`~pomata.indicators.tema` ·
{py:func}`~pomata.indicators.trima` ·
{py:func}`~pomata.indicators.kama` ·
{py:func}`~pomata.indicators.t3` ·
{py:func}`~pomata.indicators.rma` ·
{py:func}`~pomata.indicators.vwma`

### Momentum

Rate-of-change and overbought/oversold oscillators — how fast price is moving, and whether it is stretched.

{py:func}`~pomata.indicators.rsi` ·
{py:func}`~pomata.indicators.macd` ·
{py:func}`~pomata.indicators.mom` ·
{py:func}`~pomata.indicators.roc` ·
{py:func}`~pomata.indicators.trix` ·
{py:func}`~pomata.indicators.cci` ·
{py:func}`~pomata.indicators.williams_r` ·
{py:func}`~pomata.indicators.awesome_oscillator` ·
{py:func}`~pomata.indicators.aroon` ·
{py:func}`~pomata.indicators.aroon_oscillator` ·
{py:func}`~pomata.indicators.absolute_price_oscillator` ·
{py:func}`~pomata.indicators.percentage_price_oscillator` ·
{py:func}`~pomata.indicators.balance_of_power` ·
{py:func}`~pomata.indicators.chande_momentum_oscillator` ·
{py:func}`~pomata.indicators.fisher_transform` ·
{py:func}`~pomata.indicators.rsi_stochastic` ·
{py:func}`~pomata.indicators.ultimate_oscillator`

### Volatility

How much each bar moves — the True Range and its smoothed and banded forms.

{py:func}`~pomata.indicators.true_range` ·
{py:func}`~pomata.indicators.atr` ·
{py:func}`~pomata.indicators.atr_normalized` ·
{py:func}`~pomata.indicators.bollinger_bands`

### Channel

Envelopes around price — rolling extremes, ATR bands, and the Ichimoku cloud.

{py:func}`~pomata.indicators.midpoint` ·
{py:func}`~pomata.indicators.midprice` ·
{py:func}`~pomata.indicators.donchian_channels` ·
{py:func}`~pomata.indicators.keltner_channels` ·
{py:func}`~pomata.indicators.ichimoku`

### Directional movement

Wilder's trend-strength system — the directional indicators, their spread, and the smoothed ADX.

{py:func}`~pomata.indicators.dm_plus` ·
{py:func}`~pomata.indicators.dm_minus` ·
{py:func}`~pomata.indicators.di_plus` ·
{py:func}`~pomata.indicators.di_minus` ·
{py:func}`~pomata.indicators.dx` ·
{py:func}`~pomata.indicators.adx` ·
{py:func}`~pomata.indicators.adxr` ·
{py:func}`~pomata.indicators.vortex`

### Stochastic

Where the close sits within its recent high-low range.

{py:func}`~pomata.indicators.stochastic_fast` ·
{py:func}`~pomata.indicators.stochastic_slow`

### Trend

Trailing-stop trend followers that flip with the move.

{py:func}`~pomata.indicators.parabolic_sar` ·
{py:func}`~pomata.indicators.supertrend`

### Price transform

Single-bar summaries — the representative price an indicator consumes.

{py:func}`~pomata.indicators.price_average` ·
{py:func}`~pomata.indicators.price_median` ·
{py:func}`~pomata.indicators.price_typical` ·
{py:func}`~pomata.indicators.price_weighted_close`

### Statistic

Rolling regression and dispersion — slope, forecast, variance, and standard deviation.

{py:func}`~pomata.indicators.linear_regression` ·
{py:func}`~pomata.indicators.linear_regression_slope` ·
{py:func}`~pomata.indicators.linear_regression_angle` ·
{py:func}`~pomata.indicators.linear_regression_intercept` ·
{py:func}`~pomata.indicators.time_series_forecast` ·
{py:func}`~pomata.indicators.variance_rolling` ·
{py:func}`~pomata.indicators.variance_ewma` ·
{py:func}`~pomata.indicators.standard_deviation_rolling` ·
{py:func}`~pomata.indicators.standard_deviation_ewma`

### Volume

Flow indicators that weight price by traded size.

{py:func}`~pomata.indicators.obv` ·
{py:func}`~pomata.indicators.vwap` ·
{py:func}`~pomata.indicators.accumulation_distribution` ·
{py:func}`~pomata.indicators.accumulation_distribution_oscillator` ·
{py:func}`~pomata.indicators.chaikin_money_flow` ·
{py:func}`~pomata.indicators.money_flow_index`

### Cycle

Ehlers' Hilbert-transform measures — the dominant cycle, its phase, and the trend/cycle mode.

{py:func}`~pomata.indicators.hilbert_phasor` ·
{py:func}`~pomata.indicators.hilbert_trendline` ·
{py:func}`~pomata.indicators.dominant_cycle_period` ·
{py:func}`~pomata.indicators.dominant_cycle_phase` ·
{py:func}`~pomata.indicators.sine_wave` ·
{py:func}`~pomata.indicators.mama` ·
{py:func}`~pomata.indicators.trend_mode`

## Common pains, solved

### One query, not five round-trips

A trend filter, a momentum gate, and a volatility read usually mean three passes over the frame and three
intermediate DataFrames to stitch back together. Because every indicator is a `pl.Expr`, they all live in one lazy
pipeline — the optimizer fuses the scan, and a single `.collect()` materializes the lot:

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
>>> signals.select(pl.col("close"), pl.col("vol").round(4), pl.col("long"))
shape: (8, 3)
┌───────┬────────┬──────┐
│ close ┆ vol    ┆ long │
│ ---   ┆ ---    ┆ ---  │
│ f64   ┆ f64    ┆ bool │
╞═══════╪════════╪══════╡
│ 9.5   ┆ null   ┆ null │
│ 10.5  ┆ null   ┆ null │
│ 11.5  ┆ 1.3333 ┆ null │
│ 11.0  ┆ 1.3889 ┆ true │
│ 12.5  ┆ 1.5926 ┆ true │
│ 13.5  ┆ 1.5617 ┆ true │
│ 13.0  ┆ 1.5412 ┆ true │
│ 14.5  ┆ 1.6941 ┆ true │
└───────┴────────┴──────┘
```

The regime filter (`fast > slow`) and the momentum confirmation (`rsi > 50`) compose as ordinary boolean
expressions; nothing leaves Polars until you ask for it.

### A panel that can't leak across assets

A stacked multi-ticker frame is the classic trap: a window that spills from one symbol's tail into the next fabricates
signals that never existed. Wrap the call in `.over("ticker")` and each group is reduced on its own — windows and
recursions restart at every boundary:

```{doctest}
>>> from pomata.indicators import ema
>>>
>>> panel = pl.DataFrame(
...     {
...         "ticker": ["A", "A", "A", "A", "B", "B", "B", "B"],
...         "close":  [10.0, 11.0, 12.0, 13.0, 100.0, 90.0, 95.0, 105.0],
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
│ A      ┆ 10.0  ┆ null  ┆ null  │
│ A      ┆ 11.0  ┆ null  ┆ null  │
│ A      ┆ 12.0  ┆ 11.0  ┆ 11.0  │
│ A      ┆ 13.0  ┆ 12.0  ┆ 12.0  │
│ B      ┆ 100.0 ┆ null  ┆ 56.0  │
│ B      ┆ 90.0  ┆ null  ┆ 73.0  │
│ B      ┆ 95.0  ┆ 95.0  ┆ 84.0  │
│ B      ┆ 105.0 ┆ 100.0 ┆ 94.5  │
└────────┴───────┴───────┴───────┘
```

With `.over`, ticker `B` opens its own warm-up (`None, None`) and prices in from `100.0`. Without it, `B`'s first
value is `56.0` — a number contaminated by `A`'s tail, the cross-asset leak made visible.

### A signal that can't peek at the future

A signal computed at the close of bar *t* must not be acted on until bar *t+1*; getting that alignment wrong is
look-ahead that flatters every backtest. `pomata` makes it mechanical: the warm-up is `null` (never a fabricated
value to trade on), and a single `.shift(1)` moves a close-computed decision onto the next bar:

```{doctest}
>>> bars = pl.DataFrame({"close": [10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.5, 12.0]})
>>> signal = (rsi(pl.col("close"), 3) > 50.0).cast(pl.Int8)
>>> res = bars.with_columns(
...     decided_at_close=signal,
...     acted_next_bar=signal.shift(1),
... )
>>> res
shape: (8, 3)
┌───────┬──────────────────┬────────────────┐
│ close ┆ decided_at_close ┆ acted_next_bar │
│ ---   ┆ ---              ┆ ---            │
│ f64   ┆ i8               ┆ i8             │
╞═══════╪══════════════════╪════════════════╡
│ 10.0  ┆ null             ┆ null           │
│ 11.0  ┆ null             ┆ null           │
│ 12.0  ┆ null             ┆ null           │
│ 11.0  ┆ 1                ┆ null           │
│ 10.0  ┆ 0                ┆ 1              │
│ 9.0   ┆ 0                ┆ 0              │
│ 10.5  ┆ 1                ┆ 0              │
│ 12.0  ┆ 1                ┆ 1              │
└───────┴──────────────────┴────────────────┘
```

`acted_next_bar` is `decided_at_close` slid one bar forward — the decision lands where it can actually be filled, and
the warm-up `null` never becomes a phantom position.

### Multi-line indicators, one struct column

Bollinger Bands, MACD, and the stochastics are several series at once. Rather than return a loose tuple you have to
re-align, `pomata` packs them into a single `pl.Struct`: pull one line with `.struct.field(...)`, or fan them all out
into columns with `.struct.unnest()`:

```{doctest}
>>> from pomata.indicators import macd
>>>
>>> frame = pl.DataFrame({"close": [10.0, 11.0, 12.0, 11.0, 12.0, 13.0, 14.0, 13.0, 15.0, 16.0]})
>>> lines = macd(pl.col("close"), window_fast=2, window_slow=4, window_signal=2)
>>> frame.select("close", lines.struct.field("histogram").round(4).alias("hist"))
shape: (10, 2)
┌───────┬─────────┐
│ close ┆ hist    │
│ ---   ┆ ---     │
│ f64   ┆ f64     │
╞═══════╪═════════╡
│ 10.0  ┆ null    │
│ 11.0  ┆ null    │
│ 12.0  ┆ null    │
│ 11.0  ┆ null    │
│ 12.0  ┆ 0.0778  │
│ 13.0  ┆ 0.0965  │
│ 14.0  ┆ 0.0877  │
│ 13.0  ┆ -0.1108 │
│ 15.0  ┆ 0.0879  │
│ 16.0  ┆ 0.0849  │
└───────┴─────────┘
>>> frame.select(lines.alias("macd")).unnest("macd").with_columns(pl.all().round(4))
shape: (10, 3)
┌────────┬────────┬───────────┐
│ macd   ┆ signal ┆ histogram │
│ ---    ┆ ---    ┆ ---       │
│ f64    ┆ f64    ┆ f64       │
╞════════╪════════╪═══════════╡
│ null   ┆ null   ┆ null      │
│ null   ┆ null   ┆ null      │
│ null   ┆ null   ┆ null      │
│ 0.1667 ┆ null   ┆ null      │
│ 0.3222 ┆ 0.2444 ┆ 0.0778    │
│ 0.5341 ┆ 0.4375 ┆ 0.0965    │
│ 0.7007 ┆ 0.613  ┆ 0.0877    │
│ 0.2805 ┆ 0.3913 ┆ -0.1108   │
│ 0.655  ┆ 0.5671 ┆ 0.0879    │
│ 0.8219 ┆ 0.737  ┆ 0.0849    │
└────────┴────────┴───────────┘
```

One expression, one column, three lines inside it — aligned to the same index by construction, so there is no
re-joining and no off-by-one between the `macd`, `signal`, and `histogram` series.
