# Metrics

`pomata.metrics` is the scoring layer — sixty performance and risk statistics, each a reducing `pl.Expr` that
collapses a series to a single number (with the running `drawdown` and each `*_rolling` twin the series-valued
exceptions, one value per row or per window). Point one at a return series
(`pomata.pnl.returns_net`) or an equity curve (`pomata.pnl.equity_curve`) and it folds the whole history into the
figure you report — in the same Polars query that produced the series, `null`-skipping and `.over`-partitioning
included.

## What you get

Sixty metrics in five themes. Every name links to its full API entry; each `*_rolling` twin is the windowed form of
the reducer it sits beside.

### Drawdown

How far below its high-water mark the curve fell, for how long, and in what shape — point these at an equity curve.

{py:func}`~pomata.metrics.drawdown` ·
{py:func}`~pomata.metrics.max_drawdown` ·
{py:func}`~pomata.metrics.max_drawdown_duration` ·
{py:func}`~pomata.metrics.drawdown_rolling` ·
{py:func}`~pomata.metrics.pain_index` ·
{py:func}`~pomata.metrics.ulcer_index` ·
{py:func}`~pomata.metrics.conditional_drawdown_at_risk`

### Performance

Total and annualized growth, and whether the equity line was a steady climb or a lucky spike.

{py:func}`~pomata.metrics.total_return` ·
{py:func}`~pomata.metrics.cagr` ·
{py:func}`~pomata.metrics.stability` ·
{py:func}`~pomata.metrics.total_return_rolling` ·
{py:func}`~pomata.metrics.cagr_rolling`

### Ratio

Return per unit of risk — the reward-to-risk family, most dividing a growth measure by a dispersion, drawdown, or
tail measure (with `probabilistic_sharpe_ratio` the confidence that the Sharpe Ratio is real, in `[0, 1]`).

{py:func}`~pomata.metrics.sharpe_ratio` ·
{py:func}`~pomata.metrics.sortino_ratio` ·
{py:func}`~pomata.metrics.calmar_ratio` ·
{py:func}`~pomata.metrics.adjusted_sharpe_ratio` ·
{py:func}`~pomata.metrics.probabilistic_sharpe_ratio` ·
{py:func}`~pomata.metrics.omega_ratio` ·
{py:func}`~pomata.metrics.burke_ratio` ·
{py:func}`~pomata.metrics.sterling_ratio` ·
{py:func}`~pomata.metrics.gain_to_pain_ratio` ·
{py:func}`~pomata.metrics.pain_ratio` ·
{py:func}`~pomata.metrics.common_sense_ratio` ·
{py:func}`~pomata.metrics.recovery_ratio` ·
{py:func}`~pomata.metrics.ulcer_performance_ratio` ·
{py:func}`~pomata.metrics.sharpe_ratio_rolling` ·
{py:func}`~pomata.metrics.sortino_ratio_rolling` ·
{py:func}`~pomata.metrics.omega_ratio_rolling`

### Relative

Performance measured against a benchmark — what the strategy added, and how much of the market's moves it captured.

{py:func}`~pomata.metrics.alpha` ·
{py:func}`~pomata.metrics.beta` ·
{py:func}`~pomata.metrics.treynor_ratio` ·
{py:func}`~pomata.metrics.information_ratio` ·
{py:func}`~pomata.metrics.modigliani_risk_adjusted_performance` ·
{py:func}`~pomata.metrics.capture_ratio` ·
{py:func}`~pomata.metrics.capture_upside_ratio` ·
{py:func}`~pomata.metrics.capture_downside_ratio` ·
{py:func}`~pomata.metrics.alpha_rolling` ·
{py:func}`~pomata.metrics.beta_rolling` ·
{py:func}`~pomata.metrics.treynor_ratio_rolling` ·
{py:func}`~pomata.metrics.information_ratio_rolling`

### Risk

Dispersion and its one-sided and higher-moment cousins, the value-at-risk trio, and the win / payoff / Kelly
bet-sizing statistics — point these at a return series.

{py:func}`~pomata.metrics.volatility` ·
{py:func}`~pomata.metrics.downside_deviation` ·
{py:func}`~pomata.metrics.skewness` ·
{py:func}`~pomata.metrics.kurtosis` ·
{py:func}`~pomata.metrics.value_at_risk` ·
{py:func}`~pomata.metrics.value_at_risk_parametric` ·
{py:func}`~pomata.metrics.value_at_risk_modified` ·
{py:func}`~pomata.metrics.conditional_value_at_risk` ·
{py:func}`~pomata.metrics.tail_ratio` ·
{py:func}`~pomata.metrics.win_rate` ·
{py:func}`~pomata.metrics.payoff_ratio` ·
{py:func}`~pomata.metrics.profit_ratio` ·
{py:func}`~pomata.metrics.kelly_criterion` ·
{py:func}`~pomata.metrics.risk_of_ruin` ·
{py:func}`~pomata.metrics.volatility_rolling` ·
{py:func}`~pomata.metrics.downside_deviation_rolling` ·
{py:func}`~pomata.metrics.skewness_rolling` ·
{py:func}`~pomata.metrics.kurtosis_rolling` ·
{py:func}`~pomata.metrics.tail_ratio_rolling` ·
{py:func}`~pomata.metrics.value_at_risk_rolling`

## Common pains, solved

### Annualization, done right

A volatility or a Sharpe Ratio is meaningless without a frequency. Pass `periods_per_year` and the square-root-of-time
scaling is applied for you — never a hand-multiplied `* sqrt(252)` you can forget or get wrong.

```{doctest}
>>> import polars as pl
>>> from datetime import datetime
>>> from pomata.metrics import volatility
>>>
>>> frame = (
...     pl.DataFrame(
...         {
...             "datetime": [datetime(2024, 1, d, 17) for d in (2, 3, 4, 5, 8, 9)],
...             "ticker": "AAPL",
...             "returns": [0.01, -0.02, 0.015, 0.005, -0.01, 0.02],
...         }
...     )
...     .with_columns(pl.col("datetime").dt.replace_time_zone("America/New_York"))
... )
>>> annual = frame.select(volatility(pl.col("returns"), periods_per_year=252)).item()
>>> round(annual, 4)
0.2442
>>> per_bar = frame.select(pl.col("returns").std()).item()   # the same number, scaled by hand
>>> round(per_bar * 252 ** 0.5, 4)
0.2442
```

The annualized figure is exactly the per-bar dispersion times `sqrt(252)` — `pomata` hands it to you, so the factor
lives in one argument instead of scattered across your call sites.

### A single `NaN`, made visible

One bad print in a feed can silently rewrite a metric. `pomata` draws a hard line: a `null` is a gap and is skipped,
but a non-null `NaN` is corrupt data and poisons the result — so you see the problem instead of a plausible lie.

```{doctest}
>>> from pomata.metrics import sharpe_ratio
>>>
>>> clean = (
...     pl.DataFrame(
...         {
...             "datetime": [datetime(2024, 1, d, 17) for d in (2, 3, 4, 5, 8, 9, 10)],
...             "ticker": "AAPL",
...             "returns": [0.03, None, 0.02, -0.015, 0.01, 0.005, -0.02],
...         }
...     )
...     .with_columns(pl.col("datetime").dt.replace_time_zone("America/New_York"))
... )
>>> clean.select(sharpe_ratio(pl.col("returns"), periods_per_year=252).round(4)).item()
4.0717
>>> poisoned = (
...     pl.DataFrame(
...         {
...             "datetime": [datetime(2024, 1, d, 17) for d in (2, 3, 4, 5, 8, 9, 10)],
...             "ticker": "AAPL",
...             "returns": [0.03, None, 0.02, -0.015, float("nan"), 0.005, -0.02],
...         }
...     )
...     .with_columns(pl.col("datetime").dt.replace_time_zone("America/New_York"))
... )
>>> poisoned.select(sharpe_ratio(pl.col("returns"), periods_per_year=252).round(4)).item()
nan
```

The `null` is skipped and the clean series reduces to `4.0717`; swap one value for a `NaN` and the whole metric is
`nan`, loudly, rather than a number that looks fine.

### The full-sample number, and its rolling twin

Sometimes you want one verdict for the whole track record; sometimes you want to watch it drift. Every reducer that
earns it has a `_rolling` twin over a trailing window — the same statistic recomputed per window, except the
`total_return` and `cagr` twins, which as endpoint quantities measure the window's own first-to-last growth (dividing
by the window's first equity, not the unit start).

```{doctest}
>>> from pomata.metrics import sharpe_ratio_rolling
>>>
>>> frame = (
...     pl.DataFrame(
...         {
...             "datetime": [datetime(2024, 1, d, 17) for d in (2, 3, 4, 5, 8, 9, 10, 11)],
...             "ticker": "AAPL",
...             "returns": [0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02, 0.012],
...         }
...     )
...     .with_columns(pl.col("datetime").dt.replace_time_zone("America/New_York"))
... )
>>> frame.select(sharpe_ratio(pl.col("returns"), periods_per_year=252).round(4)).item()
3.6098
>>> frame.with_columns(rolling=sharpe_ratio_rolling(pl.col("returns"), 4, periods_per_year=252).round(4))
shape: (8, 4)
┌────────────────────────────────┬────────┬─────────┬─────────┐
│ datetime                       ┆ ticker ┆ returns ┆ rolling │
│ ---                            ┆ ---    ┆ ---     ┆ ---     │
│ datetime[μs, America/New_York] ┆ str    ┆ f64     ┆ f64     │
╞════════════════════════════════╪════════╪═════════╪═════════╡
│ 2024-01-02 17:00:00 EST        ┆ AAPL   ┆ 0.03    ┆ null    │
│ 2024-01-03 17:00:00 EST        ┆ AAPL   ┆ -0.01   ┆ null    │
│ 2024-01-04 17:00:00 EST        ┆ AAPL   ┆ 0.02    ┆ null    │
│ 2024-01-05 17:00:00 EST        ┆ AAPL   ┆ -0.015  ┆ 4.484   │
│ 2024-01-08 17:00:00 EST        ┆ AAPL   ┆ 0.01    ┆ 1.2011  │
│ 2024-01-09 17:00:00 EST        ┆ AAPL   ┆ 0.005   ┆ 5.3923  │
│ 2024-01-10 17:00:00 EST        ┆ AAPL   ┆ -0.02   ┆ -5.3923 │
│ 2024-01-11 17:00:00 EST        ┆ AAPL   ┆ 0.012   ┆ 1.8776  │
└────────────────────────────────┴────────┴─────────┴─────────┘
```

The reducer folds the eight bars to a single `3.6098`; the twin warms up for `window - 1` bars, then re-evaluates the
same ratio over each trailing window of four.

### Downside risk, not total volatility

The Sharpe Ratio punishes a big up-move exactly as hard as a big down-move — but no one complains about the upside.
`sortino_ratio` divides by the downside deviation only, so a strategy with violent gains and mild losses scores far
higher than its Sharpe Ratio lets on.

```{doctest}
>>> from pomata.metrics import sortino_ratio
>>>
>>> frame = (
...     pl.DataFrame(
...         {
...             "datetime": [datetime(2024, 1, d, 17) for d in (2, 3, 4, 5, 8, 9, 10, 11)],
...             "ticker": "AAPL",
...             "returns": [0.05, -0.01, 0.06, -0.012, 0.04, -0.008, 0.05, -0.01],
...         }
...     )
...     .with_columns(pl.col("datetime").dt.replace_time_zone("America/New_York"))
... )
>>> frame.select(sharpe_ratio(pl.col("returns"), periods_per_year=252).round(4)).item()
9.7595
>>> frame.select(sortino_ratio(pl.col("returns"), periods_per_year=252).round(4)).item()
44.4575
```

The large upside swings inflate the Sharpe Ratio denominator but never touch the Sortino Ratio one — so the
Sortino Ratio, blind to harmless volatility, lands far above the Sharpe Ratio.
