# Indicators

`pomata.indicators` is the technical-analysis layer — 75 classic studies, from simple moving averages to Ehlers'
cycle measures.

Each is a pure `pl.Expr` factory over your OHLCV columns, verified to the `float64` floor against an oracle — an
independent, deliberately naive second implementation for all but a handful of irreducibly-sequential studies, which
rest on hand-checked golden masters and the TA-Lib differential instead — so any number of them fuse into a single
Polars query with no glue code between the steps.

## What you get

Seventy-five studies across eleven categories. Every name links to its full signature and definition in the API
reference.

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

Wilder's trend-strength system — the directional indicators, their spread, and the smoothed ADX — plus the Vortex
oscillator.

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

## Conventions

Everything an indicator call assumes, in one place — each stated in full on the function's page in the
{doc}`API Reference <../api/index>`.

- **Column roles are explicit.** An indicator takes a `pl.Expr` for every OHLCV role it needs (`high`, `low`,
  `close`, `volume`, …) — pass `pl.col(...)` for each; nothing is inferred from column names.
- **Warm-up is `null`.** Until the longest look-back fills, the output is `null` — never zero, never a fabricated
  seed ({doc}`Design <../design>`, idea 4). Each docstring's Returns section states the exact leading-`null` count.
- **Multi-line studies return one `pl.Struct`.** {py:func}`~pomata.indicators.bollinger_bands`,
  {py:func}`~pomata.indicators.macd`, the stochastics, {py:func}`~pomata.indicators.ichimoku` and their
  kin come back as a single struct column: pick a line with `.struct.field(...)` or fan them out with `.unnest(...)`
  ({doc}`Design <../design>`, idea 6, shows both).
- **The TA-Lib relation is declared per function.** 58 indicators match the C reference bar for bar (re-proven by
  the non-gating differential tier), 3 keep a documented charting-authority convention, and 14 have no TA-Lib twin —
  each docstring states its side, and the method lives on {doc}`Correctness <../correctness>`.

Where next: the composable grammar, the panel `.over` story, and the no-look-ahead contract live in
{doc}`Design <../design>`; the end-to-end journey is the {doc}`Tutorial <../tutorial>`; measured speed is on
{doc}`Benchmarks <../benchmarks/index>`.
