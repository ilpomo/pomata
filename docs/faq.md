# FAQ

Short answers to the questions that come up before you write any code.

## Which columns does pomata expect?

None, by name. Every function takes a `pl.Expr`, not a column name — you write `pl.col("close")`, so the column is
yours to call whatever you like. The docstrings say "high", "low", "close" because that is the convention, not a
requirement; rename freely and pass the matching expression.

## Lazy or eager — does it matter?

It does not. A function returns the same `pl.Expr` either way, so it runs inside a `DataFrame.select` or a
`LazyFrame.with_columns` without a single change. Build on a `LazyFrame` when the job is large and let Polars fuse and
stream it; `.collect()` when you want the result. The numbers are identical to the bit — the test suite checks exactly
that.

## Do I have to shift the signal myself?

Yes, and that is deliberate. `pomata` never shifts anything for you: a signal computed on bar *t* sits on bar *t*, and
the decision to act on the *next* bar is yours to make with `.shift(1)`. Putting that one call in your own code is
what makes the no-look-ahead choice visible and auditable instead of buried in a library. See the
[tutorial](tutorial.md) for it in context.

## What `periods_per_year` should I pass?

However many bars a year holds at *your* sampling. Daily bars on trading days: `252`. Weekly: `52`. Monthly: `12`.
Hourly on a 6.5-hour session: about `1638`. It is only an annualization factor for the ratios and CAGR — it never
resamples your data or guesses the frequency, so it has to match the bars you actually pass.

## What is the difference between `null` and `NaN`?

A `null` is missing data; a `NaN` is a real floating-point value that happens to be "not a number". `pomata` keeps the
two apart, the way Polars does, because they mean different things: a `null` is a gap to skip or carry across, while a
`NaN` is a value that propagates and will poison a sum or a reduction it touches. If a metric comes back `NaN`, you
have a real `NaN` somewhere upstream — not a missing value. [trust](trust.md) states the rule per family; each
function's docstring **Note** spells out its own exact null/NaN contract.

## Which Python and Polars versions are supported?

Python 3.12 and newer, and Polars 1.39 or newer — one runtime dependency, nothing else. The Polars floor only moves
when something genuinely needs it, and a CI job proves the floor still builds on every run.

## How do I install it?

`pip install pomata`, or `uv add pomata`. See [installation](installation.md).

## Why Polars only — no pandas, no NumPy?

Because the whole design falls out of it. One dependency to vet, expressions that compose instead of objects that
wrap your data, and lazy execution and streaming for free. You stay in one engine from the raw bars to the final
metric, with nothing to convert in between.
