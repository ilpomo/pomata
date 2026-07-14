# Troubleshooting

Sorted by what you actually see, not by cause. Most of these are `pomata` behaving exactly as designed — the fix is
usually one line.

## It raised `TypeError` / `ValueError` before I even called `.collect()`

That is the input validation, and it is eager on purpose: every factory checks its arguments at expression-build
time, not at execution time, so a bad call fails on the line that wrote it. A `TypeError` means an argument that
should be an expression is not one — pass `pl.col("close")`, not the bare string `"close"`. A `ValueError` means a
scalar parameter is outside its documented domain: a `window` below its minimum, a negative `rate`, a `confidence`
outside `(0, 1)`, a non-finite knob. The message names the parameter and the bound, and each function's `Raises`
section lists exactly what it checks.

## My output is all `null` (or null for far longer than I expected)

That is the {term}`warm-up`. A window of length `n` cannot produce a value until it has seen `n` observations, so the first
`n - 1` rows are `null` — and the chained averages stack that up: `dema` warms up over `2 * (n - 1)` rows, `t3` over
`6 * (n - 1)`, the cycle indicators over 32 or 63 rows depending on the measure. If the *whole* column is `null`, your series is shorter than the
warm-up the function owes. This is not a bug and it is not negotiable: seeding a window with a fabricated value would
be a lie that compounds downstream. Each function documents its exact warm-up length.

A second, sneakier cause: a `null` in the *input*. A leading `null` does not consume warm-up budget, and an interior
`null` propagates through a recursion — so a column with gaps warms up later than a clean one.

## My multi-asset result is contaminated across tickers

You left off `.over`. Without it, a window or a recursion runs down the entire column and happily averages the end of
one ticker into the start of the next.

```python
# wrong: AAPL's last bars leak into GOOG's first
frame.with_columns(sma=sma(pl.col("close"), 20))

# right: each ticker is computed on its own
frame.with_columns(sma=sma(pl.col("close"), 20).over("ticker"))
```

The same applies to the `.shift(1)` on your signal and to anything else with memory: if it spans bars, it needs the
`.over`. Sort by ticker then time first, so the groups are contiguous.

## A whole metric came back `NaN`

A `NaN` reached it — and a `NaN` is not a missing value, it is a real number that contaminates everything it touches.
One `NaN` in a returns series is enough to turn a Sharpe ratio into `NaN`, because the mean and the standard deviation
both see it. `pomata` will not quietly drop it for you; that would hide a real problem in your data.

Track down where it was born. The usual sources are a `0 / 0` (a flat window, a zero denominator), an `inf - inf`, or
an `inf` that arrived from an upstream divide-by-zero. A `null`, by contrast, is skipped or carried across and will
*not* poison a reduction — so if you meant "missing", make sure you have a `null`, not a `NaN`.

## Eager and lazy give me different numbers

They should not — the suite pins eager and lazy to bit-for-bit agreement on every function, so `pomata` is almost
never the difference. Look upstream first: a different row order reaching the two paths, an unsorted join, or a
non-deterministic source (a set iteration, an unsorted `group_by` without `maintain_order`). Pin the order before the
`pomata` call and the two paths converge.

## A rolling statistic loses precision on an extreme series

Known and documented. A two-pass rolling sum can shed digits when an entire window collapses onto the float-precision
floor of a much larger value that recently passed through it — it takes a deliberately adversarial path (a price
dropping many orders of magnitude bar to bar) to trigger. It is noted on the affected indicators, and the oscillators
with a hard bound are clamped rather than left to drift. {doc}`trust` has the full account.
