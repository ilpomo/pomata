# Glossary

The house vocabulary, in one place. Every term below is used with exactly this meaning across the API docstrings,
the trust pages, and [`CORRECTNESS.md`](https://github.com/ilpomo/pomata/blob/main/CORRECTNESS.md); link a term
anywhere in the docs with the `{term}` role.

```{glossary}
warm-up
: The leading rows a function returns as `null` because its window or recursion has not yet seen enough data. Always
  `null`, never a fabricated value — a fabricated warm-up is a silent look-ahead. Each docstring states the exact
  count (e.g. `window - 1` for a rolling mean, `2(window - 1)` for a DEMA).

oracle
: The second, deliberately naive implementation of a function's published formula, sharing no code with the shipped
  `pl.Expr`. The two must agree to the stated tolerance on every test input, or the build is red. It lives in
  `tests/*/oracles/` and exists only to be the independent witness.

golden master
: A frozen expected output (a list of numbers in a test), pinned so a value can never drift between versions without
  a red build. The hand-computed ones encode a published example or a by-hand derivation; the rest freeze a
  previously verified run.

differential tier
: The non-gating test tier comparing each indicator with a TA-Lib counterpart against the C reference, bar for bar
  from the first defined value (a documented minority only on the converged tail). Red there flags a divergence to
  investigate, never a blocked merge.

reducing / elementwise
: A **reducing** expression folds a whole series (or group) into one value — the shape of most metrics. An
  **elementwise** expression emits one value per row — the shape of every indicator and pnl primitive, including the
  `*_rolling` metric twins.

panel
: One long frame holding many instruments, distinguished by a key column (`ticker`). Wrap any call in
  `.over("ticker")` and every window, recursion, and reduction restarts per instrument.

return flow / cash flow
: The two pnl families. The **return flow** holds a signed `weight` (a fraction of capital) and asset returns —
  dimensionless, compounding via {py:func}`~pomata.pnl.equity_curve`. The **cash flow** holds a `quantity` of units
  at a `price` — currency-denominated, additive via {py:func}`~pomata.pnl.cumulative_pnl`, where multipliers,
  dividends, and funding can be booked honestly.

null policy / NaN policy
: A function's declared, machine-verified behavior for an interior missing value (`null`) or an interior `NaN` —
  for `null`: skipped, absorbed, propagated, in-window-nulled, bridged, or latched; for `NaN`: poisoned, propagated,
  or latched. The vocabulary and the per-function declaration live in the package's policy registry
  (`src/pomata/_policy.py`); the proof lives in the test suite's policy-dispatched flow rungs
  (`tests/test_ladder.py`); the API docstrings state each function's pair in prose.

rung / ladder
: One generic test function in the canonical four-tier layout (Contract → Edge → Correctness → Properties),
  parametrized over every function it applies to (`tests/test_ladder.py`): one rung = one guarantee, checked
  identically across the whole public surface by construction.

conditioning
: How much a statistic amplifies floating-point rounding on a degenerate input (a near-constant window, a vanishing
  denominator). Tolerances are sized to each statistic's conditioning, and the genuinely ill-conditioned regimes are
  documented limits, not hidden ones — see
  [`CORRECTNESS.md`](https://github.com/ilpomo/pomata/blob/main/CORRECTNESS.md).
```
