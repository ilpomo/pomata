# Correctness

Most libraries ask you to trust their output. `pomata` proves it. The premise is simple and a little obsessive:
**every function is written twice.** The shipped version is tuned to vectorize in Polars; a second, deliberately
naive *oracle* re-derives the same published formula and shares no code with it. The two must agree — on a fixed
series, on frozen *golden-master* numbers, and on thousands of randomly fuzzed inputs — or the build is red. This
page is the method behind every number: how the agreement is checked, and why each promise is the size it is.

It is also an honest page. It does not claim the code is free of bugs; no one can. It claims something narrower and
checkable: that every function agrees with an independent transcription of its published formula, satisfies a set of
stated mathematical invariants, and matches the industry reference where the two are meant to match.

The same method covers all three families — indicators, PnL, and metrics — but holds each to the standard that
actually catches its bugs. Indicators must be right *to the digit*: they reproduce a fixed formula, so there is one
correct number, and even a public reference — TA-Lib — to check it against. PnL and metrics are simpler arithmetic
whose failures live at the edges — a `null`, a `NaN`, a zero denominator, a warm-up — not in the fifteenth digit. So
the proof below splits in two: indicators to the last figure, PnL and metrics at the boundaries.

## What every function survives

The same four-tier rung set runs over all three families, under **100% branch coverage**:

```{list-table}
:header-rows: 1
:widths: 20 80

* - Tier
  - What it proves
* - **Contract**
  - The output's shape, dtype (`Float64`), and length; that it is a lazy `pl.Expr` (eager and lazy agree to the bit); and that under `.over(...)` each group is computed independently, never spanning a boundary.
* - **Edge**
  - The exact warm-up length, and that a `null` and a `NaN` propagate exactly as documented — across the boundaries: an empty series, a single row, an all-`null` column, a window longer than the data, an interior gap.
* - **Correctness**
  - Agreement with the independent oracle to a relative `1e-10` (most values match to the last bit) on a fixed, realistic series, plus frozen, hand-checked golden-master values.
* - **Properties**
  - The same oracle agreement *and* the mathematical invariants (bounds, scale behavior, monotonicity) over the full `[-1e6, 1e6]` fuzz domain, with missing data freely interleaved — driven by Hypothesis.
```

## Two independent witnesses — and where there is only one

The oracle is the second witness, and it earns that title only when it is genuinely independent. For the great
majority of functions — anything expressible as composed Polars expressions — the naive loop and the vectorized
expression graph are unrelated computations, so a single transcription bug would have to occur *identically* in both
to hide, which is vanishingly unlikely: agreement is evidence. It holds for the exponential averages too (the
unadjusted EMA, Wilder's RMA, and the ATR / RSI / MACD family built on them): the shipped code runs one sequential
pass, but the linear recurrence has a closed form, so the oracle is an independent unrolled weighted sum rather than a
copy of the same loop — a forward-carry error cannot hide in it.

A few indicators are irreducibly sequential with no such closed form, so their oracle necessarily resembles the
shipped pass and confirms internal consistency, not independence — KAMA, the parabolic SAR, the Fisher transform,
SuperTrend, and the Hilbert-transform cycle cluster (the dominant-cycle period and phase, the phasor, the trendline,
the sine wave, the trend flag, and MAMA). For these the second witness is elsewhere, and stated rather than
overstated: the parabolic SAR is anchored to golden masters hand-computed from Wilder's published rules, including
the seeding and reversal branches the recurrence cannot reach by symmetry; the others rest on frozen golden masters
that catch any drift; and KAMA and the cycle cluster are additionally cross-checked against TA-Lib in the non-gating
differential tier.

## Indicators: proven to the digit

Indicators reproduce a fixed formula, so the bar is numeric and absolute. The headline is **ten significant figures**:
every indicator reproduces its oracle to a relative **`1e-10`** on any finite input within a sane dynamic range. That
number is chosen, not guessed — and enforced:

- **It dwarfs the data.** Market feeds carry four to eight significant figures. Ten figures means `pomata` never adds
  error you could observe — lossless against its own input, with orders of headroom to spare.
- **It sits far above the noise floor.** A `float64` holds ~16 figures; honest rounding between the streaming
  implementation and the two-pass oracle lands at the `float64` noise floor. `1e-10` is orders above that — tight
  enough to reject any real coding error, loose enough that a last-bit difference never flakes the suite.
- **It is verified, not asserted.** The property tier holds every indicator to `1e-10` across the full random domain
  — except the one-pass rolling family (whose sliding-window sums and exponential recurrences round differently from
  a fresh two-pass recompute), held to the per-family band each spec declares. In practice the agreement is *far*
  tighter: about **half the outputs match the oracle to the last bit** (a relative difference of exactly zero), and
  the rest land at the noise floor — never fewer than the promised ten figures.

:::{admonition} What pomata does *not* claim
:class: warning
Not the absence of all bugs, and not correctness outside the documented input domain. One known limit: a rolling-sum
statistic loses precision only when an entire window collapses onto the float-precision floor of a much larger value
that recently passed through it — it takes a deliberately adversarial series (a price dropping seven orders of
magnitude bar-to-bar) to build that pattern. It is documented on the affected indicators, and the oscillators whose
bound is unconditional — Chande Momentum Oscillator, Money Flow Index, Chaikin Money Flow — are clamped to that bound rather than
papered over. For the indicators with no public reference, oracle agreement demonstrates internal consistency between
two independent transcriptions of the same published formula — not correctness of the formula against an external
authority.
:::

### The receipt: against the public reference

For indicators there is also a public yardstick — **TA-Lib**, the de-facto industry C implementation, a genuinely
different computation path. Here is `rsi(14)`, the last value of a deterministic 400-bar series, at full `float64`
display precision:

```text
pomata      85.20908701341023
reference   85.20908701341023   ← independent reimplementation: identical, to the last bit
TA-Lib      85.20908701341024   ← fifteen figures identical; differs only at the float64 floor
```

The same five indicators on the same series — each delta the relative residual against that column's reference —
reproducing it exactly or at the `float64` noise floor, never worse than thirteen figures:

```{list-table}
:header-rows: 1
:widths: 28 38 17 17

* - indicator
  - pomata
  - vs reimpl.
  - vs TA-Lib
* - `sma(20)`
  - `105.15146076264764`
  - exact
  - `1e-15`
* - `ema(20)`
  - `107.7299930892346`
  - `1e-15`
  - `1e-16`
* - `rsi(14)`
  - `85.20908701341023`
  - exact
  - `2e-16`
* - `atr(14)`
  - `1.904174462198776`
  - `5e-16`
  - `2e-15`
* - `macd(12,26,9)`
  - `2.523444380829531`
  - `5e-14`
  - `6e-15`
```

The differential tier (non-gating) compares every indicator that has a TA-Lib counterpart against the C reference,
bar for bar from the first defined value. A documented minority is compared only on the converged tail — always with
a reason its warm-up differs from TA-Lib's (Wilder's first True Range, the independent MACD / Chaikin EMAs, the
Ehlers Hilbert pipeline's long warm-up, the Parabolic SAR cold start) — never a steady-state disagreement. A few more
map to a TA-Lib function but keep a deliberate convention of their own (ADXR's averaging lag, the Chande momentum
oscillator's smoothing, OBV's origin): they follow the charting authorities rather than TA-Lib, so they are held out
of the differential on purpose, each divergence documented on the function itself. The indicators with no TA-Lib twin
rest on the independent oracle and their golden masters instead.

## PnL and metrics: proven at the edges, not in the digits

Their math is simple; their correctness lives where the bad things happen, not in the fifteenth digit. The same
machine applies — independent oracle, four-tier rung set, golden masters — but the characteristic invariants change:

- **PnL** — cost monotonicity (more cost, less PnL); the additive-vs-compounded split (`cumulative_pnl` sums currency
  P&L, `equity_curve` compounds returns); no look-ahead (every bar uses only past data); and a defined behavior
  for **every** degenerate input — `null`, `NaN`, `0`, `±inf`, warm-up.
- **Metrics** — the annualization identities; scale-equivariance (a Sharpe Ratio is invariant to leverage); closed-form
  checks (the Sharpe Ratio of constant returns, the drawdown of a monotone series); and a defined behavior for every
  degenerate input — for the reducing (scalar) metrics a `null` is skipped and a non-null `NaN` poisons the result,
  while the rolling (series-shaped) metrics follow the windowed policy (a `null` nulls the windows that
  overlap it, a `NaN` propagates and then recovers) — except the endpoint and peak-tracking series (`cagr_rolling`,
  `total_return_rolling`, the running drawdown), where an interior value never crosses a window boundary and only
  the affected rows are nulled; a degenerate denominator is reported as `±inf`/`NaN`, never silently clipped.

## Tolerances, by conditioning

When the implementation and the oracle agree they still round differently, and how far they drift depends on the
statistic's *conditioning*, not on a round number. Every tolerance is therefore a named constant — sized to the
worst-case residual its conditioning predicts, plus a margin — and they all live in one file
(`tests/support/tolerances.py`). The enforced guarantee is the relative `1e-10` above; the well-conditioned kernels
beat it by orders, and the families whose one-pass form rounds away from a fresh two-pass recompute declare their own
wider band per spec. Where a fixed relative band would be wrong — an output that cancels toward zero, an input at an
extreme magnitude — an absolute floor is sized to the data instead. No band is a bare number: each carries a one-line
reason for its value, so a tolerance can never drift into superstition.

## Re-run it yourself

None of this is something you have to take on faith. The full gate runs from a clean clone (`uv sync`, then the
lint / type / test commands listed in `CONTRIBUTING.md`); the published figures regenerate from
`scripts/precision_table.py`, and the realized headroom under the guarantee from `scripts/calibrate_tolerances.py`.
The suite's own design — the declaration language, the derived test sizing, and exactly what each rung proves — is
documented alongside the tests in `tests/README.md`.
</content>
</invoke>
