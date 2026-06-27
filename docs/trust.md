# Why you can trust pomata

Most libraries ask you to trust their output. `pomata` proves it. The premise is simple and a little obsessive:
**every function is written twice.** The shipped version is tuned to vectorize in Polars; a second, deliberately
naive *oracle* re-derives the same published formula and shares no code with it. The two must agree — on a fixed
series, on frozen golden-master numbers, and on thousands of randomly fuzzed inputs — or the build is red.

## What every function survives

The same four-tier ladder runs over all three families, under **100% branch coverage**:

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
  - Bit-level agreement with the independent oracle on a fixed, realistic series, plus frozen, hand-checked golden-master values.
* - **Properties**
  - The same oracle agreement *and* the mathematical invariants (bounds, scale behavior, monotonicity) over the full `[-1e6, 1e6]` fuzz domain, with missing data freely interleaved — driven by Hypothesis.
```

## The precision pomata guarantees

The headline is **ten significant figures**: every indicator reproduces its oracle to a relative **`1e-10`** on any
finite input within a sane dynamic range. That number is chosen, not guessed — and enforced:

- **It dwarfs the data.** Market feeds carry four to eight significant figures. Ten figures means `pomata` never adds
  error you could observe — lossless against its own input, with orders of headroom to spare.
- **It sits far above the noise floor.** A `float64` holds ~16 figures; honest rounding between the streaming
  implementation and the two-pass oracle lands around `1e-15`. `1e-10` is five orders above that — tight enough to
  reject any real coding error, loose enough that a last-bit difference never flakes the suite.
- **It is verified, not asserted.** The property tier holds every indicator to `1e-10` across the full random domain.
  In practice the agreement is *far* tighter: about **half the outputs match the oracle to the last bit** (a relative
  difference of exactly zero), and the rest land at the noise floor — typically thirteen to fifteen figures, never
  fewer than the promised ten.

:::{admonition} What pomata does *not* claim
:class: warning
Not the absence of all bugs, and not correctness outside the documented input domain. One known limit: a rolling-sum
statistic loses precision only when an entire window collapses onto the float-precision floor of a much larger value
that recently passed through it — it takes a deliberately adversarial series (a price dropping seven orders of
magnitude bar-to-bar) to build that pattern. It is documented on the affected indicators, and the oscillators whose
bound is unconditional — Chande Momentum Oscillator, Money Flow Index, Chaikin Money Flow — are clamped to that bound rather than
papered over.
:::

## The receipt: indicators vs the public reference

For indicators there is also a public yardstick — **TA-Lib**, the de-facto industry C implementation, a genuinely
different computation path. Here is `rsi(14)`, the last value of a deterministic 400-bar series, to every digit a
`float64` holds:

```text
pomata      85.20908701341023
reference   85.20908701341023   ← independent reimplementation: identical, to the last bit
TA-Lib      85.20908701341024   ← fifteen figures identical; differs only at the float64 floor
```

The same five indicators on the same series — most reproduce the reference *exactly*, the rest sit at the noise floor:

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
  - `1e-13`
* - `ema(20)`
  - `107.7299930892346`
  - `1e-13`
  - `1e-14`
* - `rsi(14)`
  - `85.20908701341023`
  - exact
  - `1e-14`
* - `atr(14)`
  - `1.904174462198776`
  - `9e-16`
  - `4e-15`
* - `macd(12,26,9)`
  - `2.523444380829531`
  - `1e-13`
  - `1e-14`
```

The differential tier (non-gating) compares most indicators against TA-Lib **bar-for-bar, from the first defined
value**. A documented minority is compared only on the converged tail — always with a reason its warm-up differs from
TA-Lib's (Wilder's first True Range, the independent MACD/Chaikin EMAs, the Ehlers Hilbert pipeline's long warm-up, the
Parabolic SAR cold start) — never a steady-state disagreement.

## PnL and metrics: proven at the edges, not in the digits

Their math is simple; their correctness lives where the bad things happen, not in the fifteenth digit. The same
machine applies — independent oracle, four-tier ladder, golden masters — but the characteristic invariants change:

- **PnL** — cost monotonicity (more cost, less PnL); the additive-vs-compounded split (`cumulative_pnl` sums currency
  P&L, `equity_curve` compounds returns); no look-ahead (every bar uses only past data); and a defined, tested behavior
  for **every** degenerate input — `null`, `NaN`, `0`, `±inf`, warm-up.
- **Metrics** — the annualization identities; scale-equivariance (a Sharpe Ratio is invariant to leverage); closed-form
  checks (the Sharpe Ratio of constant returns, the drawdown of a monotone series); and a defined behavior for every
  degenerate input — a `null` is skipped, a non-null `NaN` poisons the result, a degenerate denominator is reported as
  `±inf`/`NaN`, never silently clipped.

## The tolerance ladder

Every band is named and lives in one file (`tests/support/tolerances.py`), sized to the worst residual the statistic's
conditioning predicts — not a round number:

```{list-table}
:header-rows: 1
:widths: 32 18 50

* - Band
  - Value
  - Where it applies
* - exact
  - `1e-12`
  - recursive / stateless kernels that match the oracle to a few ULP
* - reference / property
  - `1e-10`
  - oracle agreement on fixed and fuzzed inputs — the enforced guarantee
* - scale
  - `1e-6`
  - degree-1 / degree-2 homogeneity under rescaling
* - streaming
  - `1e-3`
  - a streaming statistic evaluated at an extreme magnitude
```

## Re-run it yourself

None of this is something you have to take on faith. The full gate runs from a clean clone in one command; the
published figures regenerate from `scripts/precision_table.py`, and the realized headroom under the guarantee (it lands
around `1e-14`) from `scripts/calibrate_tolerances.py`. The complete method — the derivations, the test sizing, exactly
what is and is not claimed — is in
[`CORRECTNESS.md`](https://github.com/ilpomo/pomata/blob/main/CORRECTNESS.md).
