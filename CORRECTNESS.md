# Correctness

pomata's first promise is **verifiable correctness**. The weight is on *verifiable*: not "trust us, it is right", but
"here is exactly how we know, and here is everything you need to check it yourself." This document is that explanation —
the method behind every number, and the reason each test is the size it is rather than a figure picked by feel.

It is also an honest document. We do not claim the code is free of bugs; no one can claim that. We claim something
narrower and checkable: that every indicator agrees with an independent transcription of its published formula,
satisfies a set of stated mathematical invariants, and matches the industry reference where the two are meant to match —
and that the tests proving this are sized by derivation, not by superstition.

## Why a technical indicator is hard to get right

`prices.rolling_mean(14)` looks trivial, and the happy path is. The cost — the part that takes months, and that almost
no in-house implementation pays in full — is everything around it:

- **Transcription.** The published formula has to become code with no dropped term, no swapped index, no coefficient off
  by a digit. A wrong constant is still "an indicator"; it is just the wrong one.
- **Floating-point artifacts.** Real inputs reach the corners of IEEE-754: a squared value that underflows to zero, a
  difference of large numbers that loses all its significant digits, two values so close that rounding reorders them.
  Code that is algebraically right can still disagree with itself there.
- **Boundaries.** The warmup before the first defined value; a window equal to one, equal to the series length, or
  larger than it; a single row; an empty series. Each is an off-by-one waiting to happen.
- **Missing data.** A leading `null`, an interior `null`, a `NaN` — does it propagate, latch, or get silently skipped?

Each of these is a place a correct-looking implementation is quietly wrong on the inputs you did not think to try. The
method below exists to make each one impossible to skip.

## The method

### An independent oracle

Every indicator is written twice. The shipped version is tuned to vectorize; the reference is a second, deliberately
naive derivation of the same published formula that shares no code with it. The two are derived independently from the
source mathematics, so when they agree to within a stated floating-point tolerance, that agreement is evidence: a single
bug would have to occur *identically* in two unrelated computations to hide, which is vanishingly unlikely. The oracle
exists only to be the second witness.

This holds literally for the great majority of indicators — anything expressible as composed Polars expressions, where
the naive loop and the vectorized expression graph are genuinely unrelated. It holds, too, for the exponential averages
(the unadjusted EMA, Wilder's RMA, and the ATR / RSI / MACD / multi-EMA family built on them): although the shipped code
runs as a single sequential pass, the linear recurrence has a closed form, so the oracle is computed as an independent
unrolled weighted sum rather than a transcription of the same loop — a forward-carry error cannot hide in it.

A few indicators are irreducibly sequential with no such closed form, so their oracle necessarily resembles the shipped
pass and confirms internal consistency, not independence — and we say so rather than overstate the guarantee:

- **KAMA** — the efficiency ratio and adaptive smoothing constant are derived independently, but the recurrence they
  drive is one-shape with the implementation;
- **the parabolic SAR** — a path-dependent stop-and-reverse state machine the oracle must replay branch for branch;
- **the Hilbert-transform cycle cluster** (the dominant-cycle period and phase, the phasor, the trendline, the sine
  wave, the trend flag, and MAMA) — the FIR and quadrature stages are independent, but the adaptive dominant-cycle
  period feeds back into its own measurement and the stages built on it.

Their second witness is elsewhere: the parabolic SAR is anchored to golden masters hand-computed from Wilder's published
rules — including the seeding and reversal branches the recurrence cannot reach by symmetry — while KAMA and the cycle
cluster are pinned to frozen golden masters that catch any drift and are checked against TA-Lib in the non-gating
differential tier.

### A laddered test suite

Every function declares its whole testing contract as one frozen dataclass of pure data — a spec, one file per
function — and a single ladder of generic test functions checks every declared fact identically across the whole
public surface (see `tests/README.md` for the full design). The declaration language itself enforces completeness:
required fields have no default, the spec tuples must match the public `__all__` exactly (a function cannot be
exported untested), and an input regime may be excluded from the random tier only together with a fixed case that
demonstrates what the function returns there. Four tiers, each aimed at a different threat:

- **Contract** — shape, dtype, length, laziness, and that the function partitions independently under `.over(...)`.
  The structural promises the type system cannot state.
- **Edge** — the dangerous regimes, pinned as data and checked **deterministically**: the exact warmup count (per
  struct field); an empty series; an all-`null` series; a single row; a window longer than the series; an interior
  `null`; a `NaN`. The regimes that bite only some functions — a constant window, a zero-variance benchmark, an
  exactly-flat day — are pinned where they bite, each with a written reason. These are not left to chance — and
  that fact is what lets the random tier below stay small.
- **Correctness** — agreement with the oracle on a fixed realistic series, plus a frozen golden master so a value can
  never drift unnoticed between versions.
- **Properties** — the oracle-agreement again, and the mathematical invariants (scale behavior per declared axis,
  boundedness, null handling), now over inputs drawn at random.

### A differential against the industry reference

Separately, and without gating the build, each indicator that TA-Lib also implements is compared to it — the reference
the industry has used since 2007 — which covers **58 of the 75**. With the canonical seeding most of them match TA-Lib
**bar for bar, from the first defined value**, so the comparison runs over the whole series at the same `1e-10` band as
the internal oracle. A documented minority is checked only on the converged tail — each a case where TA-Lib itself
deviates from the indicator's author over the warm-up (Wilder's first true range, the independent MACD / Chaikin EMAs)
or carries a long, implementation-specific lead-in (the Hilbert cycle pipeline, the Parabolic SAR cold start). Three
more map to a TA-Lib function but keep a deliberate convention of their own — ADXR's averaging lag, the Chande momentum
oscillator's smoothing, OBV's origin — so they do not agree with TA-Lib even at steady state and are held out of the
differential as documented divergences; each is justified against the charting authorities, not hidden. The remaining **14 have no TA-Lib twin at all** (SuperTrend,
VWAP, Ichimoku, Vortex, the Hull and Keltner / Donchian families, the volume-normalized CMF, the EWMA variance /
deviation pair, and a few more): the differential tier cannot reach them, and they rest on the independent oracle and
their golden masters instead.

## How much precision we guarantee, and where

A correctness claim needs a number. pomata's is **ten significant figures**: every indicator reproduces its independent
oracle to a relative `1e-10` on any finite input within a sane dynamic range. That single promise is the headline; the
test ladder above is how it is checked, and `tests/support/tolerances.py` is its machine-readable home. In practice the
agreement is far tighter than the promise: about half of the indicator outputs reproduce the oracle to the last bit (a
relative difference of exactly zero), and the rest land at the `float64` noise floor — typically thirteen to fifteen
figures, never fewer than the guaranteed ten.

Why `1e-10`, and why it is "safe no matter what" for this library:

- **It dwarfs the data.** Market feeds carry four to eight significant figures (a price like `123.45`, a volume); ten
  figures means the indicator never adds error you could observe — it is lossless against its own input, with two to
  six orders of headroom to spare.
- **It sits far above the float-64 noise floor.** A `float64` holds about sixteen significant figures, so legitimate
  rounding between the streaming implementation and the two-pass oracle lands around `1e-15`. `1e-10` is five orders
  above that: tight enough to reject any real coding error, loose enough that a last-bit difference never flakes.
- **It is verified, not asserted.** The property tier holds every indicator to `1e-10` over the full random fuzz domain,
  and that bound is the enforced guarantee. The realized *headroom* under it is recomputable from a clean clone with
  `scripts/calibrate_tolerances.py`, which fuzzes a representative well-conditioned set across multiple seeds and reports
  the worst relative residual — it lands around `1e-14` (a handful of `float64` noise-floor ULPs), about four orders
  inside the guarantee. The bound was sized to that measured headroom, not picked by feel.

### Where it stops: the float-conditioning limit

The one place `1e-10` cannot hold is also the one place no real market reaches. A rolling-sum statistic loses precision
only when an entire window collapses to the float-precision floor of a much larger value that recently passed through it
— summing a term of magnitude `1e6` beside residue thirteen orders smaller, an effective dynamic range past about
thirteen orders. This is `float64` obeying IEEE-754, not a defect in the library or tests: the small term is absorbed
into the mantissa of the large one, and no amount of careful coding recovers digits the format never stored. Random
inputs do not build that pattern — the property tier verifies `1e-10` across the full `[-1e6, 1e6]` fuzz domain under
stress — so it
takes a deliberately adversarial series (a price dropping seven orders of magnitude bar-to-bar) to construct it. pomata
documents this limit in the affected indicators rather than papering over it, and clamps the oscillators whose bound is
unconditional — the Chande momentum oscillator, the money flow index, Chaikin money flow — to that bound, so an
out-of-domain input degrades in precision rather than escaping the range.

Three places close this limit outright rather than documenting it. The rolling skewness and kurtosis recompute any
window at residue risk exactly (a fresh two-pass mean-centered moment wherever a value two-plus orders of magnitude
above the window's own scale has already slid out), so one bad tick can no longer poison every later window. The
reducing dispersions pin an exactly-constant series to exactly zero, so the ratios built on them (Sharpe and its
family, the information ratio's tracking error) degenerate to the documented signed infinity instead of a
residue-driven huge finite. And the rolling dispersions — the annualized rolling volatility, the rolling variance and
standard deviation, and the Bollinger bands built on them — pin a bit-constant window to exactly zero, so a departed
outlier's residue in the incremental kernel can never be reported as spread. The windowed win / loss ratios (the Chande momentum oscillator, the rolling omega) keep
the documented dynamic-range limit above instead: their sliding sums are guarded exactly at the bit-constant edge and
clamped where the output is bounded, and the residue regime needs a spread tens of orders of magnitude beyond any
market series.

A related caveat applies wherever an output cancels toward zero, not only to one family: the squaring statistics
(variance, standard deviation, Bollinger bands) as their near-constant-window output approaches zero, and equally any
mean or sum (an SMA over a window straddling large values of either sign, the price transforms) at a near-zero result.
A relative error is amplified as the denominator vanishes, so the agreement there is held to an absolute floor sized to
the input magnitude rather than the bare relative `1e-10`. The relative bound is the guarantee everywhere the output is
meaningfully non-zero — which is everywhere real market data lands.

## How big a test has to be — and why exactly that big

This is the part most suites leave to feel. We derive it. There are two separate questions with two separate answers:
how *long* the input series must be, and how *many* random inputs to draw.

### Input length: read off the indicator

An indicator produces no defined value until its **warmup** of `W` rows, and reaches steady behavior only after its
**memory** `M`. A meaningful test series is therefore

```
S = W + M + margin
```

and none of the three terms is guessed:

- `W`, the warmup, is an exact property of the indicator — the number of leading rows it returns as `null`. For an
  `n`-period RSI, `W = n`; for the dominant-cycle period, `W = 32`; for the phase-derived cycle indicators, `W = 63`.
- `M`, the memory, is the window width for a windowed indicator (`M = w`: the output is fully formed once the window is
  full). For a recursion that retains a fraction `r = 1 - alpha` of its state each step, the seed's influence decays as
  `r^t`, so it falls below a chosen `epsilon` after

  ```
  M = ceil( ln(1/epsilon) / ln(1/r) )
  ```

  steps. This term only bites when a test must outlast a transient — comparing two differently-seeded implementations,
  for instance; an oracle that shares pomata's seeding agrees from the first defined row and needs only `M = w`.
- `margin` is a handful of rows: enough to leave several defined values for an invariant to act on, and enough spread in
  length to exercise the warmup count at more than one total size.

So a short series suffices for RSI (`W = n`, same-seeded oracle, `S = n + a few`), while a cycle indicator compared on
its converged tail needs the settling term as well. The number is computed, not chosen.

### Number of random examples: derived from what the draw is *for*

Here is the figure everyone picks by feel — "let's run 500 to be safe" — and here is why that instinct is a trap.

A property test that draws `N` random inputs is **sampling, not proving**. If a bad input occupies a fraction `p` of the
input space, the chance of drawing it at least once in `N` tries is

```
1 - (1 - p)^N
```

and to hit it with confidence `1 - delta` you need

```
N  >=  ln(1/delta) / p          (for small p)
```

The floating-point artifacts that bite real indicators are *rare* — a `p` on the order of `1e-4`. Catching one reliably
(95% of runs) would take `N` in the tens of thousands. A "safety" run of a few hundred catches it about five percent of
the time — that is, by luck of the seed, not by design. A fixed mid-sized `N` is therefore the worst of both worlds: far
too small to be a reliable net for rare artifacts, far larger than needed for anything else. 500 is not safer than 470
or 240; all three are arbitrary.

The way out is not a bigger `N`. It is to drive `p` toward zero — leave as little bad input to draw as possible — and
then size `N` for what is actually left:

- **The artifacts are driven out by construction, not hunted.** Inputs are drawn from the indicator's valid domain
  (coherent OHLC bars, never impossible ones); scale tests rescale by a lossless power of two, so an ordinary rounding
  cannot flip a comparison; and the implementation guards the true singularities (a flat window, a zero denominator). A
  residual can remain — a few indicators carry an intrinsic phase-branch discontinuity whose sub-ULP cancellation a
  rescale can still flip — but it is driven so rare (order `1e-4` per draw) that `N` is sized deliberately *below* that
  flake floor, where a larger `N` would be likelier, not less likely, to trip it.
- **The dangerous regimes are covered deterministically** by the Edge tier — the empty and single-row series, the
  all-`null` series and over-long windows, the nulls and the `NaN` each have their own pinned test with a known answer
  on every indicator, and the regimes that bite only some (a constant or monotone series, the window boundaries) are
  pinned where they bite. The random draw is not what protects them.

What is left for `N` to do is narrow: cover the general interior of the input space, and catch a *systematic* mistake. A
systematic mistake — a wrong coefficient, a shifted index — is wrong on almost every input, so the chance it survives
even a handful of independent draws is negligible. Covering a few qualitative regimes of the interior to high confidence
is the same coupon-collector arithmetic, and lands in the same range: `N` in the low tens to about a hundred. Once the
property is proven and the edges are pinned, a hundred draws is already generous; a larger number buys wall-clock, not
confidence. The figure itself is a single shared number — set once in the test configuration, identical in a local run
and in CI — that an individual family raises only when its parameter space is genuinely larger.

### Tolerances: how close is close enough, by conditioning

When the implementation and the oracle agree they still round differently, and how far they drift depends on the
statistic's *conditioning*, not on a round number. Each tolerance is a named constant whose value is the worst-case
implementation-vs-oracle residual the statistic's conditioning predicts on degenerate inputs, plus a margin:

- **degree-2, two-pass** (variance): the two forms differ by about half a ULP at worst — a tight band.
- **degree-1, square-root-amplified** (standard deviation, the EWMA / MACD signal): the square root blows the relative
  error up as the variance approaches zero, worst residual about `1e-8` — a looser band (std is looser than variance,
  precisely for this reason).
- **degree-1, well-conditioned** (the recursive, windowed, and stateless means): the residual is at most a few ULP — a
  tight band.
- **scale-invariant** (a bounded ratio, a cycle period, a `0/1` flag): the output is `O(1)` at any input magnitude, so
  the band is *absolute* — sizing an `O(1)` value to the input magnitude is meaningless.

A magnitude-dependent band is sized to the data (`input_scale ** degree * factor`), not fixed, so it is right at every
scale. Where one indicator legitimately departs — a difference of large terms that cancels, a band that would underflow
at a subnormal input — the departure carries a one-line comment saying why. A bare, unexplained tolerance is treated as
a defect, not a detail.

### Benchmark size: derived from measurement stability

Timing has its own sizing. A throughput measurement is meaningful only where the per-row cost dominates the fixed
overhead of dispatching the expression — `c * S` well above a roughly constant `overhead`, i.e. `S >> overhead / c`.
Because the slow recursions have a large `c`, they reach a stable read in *fewer* rows, not more; a million rows on a
kernel that already takes over a second per evaluation buys no extra precision, only minutes. The complexity guard needs
only a single decade: a 10x increase in rows multiplies a linear cost by 10 and a quadratic one by 100, so one 10x step
separates them with a wide margin and an additive floor absorbs the overhead-bound cheap cases.

## By family

The method, the ladder, the sizing, and the tolerance rules above are family-agnostic — they apply unchanged to
indicators, PnL, and metrics. What differs per family is only the *characteristic invariants*; the exact per-primitive
figures (warmup, parameter regimes, the tolerance factor) live in the test files, declared in a uniform "Test sizing"
header, and are not duplicated here.

<details>
<summary><b>Indicators</b> — the technical-analysis layer (the used set of TA-Lib)</summary>

The characteristic invariants are scale behavior (homogeneity of degree 1 for a price-level output; invariance for a
bounded ratio, a cycle period, or a flag), boundedness where it applies (RSI in `[0, 100]`, Williams %R in `[-100, 0]`),
the exact warmup, and `null` / `NaN` propagation — each proven against the independent oracle and, in the non-gating
differential tier, against TA-Lib. Adding an indicator is a copy job: the file's "Test sizing" header states three facts
(warmup, parameter regimes, valid domain) and the rest of the property tier follows the same shape as every sibling.

</details>

<details>
<summary><b>PnL</b> — accounting and transaction costs</summary>

Shipped. The same machine applies — an independent oracle, the four-tier ladder, golden masters, and the missing-data /
large-magnitude robustness tiers. The characteristic invariants are cost monotonicity (more cost, less PnL), the
additive-vs-compounded cumulation split (`cumulative_pnl` sums currency P&L, `equity_curve` compounds returns),
no look-ahead (every bar uses only past data), and a defined, documented behavior for every degenerate input
(`null` / `NaN` / `0` / `±inf` / warm-up).

</details>

<details>
<summary><b>Metrics</b> — performance & risk statistics</summary>

Shipped. The same machine applies unchanged — an independent oracle, the four-tier ladder, derived sizing, named
tolerances. The characteristic invariants are the annualization identities, scale-equivariance (a Sharpe ratio is
invariant to leverage), closed-form checks (the Sharpe of constant returns, the drawdown of a monotone series), and a
defined, documented behavior for every degenerate input (`null` skipped, a non-null `NaN` poisoning the result, and a
degenerate denominator reported as `±inf` / `NaN`, never clipped).

</details>

## What we claim, precisely

We prove, and you can re-run:

- the output's shape, dtype, length, and laziness;
- the exact warmup, and that a `null` or `NaN` propagates as specified;
- agreement with an independent oracle to a stated floating-point tolerance, across the valid input domain;
- the documented invariants — scale behavior, bounds, monotonicity where it applies;
- parity with TA-Lib, where a counterpart exists (58 of the 75), from the first defined value — a documented minority
  only on the converged tail, every deliberate divergence documented;
- that the suite covers the whole public surface by construction (the spec/`__all__` bijection fails any collection
  that misses a function) and that every input regime excluded from the random tier is still witnessed by a fixed
  case with a written, measured reason;
- that the suite bites: it was validated by reintroducing the historically fixed defects and a vetted catalog of
  one-line semantic mutations across every source module, and it kills the full catalog — the survivors of the
  first pass were themselves turned into fixed cases.

We do **not** claim the absence of all bugs, or correctness on inputs outside the documented domain. One limit is worth
naming plainly: for the irreducibly-sequential indicators (KAMA, the parabolic SAR, the Hilbert cycle cluster) the oracle
shares its structure with the implementation by construction, and most of their golden masters are the implementation's
own frozen output — so in principle a transcription error in a seed or warm-up *value* (the counts
are pinned independently) could slip past those two tiers. Two things close that gap: the differential against TA-Lib —
an independent C implementation — now agrees **bar for bar from the first defined value** for most indicators (the
canonical seeding makes the warm-up match too, not just the tail), and a handful of golden masters are hand-computed
from the published definition. "Verifiable" is a promise about evidence, not omniscience: everything above is a test you
can read and re-run, and a number you can recompute for yourself.
