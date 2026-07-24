# Benchmarks

How fast every public function runs, measured the same way for all three families and rendered from a single results file — the charts are inline SVG (text, not images), so the numbers are selectable and indexable. Fast is only worth trusting if it is correct: how every number is verified is the [Correctness](../correctness.md) page — the two are one argument.

## Setup

- **Machine** — macOS 26.5.2, Apple Silicon (arm)
- **polars** — 1.39.0, 8 threads
- **pomata** — [`v0.5.3`](https://github.com/ilpomo/pomata/releases/tag/v0.5.3), Python 3.14.2
- **Protocol** — median of up to 5 (3 past 0.5 s, 1 past 1.5 s), one warm-up, GC off
- **Inputs** — seeded deterministic OHLCV random walks (seed `20260719`), a fresh frame per size

## Scaling

Every public function is **O(n)** in the number of rows: the cost grows linearly with the data. The window is free — **O(1)** in the window size — wherever Polars offers a streaming rolling primitive, which is everywhere except ten window-composed studies (`wma`, `hma`, `cci`, `aroon`, `aroon_oscillator`, and the rolling-regression family), built from per-offset shifts because no streaming primitive exists for their shape: those scale as **O(n·w)**. None of this is a promise: a nightly complexity guard measures the whole surface, holds every function to its class, and keeps the exception list exact.

## Why `rolling_mean` is the anchor

Absolute milliseconds depend on the machine, so every result is read against a fixed yardstick: a native Polars `rolling_mean` on the same frame. It is the right normalizer because it is a first-class Polars primitive present on every install — no extra dependency, no version skew — and it is the honest one: an external technical-analysis library would only cover part of `pomata`'s surface, so it could not anchor all 153 functions on equal terms. A `vs rolling_mean` of `1.00×` means a function costs exactly what that primitive costs on the same data; below `1.00×` is cheaper.

## What the oracle is

Each function is also timed against its **oracle** — the plain-Python, obviously-correct reimplementation that the test suite checks every result against (see [Correctness](../correctness.md)). It is not arbitrary code: it is the same oracle that proves `pomata` correct, so `vs oracle` measures exactly how much the vectorized Polars form buys you over the straightforward loop. The `vs oracle` column's row count can differ per family: it is always the largest size at which that family's slowest naive form still completes (a few oracles are super-linear, so the honest common size is smaller where they live).

```{toctree}
:maxdepth: 1

indicators
pnl
metrics
```
