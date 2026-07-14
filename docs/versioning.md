# Versioning and stability

What a version number promises, and what it does not, so you can decide how tightly to pin.

## Semantic versioning, with a pre-1.0 caveat

`pomata` follows [SemVer](https://semver.org). The caveat is the leading zero: while the version is `0.x`, the public
API is not frozen, and a **minor** bump (`0.1` to `0.2`) may carry a breaking change. That is SemVer's own rule for
`0.x`, not a local invention — it is the room a young library needs to get the surface right before committing to it.

If you depend on `pomata` today, pin accordingly: `pomata~=0.x` is too loose for production, `pomata==0.x.*` is honest
about the pre-1.0.0 reality.

## What counts as a breaking change

A change to the public surface: a renamed or removed function, a changed default, a reordered or newly keyword-only
parameter, or a different documented output for the same input. Adding a function, adding an optional keyword with a
backward-compatible default, fixing a result to match its documented formula, tightening a docstring — none of those
are breaking.

Numerical results sit slightly apart. `pomata` guarantees agreement with its oracle to ten significant figures, not
to the last bit; a release may move a value within that band (a re-vectorization, a Polars upgrade) without it being a
breaking change. A move *outside* the band is a bug, and is treated as one.

## How changes are announced

Every release has notes on the [releases page](https://github.com/ilpomo/pomata/releases), grouped by label, with
breaking changes called out first. There is no separate changelog file to fall out of sync — the release notes are
the changelog.

## Deprecation

Where a deprecation is feasible before `1.0`, the old path keeps working for one minor cycle and emits a warning that
names its replacement, then is removed in the following minor. Where a clean transition is not feasible pre-1.0, the
change ships in a minor bump and is called out at the top of the release notes. After `1.0`, removals wait for the
next major.

## Supported Python and Polars

Python **3.12 and newer**. New Python releases are adopted as the ecosystem (Polars included) supports them; the
oldest line is dropped only once it reaches end of life.

Polars **1.39 and newer**. The floor is the oldest release the full suite passes on — bisected, not guessed — and a
nightly, non-gating CI job re-proves it (a break surfaces as a maintenance signal rather than blocking a merge) by
installing the lowest allowed version and running the whole suite against it. It
moves up only when a feature or a fix genuinely requires it, never for fashion.

## At 1.0

The `1.0` release freezes the public API under the full SemVer contract: no breaking change without a major bump, and
a deprecation window for anything on its way out. The pre-1.0 work is about earning that line, not rushing to it.
