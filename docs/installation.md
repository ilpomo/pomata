# Installation

`pomata` has a single runtime dependency — **Polars** — and supports **Python 3.12 and newer**.

```bash
pip install pomata
# or
uv add pomata
```

That is all. Every function is a free-standing `pl.Expr` factory, so there is nothing else to wire up.

## Requirements

Python
: ≥ 3.12

Polars
: ≥ 1.42

Operating systems
: Linux · macOS · Windows

## From source

```bash
git clone https://github.com/ilpomo/pomata
cd pomata
uv sync
```

## For contributors

`pomata` uses [uv](https://docs.astral.sh/uv/) for Python, dependencies, and the virtual environment. The optional
dependency groups are `differential` (adds TA-Lib for the cross-reference parity tier) and the gate groups
(`lint` / `typecheck` / `test` / `coverage` / `docs`). The full gate — lint, four type checkers, doctests, and 100%
branch coverage — runs in one command; see {doc}`trust` and the
[contributing guide](https://github.com/ilpomo/pomata/blob/main/CONTRIBUTING.md).
