"""
Fit each public function's empirical complexity, cheaply, at small sizes — no heavy benchmark, safe on 8 GB.

Two axes, both from tiny seeded frames:

- **rows**: time at a geometric ladder of small row counts (default params); the log-log slope is the growth
  exponent (~1 linear, ~2 quadratic). O(n) and O(n log n) are indistinguishable by slope alone — reported as such.
- **window**: for a function with a single ``window`` parameter, time at a fixed small row count while the window
  grows; a flat curve is O(1) in the window (Polars' streaming rolling), a rising one means the window costs.

Also flags the ``map_batches`` Python kernels (a structural fact: their constant factor is huge, which the exponent
cannot see). Writes a JSON of per-function results and prints a summary.

    uv run python scripts/complexity.py --out /tmp/complexity.json
"""

import argparse
import gc
import inspect
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl  # noqa: E402

import tests.all_declarations  # noqa: F401, E402  - registration side effects
from tests.support.declaration import build_expr  # noqa: E402
from tests.support.registry import registry_all  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import _frame  # noqa: E402  - reuse the seeded coherent-OHLC frames

# A wide row span so the linear growth clears the fixed per-select overhead of the very fast functions (which a raw
# log-log slope reads as "sub-linear"); the window sweep sits at a mid size where a windowed cost would be visible.
_ROW_SIZES = (4_000, 16_000, 64_000, 256_000)
_WIN_ROWS = 32_000
_WIN_SIZES = (8, 16, 32, 64, 128, 256)


def _timed(fn) -> float:
    gc.disable()
    try:
        start = time.perf_counter_ns()
        fn()
        return (time.perf_counter_ns() - start) / 1e9
    finally:
        gc.enable()


def _measure(fn) -> float:
    fn()  # warm-up
    return statistics.median(_timed(fn) for _ in range(5))


def _r2(xs: list[float], ys: list[float], basis) -> float:
    """R^2 of the regression ``y = a + b*basis(x)`` — an intercept ``a`` absorbs the fixed per-call overhead, so the
    fit measures the *growth* term rather than the overhead-flattened raw slope.
    """
    bs = [basis(x) for x in xs]
    n = len(xs)
    mb, my = sum(bs) / n, sum(ys) / n
    sbb = sum((b - mb) ** 2 for b in bs)
    if sbb == 0:
        return 0.0
    slope = sum((b - mb) * (y - my) for b, y in zip(bs, ys, strict=True)) / sbb
    intercept = my - slope * mb
    ss_res = sum((y - (intercept + slope * b)) ** 2 for b, y in zip(bs, ys, strict=True))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return 1.0 - ss_res / ss_tot if ss_tot else 1.0


def _best_model(xs: list[float], ys: list[float], models: dict) -> tuple[str, dict[str, float]]:
    """The basis whose intercept regression fits best, with every model's R^2 for the transcript."""
    scores = {name: round(_r2(xs, ys, basis), 4) for name, basis in models.items()}
    return max(scores, key=lambda k: scores[k]), scores


_ROW_MODELS = {"O(n)": lambda x: x, "O(n log n)": lambda x: x * math.log(x), "O(n^2)": lambda x: x * x}
_WIN_MODELS = {"O(1)": lambda w: 0.0, "O(log w)": lambda w: math.log(w), "O(w)": lambda w: w}


def _window_param(declaration) -> str | None:
    """The single window parameter to sweep, or None (no window, or several — window sweep skipped)."""
    windowish = [key for key in declaration.params if "window" in key]
    if declaration.window is not None and len(windowish) == 1:
        return declaration.window
    return None


def _is_kernel(declaration) -> bool:
    """A map_batches Python kernel: the factory source calls ``map_batches`` (a huge constant the exponent misses)."""
    try:
        return "map_batches" in inspect.getsource(declaration.factory)
    except OSError:
        return False


def _fit_rows(declaration) -> dict:
    expr = build_expr(declaration)
    points: list[tuple[int, float]] = []
    for n in _ROW_SIZES:
        frame = _frame(declaration, n)
        try:
            took = _measure(lambda: frame.select(expr))  # noqa: B023
        except Exception:  # noqa: BLE001 - a function that cannot run at a size is skipped, not fatal
            continue
        if took > 0:
            points.append((n, took))
    if len(points) < 3:
        return {"class": "unmeasured", "points": points}
    xs = [float(n) for n, _ in points]
    ys = [t for _, t in points]
    # The endpoint exponent rules OUT super-linearity independently of the fixed overhead: overhead only depresses the
    # apparent exponent, so a real O(n^2) (which grows ~ data^2, dwarfing any overhead) cannot hide under a low value.
    endpoint_exp = math.log(ys[-1] / ys[0]) / math.log(xs[-1] / xs[0])
    scores = {name: round(_r2(xs, ys, basis), 4) for name, basis in _ROW_MODELS.items()}
    suspect = endpoint_exp >= 1.4 and scores["O(n^2)"] - max(scores["O(n)"], scores["O(n log n)"]) > 0.02
    cls = "O(n^2) — SUSPECT" if suspect else "O(n)"  # linear; O(n log n) is indistinguishable by fit
    return {"class": cls, "endpoint_exp": round(endpoint_exp, 3), "scores": scores, "points": points}


def _fit_window(declaration) -> dict:
    param = _window_param(declaration)
    if param is None:
        return {"class": "n/a (no single window)", "points": []}
    frame = _frame(declaration, _WIN_ROWS)
    points: list[tuple[int, float]] = []
    for w in _WIN_SIZES:
        try:
            expr = build_expr(declaration, **{param: w})
            took = _measure(lambda: frame.select(expr))  # noqa: B023
        except Exception:  # noqa: BLE001 - an invalid window (a constraint) is skipped
            continue
        if took > 0:
            points.append((w, took))
    if len(points) < 3:
        return {"class": "unmeasured", "points": points, "param": param}
    xs = [float(w) for w, _ in points]
    ys = [t for _, t in points]
    ratio = ys[-1] / ys[0]  # points are window-ascending: time at the largest window vs the smallest
    if ratio < 1.3:
        cls = "O(1)"  # the window is free — a streaming rolling primitive
    else:
        cls, _ = _best_model(xs, ys, {k: v for k, v in _WIN_MODELS.items() if k != "O(1)"})
    return {"class": cls, "ratio": round(ratio, 2), "points": points, "param": param}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/complexity.json")
    args = parser.parse_args(argv)

    declarations = sorted(registry_all(), key=lambda d: (d.family, d.name))
    results: dict[str, dict] = {}
    for index, declaration in enumerate(declarations, 1):
        key = f"{declaration.family}.{declaration.name}"
        rows = _fit_rows(declaration)
        window = _fit_window(declaration)
        kernel = _is_kernel(declaration)
        results[key] = {"rows": rows, "window": window, "kernel": kernel}
        flag = " [kernel]" if kernel else ""
        print(
            f"[{index:>3}/{len(declarations)}] {key:44} rows={rows['class']:32} "
            f"window={window['class']}{flag}",
            flush=True,
        )

    Path(args.out).write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
