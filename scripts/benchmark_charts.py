"""
Render the benchmark pages (Markdown + inline SVG) from ``results.json`` — no plotting library, no dependency.

Every chart is a hand-built SVG inlined into the page, so it inherits the site's CSS: axes and text use
``currentColor``, the bars use Furo's own brand variable, and the ``rolling_mean`` reference line uses a themeable
custom property (both defined in ``docs/_static/custom.css``), so a chart follows the light/dark theme with zero
JavaScript and no invented colors. Every number derives from the results file; the measuring setup is stated once,
on the overview page.

Usage::

    uv run python scripts/benchmark_charts.py                # rewrite docs/benchmarks/*.md from the results
    uv run python scripts/benchmark_charts.py --results PATH --out-dir docs/benchmarks
"""

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Nav order: indicators, pnl, metrics (mirrors the sidebar).
_FAMILIES = ("indicators", "pnl", "metrics")
_FAMILY_TITLES = {"pnl": "PnL"}
_BAR = "var(--color-brand-primary)"  # the theme's blue
_ANCHOR = "var(--color-benchmark-anchor)"  # the rolling_mean reference line (a themeable red)


def _fmt_rows(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)


def _fmt_ms(seconds: float) -> str:
    """A time in milliseconds, one fixed unit across every row and every table."""
    return f"{seconds * 1e3:.2f} ms"


def _fmt_mult(x: float) -> str:
    """A speed-up multiple, compact for large magnitudes (the super-linear oracles run into the hundred-thousands)."""
    if x < 1000:
        return f"{round(x):,}×"
    if x < 1e6:
        return f"{round(x / 1e3):,}k×"
    return f"{x / 1e6:,.1f}M×"


def _api_link(family: str, name: str) -> str:
    """A MyST cross-reference to the function's API Reference entry (renders as the bare name, links to the docs)."""
    return f"{{py:func}}`~pomata.{family}.{name}`"


def _common_target(series: dict) -> int:
    """The largest row count every function reached — the one size at which all functions are compared."""
    per_function = [{int(size) for size in entry["sizes"]} for entry in series.values()]
    return max(set.intersection(*per_function))


def _oracle_target(series: dict) -> int:
    """The largest size at which EVERY function's naive oracle completed — where the ``vs oracle`` column is honest
    for all (the O(n^2)/O(n^3) oracles cannot be measured at the throughput size, so this size is smaller). A
    conservative choice: the vectorized form pulls further ahead as the data grows, so the ratio here is a floor.
    """
    return max(
        n
        for n in {int(size) for entry in series.values() for size in entry["sizes"]}
        if all(entry["sizes"].get(str(n), {}).get("oracle_s") is not None for entry in series.values())
    )


def _throughput_chart(family: str, ranked: list[tuple[str, float]], anchor: float) -> str:
    """A log-x throughput chart: one blue bar per function (name links to the API), a red dashed vertical line at the
    ``rolling_mean`` anchor so the distance from the benchmark reads at a glance across every row.
    """
    bar_h, gap, label_w, value_w, pad_l, pad_r, font = 24, 8, 320, 84, 10, 10, 16
    width = 820
    plot_x0, plot_x1 = label_w + pad_l, width - value_w - pad_r
    top = len(ranked) * (bar_h + gap)
    height = top + 40
    values = [v for _, v in ranked] + [anchor]
    lo, hi = math.log10(min(values) * 0.7), math.log10(max(values) * 1.4)

    def px(value: float) -> float:
        return plot_x0 + (math.log10(value) - lo) / (hi - lo) * (plot_x1 - plot_x0)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'style="max-width:{width}px;width:100%;height:auto;font:{font}px sans-serif;color:inherit">'
    ]
    # the rolling_mean reference line, spanning every row
    ax = px(anchor)
    parts.append(
        f'<line x1="{ax:.1f}" y1="0" x2="{ax:.1f}" y2="{top}" stroke="{_ANCHOR}" stroke-width="1.5" '
        f'stroke-dasharray="5 3"/>'
    )
    parts.append(
        f'<text x="{ax:.1f}" y="{top + 30}" text-anchor="middle" fill="{_ANCHOR}">'
        f"rolling_mean ({anchor / 1e6:,.0f} M/s)</text>"
    )
    text_y = (bar_h + font) / 2
    y = 0
    for name, value in ranked:
        bx = px(value)
        href = f"../api/{family}.html#pomata.{family}.{name}"
        parts.append(
            f'<a href="{href}"><text x="{label_w - 10}" y="{y + text_y:.0f}" text-anchor="end" '
            f'fill="var(--color-link)" text-decoration="underline">{name}</text></a>'
        )
        parts.append(
            f'<rect x="{plot_x0}" y="{y}" width="{bx - plot_x0:.1f}" height="{bar_h}" rx="2" '
            f'fill="{_BAR}" fill-opacity="0.85"/>'
        )
        parts.append(
            f'<text x="{bx + 6:.1f}" y="{y + text_y:.0f}" fill="currentColor" opacity="0.85">'
            f"{value / 1e6:,.1f} M/s</text>"
        )
        y += bar_h + gap
    parts.append("</svg>")
    return "".join(parts)


def _family_page(family: str, data: dict, out_dir: Path) -> None:
    series = {k.split(".", 1)[1]: v for k, v in data["series"].items() if k.startswith(f"{family}.")}
    target = _common_target(series)
    oracle_size = _oracle_target(series)
    anchor = statistics.median(
        target / entry["sizes"][str(target)]["anchor_s"]
        for entry in series.values()
        if entry["sizes"][str(target)]["anchor_s"]
    )
    ranked = sorted(
        ((name, target / entry["sizes"][str(target)]["pomata_s"]) for name, entry in series.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    lines = [
        f"# {_FAMILY_TITLES.get(family, family.capitalize())}",
        "",
        f"All {len(series)} `pomata.{family}` functions at **{target:,} rows**, every one on the same seeded frame. "
        f"See [Benchmarks](index.md) for the machine, the protocol, and what the `rolling_mean` anchor and the "
        f"oracle mean.",
        "",
        "## Throughput",
        "",
        "Rows processed per second (higher is better). The red line is a native Polars `rolling_mean` on the same "
        "frame — bars past it are faster than that primitive, bars short of it are slower.",
        "",
        "```{raw} html",
        _throughput_chart(family, ranked, anchor),
        "```",
        "",
        f"## Results at {target:,} rows",
        "",
        "Sorted by throughput. `vs rolling_mean` is a `pomata` call's cost relative to a native Polars "
        "`rolling_mean` on the same frame (below `1.00×` is cheaper than that primitive). `vs oracle` is how many "
        f"times faster `pomata` is than its oracle, measured at **{oracle_size:,} rows** — the largest size "
        "the slowest naive form still completes (a few oracles are super-linear); it is a conservative floor, since "
        "`pomata` pulls further ahead as the data grows.",
        "",
        *_results_table(family, ranked, series, target, oracle_size),
    ]
    (out_dir / f"{family}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _results_table(family: str, ranked: list[tuple[str, float]], series: dict, target: int, oracle_size: int) -> list[str]:
    """The per-function results as a markdown table, in throughput-descending order: throughput and cost at ``target``
    rows, the oracle speedup at ``oracle_size`` rows (where every naive form completes), rounded to a whole multiple.
    """
    rows = [
        "{.benchmark-table}",
        "| function | time | throughput | vs `rolling_mean` | vs oracle |",
        "| :-- | --: | --: | --: | --: |",
    ]
    for name, throughput in ranked:
        cell = series[name]["sizes"][str(target)]
        oracle_cell = series[name]["sizes"][str(oracle_size)]
        anchor_cost = cell["pomata_s"] / cell["anchor_s"] if cell["anchor_s"] else float("nan")
        oracle = oracle_cell["oracle_s"] / oracle_cell["pomata_s"]
        rows.append(
            f"| {_api_link(family, name)} | {_fmt_ms(cell['pomata_s'])} | {throughput / 1e6:,.1f} M/s | "
            f"{anchor_cost:.2f}× | {_fmt_mult(oracle)} |"
        )
    return rows


def _machine_label(machine: dict) -> str:
    """A human-readable machine line (the raw ``platform.platform()`` string repeats the architecture three times)."""
    raw = machine["platform"]
    if raw.startswith("macOS"):
        version = raw.split("-")[1]
        return f"macOS {version}, Apple Silicon ({machine['processor']})"
    return f"{raw} ({machine['processor']})"


def _release_label(version: str) -> str:
    """The measured release as a linked tag: a dev version names its commit (``+g<sha>``), and the tag reachable
    from that commit is the release the measured tree carries; a clean ``X.Y.Z`` is its own tag."""
    if re.fullmatch(r"\d+\.\d+\.\d+", version):
        tag = f"v{version}"
    else:
        commit = re.search(r"\+g([0-9a-f]+)", version)
        if commit is None:
            return version
        described = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0", commit.group(1)],
            capture_output=True,
            text=True,
            check=False,
        )
        if described.returncode != 0:
            return version
        tag = described.stdout.strip()
    return f"[`{tag}`](https://github.com/ilpomo/pomata/releases/tag/{tag})"


def _index_page(data: dict, out_dir: Path) -> None:
    machine = data["metadata"]["machine"]
    protocol = data["metadata"]["protocol"]
    lines = [
        "# Benchmarks",
        "",
        "How fast every public function runs, measured the same way for all three families and rendered from a single "
        "results file — the charts are inline SVG (text, not images), so the numbers are selectable and indexable. "
        "Fast is only worth trusting if it is correct: how every number is verified is the "
        "[Correctness](../correctness.md) page — the two are one argument.",
        "",
        "## Setup",
        "",
        f"- **Machine** — {_machine_label(machine)}",
        f"- **polars** — {machine['polars']}, {machine['threads']} threads",
        f"- **pomata** — {_release_label(machine['pomata'])}, Python {machine['python']}",
        f"- **Protocol** — {protocol['runs']}",
        f"- **Inputs** — seeded deterministic OHLCV random walks (seed `{protocol['seed']}`), a fresh frame per size",
        "",
        "## Scaling",
        "",
        "Every public function is **O(n)** in the number of rows: the cost grows linearly with the data. The "
        "window is free — **O(1)** in the window size — wherever Polars offers a streaming rolling primitive, which "
        "is everywhere except ten window-composed studies (`wma`, `hma`, `cci`, `aroon`, `aroon_oscillator`, and "
        "the rolling-regression family), built from per-offset shifts because no streaming primitive exists for "
        "their shape: those scale as **O(n·w)**. None of this is a promise: a nightly complexity guard measures "
        "the whole surface, holds every function to its class, and keeps the exception list exact.",
        "",
        "## Why `rolling_mean` is the anchor",
        "",
        "Absolute milliseconds depend on the machine, so every result is read against a fixed yardstick: a native "
        "Polars `rolling_mean` on the same frame. It is the right normalizer because it is a first-class Polars "
        "primitive present on every install — no extra dependency, no version skew — and it is the honest one: an "
        "external technical-analysis library would only cover part of `pomata`'s surface, so it could not anchor all "
        "153 functions on equal terms. A `vs rolling_mean` of `1.00×` means a function costs exactly what that "
        "primitive costs on the same data; below `1.00×` is cheaper.",
        "",
        "## What the oracle is",
        "",
        "Each function is also timed against its **oracle** — the plain-Python, obviously-correct "
        "reimplementation that the test suite checks every result against (see [Correctness](../correctness.md)). It "
        "is not arbitrary code: it is the same oracle that proves `pomata` correct, so `vs oracle` measures exactly "
        "how much the vectorized Polars form buys you over the straightforward loop. The `vs oracle` column's row "
        "count can differ per family: it is always the largest size at which that family's slowest naive form still "
        "completes (a few oracles are super-linear, so the honest common size is smaller where they live).",
        "",
        "```{toctree}",
        ":maxdepth: 1",
        "",
        *_FAMILIES,
        "```",
    ]
    (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="docs/_static/benchmarks/results.json")
    parser.add_argument("--out-dir", default="docs/benchmarks")
    args = parser.parse_args(argv)
    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for family in _FAMILIES:
        _family_page(family, data, out_dir)
    _index_page(data, out_dir)
    print(f"written: {out_dir}/index.md + {len(_FAMILIES)} family pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
