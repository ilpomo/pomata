"""
Guard the README and docs-site public-surface catalogs against drift.

The README advertises each family's full function list (in the collapsible "All N ..." blocks), a headline count, and a
per-category breakdown; the docs site repeats the same catalogs. These tests recount and re-catalog those figures from
the live package on every run, so a rename or addition that is not mirrored in the prose reddens the build instead of
rotting silently. The per-category bullets are held two ways: the indicators and metrics categories must list exactly
the public functions their source module defines (``category == module``), and the pnl block — which partitions by
economic units, not by module — must list exactly the cash-flow and returns-flow functions the declarations mark with
:class:`~tests.pnl.enums.SpaceCost`. The TA-Lib coverage split is recounted from each indicator's declared
:class:`~tests.indicators.enums.RelationTalib`.
"""

import ast
import re
import tomllib
from collections import defaultdict
from pathlib import Path
from types import ModuleType

import pytest

import pomata.indicators
import pomata.metrics
import pomata.pnl
import tests.all_declarations as _registered
from tests.support.registry import registry_indicators, registry_metrics, registry_pnl

# ``all_declarations`` is imported only to run its registration side effects; nothing is referenced from it directly.
del _registered

_ROOT = Path(__file__).parent.parent
_README = _ROOT / "README.md"
_DOCS = _ROOT / "docs"
_PYPROJECT = _ROOT / "pyproject.toml"

_FAMILIES: dict[str, ModuleType] = {
    "indicators": pomata.indicators,
    "metrics": pomata.metrics,
    "pnl": pomata.pnl,
}
_FAMILY_HEADERS: dict[str, str] = {
    "indicators": "## Technical Indicators",
    "metrics": "## Performance & Risk Metrics",
    "pnl": "## PnL Accounting",
}

# Every public function's defining source-module stem (e.g. ``rsi`` -> ``momentum``), so a README category can be held
# to the module it names.
_MODULE_OF: dict[str, str] = {
    name: getattr(module, name).__module__.rsplit(".", 1)[-1]
    for module in _FAMILIES.values()
    for name in module.__all__
}


def _family_block(header: str) -> str:
    """The text of the collapsible ``<details>`` block immediately following a family's ``## <section>`` header."""
    text = _README.read_text(encoding="utf-8")
    header_at = text.index(header)
    open_at = text.index("<details>", header_at)
    close_at = text.index("</details>", open_at)
    return text[open_at:close_at]


def _listed_names(header: str) -> set[str]:
    """Every backtick-quoted name in a family's ``<details>`` block (the category bullets hold only names)."""
    return set(re.findall(r"`([a-z_][a-z0-9_]*)`", _family_block(header)))


def _category_bullets(header: str) -> list[tuple[str, int | None, list[str]]]:
    """Each ``- **category** (N)? — names`` bullet: its label, the declared ``(N)`` count (or ``None``), and its
    listed names.
    """
    bullets: list[tuple[str, int | None, list[str]]] = []
    for line in _family_block(header).splitlines():
        match = re.match(r"- \*\*(.+?)\*\*(?: \((\d+)\))? — (.+)", line)
        if match:
            listed = re.findall(r"`([a-z_][a-z0-9_]*)`", match.group(3))
            bullets.append((match.group(1), int(match.group(2)) if match.group(2) else None, listed))
    return bullets


def test_readme_indicator_list_matches_all() -> None:
    """The README indicator list equals ``pomata.indicators.__all__``."""
    assert _listed_names("## Technical Indicators") == set(pomata.indicators.__all__)


def test_readme_metric_list_matches_all() -> None:
    """The README metric list equals ``pomata.metrics.__all__``."""
    assert _listed_names("## Performance & Risk Metrics") == set(pomata.metrics.__all__)


def test_readme_pnl_list_matches_all() -> None:
    """The README PnL list equals ``pomata.pnl.__all__``."""
    assert _listed_names("## PnL Accounting") == set(pomata.pnl.__all__)


def test_readme_headline_counts_match_all() -> None:
    """Each family's ``<details>`` summary count (``All N ...``) equals the family's ``__all__`` length."""
    text = _README.read_text(encoding="utf-8")
    assert f"All {len(pomata.indicators.__all__)} indicators" in text
    assert f"All {len(pomata.pnl.__all__)} PnL functions" in text
    assert f"All {len(pomata.metrics.__all__)} metrics" in text


def test_readme_category_counts_match_listing() -> None:
    """Each ``**category** (N)`` parenthetical equals the number of names listed in that bullet (indicators block)."""
    bullets = [(label, count, listed) for label, count, listed in _category_bullets("## Technical Indicators")]
    counted = [(label, count, listed) for label, count, listed in bullets if count is not None]
    assert counted, "expected per-category (N) counts in the indicators block"
    for label, count, listed in counted:
        assert count == len(listed), f"indicators category {label!r}: declared {count}, listed {len(listed)}"


@pytest.mark.parametrize("family", ["indicators", "metrics"])
def test_readme_category_bullets_match_source_modules(family: str) -> None:
    """
    Each README category lists exactly the public functions its source module defines, and the categories cover every
    module — the ``category == module`` membership the count guard alone does not prove (a function filed under the
    wrong category, or a whole module missing a bullet, passes a count check but fails this).
    """
    by_module: dict[str, set[str]] = defaultdict(set)
    for name in _FAMILIES[family].__all__:
        by_module[_MODULE_OF[name]].add(name)
    bullets = _category_bullets(_FAMILY_HEADERS[family])
    stems = {label.replace(" ", "_") for label, _count, _listed in bullets}
    assert stems == set(by_module), f"{family}: README categories {sorted(stems)} != modules {sorted(by_module)}"
    for label, _count, listed in bullets:
        stem = label.replace(" ", "_")
        assert set(listed) == by_module[stem], (
            f"{family}: category {label!r} lists {sorted(listed)} != module {stem} {sorted(by_module[stem])}"
        )


def test_readme_pnl_cost_space_split_matches_declarations() -> None:
    """
    The README's cash-flow / returns-flow bullets list exactly the functions the declarations mark ``SpaceCost.CASH``
    and ``SpaceCost.RETURNS`` — the units partition, stated as declared data (a bullet reclassifying a function, or a
    ``space`` change, now reddens the page).
    """
    bullets = {label: set(listed) for label, _count, listed in _category_bullets("## PnL Accounting")}
    cash = {name for name, d in registry_pnl.items() if d.space is not None and d.space.name == "CASH"}
    returns = {name for name, d in registry_pnl.items() if d.space is not None and d.space.name == "RETURNS"}
    assert bullets.get("cash flow") == cash, (
        f"cash flow bullet {sorted(bullets.get('cash flow', ()))} != {sorted(cash)}"
    )
    assert bullets.get("returns flow") == returns, (
        f"returns flow bullet {sorted(bullets.get('returns flow', ()))} != {sorted(returns)}"
    )


_FAMILY_PAGES = {
    "indicators": (_DOCS / "families" / "indicators.md", pomata.indicators),
    "metrics": (_DOCS / "families" / "metrics.md", pomata.metrics),
    "pnl": (_DOCS / "families" / "pnl.md", pomata.pnl),
}


@pytest.mark.parametrize("family", sorted(_FAMILY_PAGES))
def test_family_page_catalog_matches_all(family: str) -> None:
    """
    Verifies the docs-site family page references exactly the public surface: every ``{py:func}`` role on the page
    resolves to a public name, and every public name appears at least once — a drift guard ``sphinx -W`` alone does
    not provide (a stale or misspelled entry passes it, since nitpick mode is off).
    """
    page, module = _FAMILY_PAGES[family]
    text = page.read_text(encoding="utf-8")
    pattern = r"\{py:func\}`~pomata\." + re.escape(family) + r"\.(\w+)`"
    referenced = set(re.findall(pattern, text))
    public = set(module.__all__)
    assert referenced <= public, f"stale page entries: {sorted(referenced - public)}"
    assert public <= referenced, f"missing page entries: {sorted(public - referenced)}"


# The modules whose factories run a Python kernel via ``map_batches`` — the set docs/design.md's performance note
# enumerates (the Ehlers cycle family, the seeded EMA family incl. KAMA, SAR / SuperTrend, the Fisher Transform, the
# rolling moments, the PSR normal CDF). A kernel added elsewhere (or removed) must update that prose, so the set is
# pinned both ways.
_KERNEL_MODULES = frozenset({"cycle", "moving_average", "trend", "momentum", "risk", "ratio"})


def test_readme_reducer_count_matches_all() -> None:
    """The README's "N reduce the whole history" figure equals the actual reducer count (non-rolling, non-drawdown)."""
    names = set(pomata.metrics.__all__)
    reducers = len(names) - sum(1 for name in names if name.endswith("_rolling")) - 1  # drawdown is row-wise
    assert f"{reducers} reduce the whole history" in _README.read_text(encoding="utf-8")


def test_every_rolling_metric_has_its_reducing_twin() -> None:
    """The README's "every windowed form ships a rolling twin" pairing holds, and each declared ``rolling_of`` twin
    names exactly the reducing function it rolls.
    """
    names = set(pomata.metrics.__all__)
    unpaired = sorted(
        name for name in names if name.endswith("_rolling") and name.removesuffix("_rolling") not in names
    )
    assert not unpaired, f"rolling metrics without a reducing twin: {unpaired}"
    mismatched = sorted(
        declaration.name
        for declaration in registry_metrics.values()
        if declaration.rolling_of is not None
        and declaration.rolling_of.name != declaration.name.removesuffix("_rolling")
    )
    assert not mismatched, f"declared rolling_of twin does not name its reducing function: {mismatched}"


def test_index_family_table_matches_all() -> None:
    """The docs landing page's family table carries the real ``__all__`` count for each family."""
    text = (_DOCS / "index.md").read_text(encoding="utf-8")
    for family, module in (
        ("indicators", pomata.indicators),
        ("pnl", pomata.pnl),
        ("metrics", pomata.metrics),
    ):
        match = re.search(r"\| \*\*`pomata\." + family + r"`\*\* \| (\d+) \|", text)
        assert match is not None, f"index.md: no table row for pomata.{family}"
        assert int(match.group(1)) == len(module.__all__), f"index.md: pomata.{family} row says {match.group(1)}"


def test_python_kernel_modules_match_the_design_note() -> None:
    """The ``map_batches`` kernels live in exactly the modules the design page's performance note enumerates."""
    found: set[str] = set()
    for path in sorted((_ROOT / "src" / "pomata").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "map_batches":
                found.add(path.stem)
    assert found == _KERNEL_MODULES, f"map_batches modules {sorted(found)} != pinned {sorted(_KERNEL_MODULES)}"


def test_single_runtime_dependency_claim_holds() -> None:
    """The "Polars is the only runtime dependency" claim matches ``pyproject.toml`` exactly."""
    with _PYPROJECT.open("rb") as handle:
        dependencies = tomllib.load(handle)["project"]["dependencies"]
    assert len(dependencies) == 1, dependencies
    assert str(dependencies[0]).startswith("polars"), dependencies


def test_readme_talib_split_matches_the_partition() -> None:
    """
    Verifies the README's TA-Lib coverage split equals the declared ``RelationTalib`` partition, so reclassifying an
    indicator between the compared and excluded buckets reddens the page instead of silently staling its figures.
    """
    readme = _README.read_text(encoding="utf-8")
    total = len(pomata.indicators.__all__)
    excluded = sum(
        1
        for declaration in registry_indicators.values()
        if declaration.talib is not None and declaration.talib.name in ("NO_EQUIVALENT", "DOCUMENTED_DIVERGENCE")
    )
    compared = total - excluded
    assert f"and {compared} of the {total} cross-checked against" in readme
    assert f"(the other {excluded} have no TA-Lib twin" in readme
    assert f"for the {compared} of {total} with a twin" in readme
