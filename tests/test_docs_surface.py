"""
Guard the README's public-surface lists against drift.

The README advertises each family's full function list (in the collapsible "All N ..." blocks) and a headline count.
These tests assert those lists equal each family's ``__all__`` exactly, and that the headline counts match, so a rename
or addition that is not mirrored in the README fails the build instead of rotting silently -- the exact drift that once
left a stale ``momentum`` in the indicator list after it was renamed to ``mom``.
"""

import ast
import re
import tomllib
from pathlib import Path

import pytest
from tests.support.policies import NanPolicy, NullPolicy

import pomata.indicators
import pomata.metrics
import pomata.pnl

_README = Path(__file__).parent.parent / "README.md"


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


def _category_counts(header: str) -> list[tuple[str, int, int]]:
    """Per category bullet that declares a ``(N)`` count: its name, the declared count, and the number listed."""
    counts: list[tuple[str, int, int]] = []
    for line in _family_block(header).splitlines():
        match = re.match(r"- \*\*(.+?)\*\* \((\d+)\) — (.+)", line)
        if match:
            listed = re.findall(r"`[a-z_][a-z0-9_]*`", match.group(3))
            counts.append((match.group(1), int(match.group(2)), len(listed)))
    return counts


def test_readme_category_counts_match_listing() -> None:
    """Each ``**category** (N)`` parenthetical equals the number of names listed in that bullet (indicators block)."""
    counts = _category_counts("## Technical Indicators")
    assert counts, "expected per-category (N) counts in the indicators block"
    for name, declared, listed in counts:
        assert declared == listed, f"indicators category {name!r}: declared {declared}, listed {listed}"


_FAMILY_PAGES = {
    "indicators": (Path(__file__).parent.parent / "docs" / "families" / "indicators.md", pomata.indicators),
    "metrics": (Path(__file__).parent.parent / "docs" / "families" / "metrics.md", pomata.metrics),
    "pnl": (Path(__file__).parent.parent / "docs" / "families" / "pnl.md", pomata.pnl),
}


@pytest.mark.parametrize("family", sorted(_FAMILY_PAGES))
def test_family_page_catalog_matches_all(family: str) -> None:
    """
    Verifies the docs-site family page references exactly the public surface: every ``{py:func}`` role on the page
    resolves to a public name, and every public name appears at least once — the drift guard the hand-written
    catalogs previously lacked (a stale or misspelled entry passes ``sphinx -W``, whose nitpick mode is off).
    """
    page, module = _FAMILY_PAGES[family]
    text = page.read_text(encoding="utf-8")
    pattern = r"\{py:func\}`~pomata\." + re.escape(family) + r"\.(\w+)`"
    referenced = set(re.findall(pattern, text))
    public = set(module.__all__)
    assert referenced <= public, f"stale page entries: {sorted(referenced - public)}"
    assert public <= referenced, f"missing page entries: {sorted(public - referenced)}"


_DOCS = Path(__file__).parent.parent / "docs"
_PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"
# The modules whose factories run a Python kernel via ``map_batches`` — the set docs/concepts.md's performance note
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
    """The README's "every windowed form ships a rolling twin" pairing holds name-for-name."""
    names = set(pomata.metrics.__all__)
    unpaired = sorted(
        name for name in names if name.endswith("_rolling") and name.removesuffix("_rolling") not in names
    )
    assert not unpaired, f"rolling metrics without a reducing twin: {unpaired}"


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


def test_python_kernel_modules_match_the_concepts_note() -> None:
    """The ``map_batches`` kernels live in exactly the modules the concepts page's performance note enumerates."""
    found: set[str] = set()
    for path in sorted((Path(__file__).parent.parent / "src" / "pomata").rglob("*.py")):
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


def test_glossary_names_every_policy_state() -> None:
    """The glossary's null/NaN vocabulary names every declared policy state."""
    text = (_DOCS / "glossary.md").read_text(encoding="utf-8").lower()
    words = {
        NullPolicy.SKIPPED: "skipped",
        NullPolicy.ABSORBED: "absorbed",
        NullPolicy.PROPAGATES: "propagated",
        NullPolicy.IN_WINDOW_IS_NULL: "in-window-nulled",
        NullPolicy.BRIDGED: "bridged",
        NullPolicy.LATCHES: "latched",
        NanPolicy.POISONS: "poisoned",
    }
    missing = sorted(str(policy) for policy, word in words.items() if word not in text)
    assert not missing, f"glossary vocabulary misses: {missing}"
