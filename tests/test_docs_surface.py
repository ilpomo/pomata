"""
Guard the README's public-surface lists against drift.

The README advertises each family's full function list (in the collapsible "All N ..." blocks) and a headline count.
These tests assert those lists equal each family's ``__all__`` exactly, and that the headline counts match, so a rename
or addition that is not mirrored in the README fails the build instead of rotting silently -- the exact drift that once
left a stale ``momentum`` in the indicator list after it was renamed to ``mom``.
"""

import re
from pathlib import Path

import pytest

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
