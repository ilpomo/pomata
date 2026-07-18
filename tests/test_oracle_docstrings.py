"""
The oracle docstring contract, swept from the registry: every reference states how it treats missing data, and the
irreducibly-sequential mirrors say so instead of overstating independence — the taxonomy the documentation's
Correctness page describes. A reference whose recurrence has no closed form (the Ehlers cycle cluster, ``kama``,
``parabolic_sar``, and the shape-mirrors ``fisher_transform`` / ``supertrend``) can only replay the kernel it checks,
so its agreement confirms internal consistency, not independence; the reference discloses that in its own docstring.
Every other reference is an independent re-derivation and states its ``null`` (and ``NaN``) contract instead.

The word-boundary markers below are matched on the lowered docstring, so an incidental substring (``nan`` inside
``dominant``) can never satisfy a contract leg. :data:`STRUCTURAL_MIRRORS` and :func:`discloses_mirror` are the shared
source ``tests.test_docstrings`` reads for the opener-variant coupling.
"""

import inspect
import re

import pytest

import tests.all_declarations as _registered
from tests.support.declaration import Declaration
from tests.support.registry import registry_all

# ``all_declarations`` is imported only to run its registration side effects; nothing is referenced from it directly.
del _registered

# The irreducibly-sequential references: their recurrence has no closed form, so the reference necessarily replays the
# shipped kernel. The Ehlers cycle cluster and ``kama`` / ``parabolic_sar`` cannot even be seeded independently; the two
# shape-mirrors (``fisher_transform`` / ``supertrend``) are independent transcriptions that still share the kernel's
# fixed filters and branch structure. All eleven disclose their nature; every other reference re-derives independently.
STRUCTURAL_MIRRORS: frozenset[str] = frozenset(
    {
        "dominant_cycle_period",
        "dominant_cycle_phase",
        "hilbert_phasor",
        "hilbert_trendline",
        "mama",
        "sine_wave",
        "trend_mode",
        "fisher_transform",
        "kama",
        "parabolic_sar",
        "supertrend",
    }
)

# The phrasings by which a mirror reference discloses its nature; a docstring carrying none of them reads as an
# independence claim the taxonomy does not grant it.
_MIRROR_MARKERS: tuple[str, ...] = (
    "structural mirror",
    "internal consistency",
    "one-shape",
    "mirrors the production kernel",
    "the same seeded recurrence",
)

_NULL_LEG = re.compile(r"``none``|\bnone\b|\bnull\b|missing")
_NAN_LEG = re.compile(r"\bnan\b")
# A reference may instead delegate the missing-data contract to the reference it composes.
_COMPOSES = re.compile(r"_reference\b|reference_|composing|delegat")

_DECLARATIONS = registry_all()
_IDS = [declaration.name for declaration in _DECLARATIONS]


def _oracle_doc(declaration: Declaration) -> str:
    doc = inspect.getdoc(declaration.oracle)
    assert doc, f"{declaration.name}: the oracle has no docstring"
    return doc


def discloses_mirror(declaration: Declaration) -> bool:
    """Whether the declaration's reference docstring carries a structural-mirror disclosure marker."""
    return any(marker in _oracle_doc(declaration).lower() for marker in _MIRROR_MARKERS)


@pytest.mark.parametrize(
    "declaration",
    [d for d in _DECLARATIONS if d.name in STRUCTURAL_MIRRORS],
    ids=[n for n in _IDS if n in STRUCTURAL_MIRRORS],
)
def test_mirror_oracles_disclose_their_nature(declaration: Declaration) -> None:
    """Verifies each irreducibly-sequential reference states it confirms consistency, not independence."""
    assert discloses_mirror(declaration), f"{declaration.name}: no structural-mirror disclosure"


@pytest.mark.parametrize(
    "declaration",
    [d for d in _DECLARATIONS if d.name not in STRUCTURAL_MIRRORS],
    ids=[n for n in _IDS if n not in STRUCTURAL_MIRRORS],
)
def test_independent_oracles_state_their_missing_data_contract(declaration: Declaration) -> None:
    """
    Verifies each independent reference states its null leg, and its NaN leg unless it is a rolling wrapper (whose
    windows delegate the reduced core's contract) or visibly composes the reference that owns it.
    """
    doc = _oracle_doc(declaration)
    lowered = doc.lower()
    assert _NULL_LEG.search(lowered), f"{declaration.name}: the docstring states no null contract"
    if declaration.name.endswith("_rolling"):
        return
    nan_stated = _NAN_LEG.search(lowered) is not None or _COMPOSES.search(doc) is not None
    assert nan_stated, f"{declaration.name}: the docstring neither states a NaN contract nor composes a reference"
