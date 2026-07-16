"""
The oracle docstring contract, swept from the registry: every reference states how it treats missing data, and the
irreducibly-sequential mirrors say so instead of overstating independence — the taxonomy the documentation's Correctness
page describes, keyed by :data:`pomata._policy.NO_ORACLE` plus the named structural-mirror references. The word-boundary
markers
below are matched on the lowered docstring, so an incidental substring (``nan`` inside ``dominant``) can never
satisfy a contract leg.
"""

import inspect
import re

import pytest
from tests.all_specs import ALL_SPECS
from tests.support.spec import Spec, spec_id

from pomata._policy import NO_ORACLE

# The irreducibly-sequential references: NO_ORACLE carries the golden-pinned cycle cluster (their references ARE the
# pipeline mirror), and the four named ones have a reference of their own that still replays the kernel's shape.
_STRUCTURAL_MIRRORS: frozenset[str] = NO_ORACLE | frozenset({"fisher_transform", "kama", "parabolic_sar", "supertrend"})

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
_COMPOSES = re.compile(r"_reference\b|composing|delegat")


def _oracle_doc(spec: Spec) -> str:
    doc = inspect.getdoc(spec.oracle)
    assert doc, f"{spec.name}: the oracle has no docstring"
    return doc


@pytest.mark.parametrize("spec", [spec for spec in ALL_SPECS if spec.name in _STRUCTURAL_MIRRORS], ids=spec_id)
def test_mirror_oracles_disclose_their_nature(spec: Spec) -> None:
    """Verifies each irreducibly-sequential reference states it confirms consistency, not independence."""
    doc = _oracle_doc(spec).lower()
    assert any(marker in doc for marker in _MIRROR_MARKERS), f"{spec.name}: no structural-mirror disclosure"


@pytest.mark.parametrize("spec", [spec for spec in ALL_SPECS if spec.name not in _STRUCTURAL_MIRRORS], ids=spec_id)
def test_independent_oracles_state_their_missing_data_contract(spec: Spec) -> None:
    """
    Verifies each independent reference states its null leg, and its NaN leg unless it is a rolling wrapper (whose
    windows delegate the reduced core's contract) or visibly composes the reference that owns it.
    """
    doc = _oracle_doc(spec)
    lowered = doc.lower()
    assert _NULL_LEG.search(lowered), f"{spec.name}: the docstring states no null contract"
    if spec.name.endswith("_rolling"):
        return
    nan_stated = _NAN_LEG.search(lowered) is not None or _COMPOSES.search(doc) is not None
    assert nan_stated, f"{spec.name}: the docstring neither states a NaN contract nor composes a reference"
