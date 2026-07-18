"""
The collectible math module: every ``check_*`` rung parametrized over the whole registry.

Each check is one plain function of a :class:`Declaration`; here each is wrapped in a test parametrized over
``registry_all()``, so the pytest id names the function (``test_oracle_agreement[cost_borrow]``). A check a declaration
does not activate — no golden, no pins, no warm-up, a scale-exempt function — skips cleanly with a reason. With no
declarations registered yet the parametrization is empty and every rung collects green over zero cases; the family
phases fill the registry.
"""

import pytest

import tests_new.all_declarations as _registered
from tests_new.support import rungs
from tests_new.support.declaration import Declaration
from tests_new.support.registry import registry_all

# ``all_declarations`` is imported only to run its registration side effects, so the registry below is populated
# before it is read; nothing is referenced from it directly.
del _registered

_DECLARATIONS = registry_all()
_IDS = [declaration.name for declaration in _DECLARATIONS]


@pytest.mark.parametrize("declaration", _DECLARATIONS, ids=_IDS)
def test_oracle_agreement(declaration: Declaration) -> None:
    """The factory reproduces its oracle on the deterministic probe and over the fuzz domain."""
    rungs.check_oracle_agreement(declaration)


@pytest.mark.parametrize("declaration", _DECLARATIONS, ids=_IDS)
def test_golden(declaration: Declaration) -> None:
    """The frozen golden master holds."""
    rungs.check_golden(declaration)


@pytest.mark.parametrize("declaration", _DECLARATIONS, ids=_IDS)
def test_pins(declaration: Declaration) -> None:
    """Every crafted-input case maps to its pinned lanes."""
    rungs.check_pins(declaration)


@pytest.mark.parametrize("declaration", _DECLARATIONS, ids=_IDS)
def test_behavior_null(declaration: Declaration) -> None:
    """An interior null plays out exactly as the oracle plays it out."""
    rungs.check_behavior_null(declaration)


@pytest.mark.parametrize("declaration", _DECLARATIONS, ids=_IDS)
def test_behavior_nan(declaration: Declaration) -> None:
    """An interior NaN plays out exactly as the oracle plays it out."""
    rungs.check_behavior_nan(declaration)


@pytest.mark.parametrize("declaration", _DECLARATIONS, ids=_IDS)
def test_nonfinite(declaration: Declaration) -> None:
    """Each input carrying ±inf flows through exactly as the oracle carries it."""
    rungs.check_nonfinite(declaration)


@pytest.mark.parametrize("declaration", _DECLARATIONS, ids=_IDS)
def test_twin_coherence(declaration: Declaration) -> None:
    """A rolling function's row i equals its twin reduced over the trailing window (skips a non-rolling function)."""
    rungs.check_twin_coherence(declaration)


@pytest.mark.parametrize("declaration", _DECLARATIONS, ids=_IDS)
def test_annualization(declaration: Declaration) -> None:
    """A closed-form annualization scales the output by the declared period-count ratio (skips where there is none)."""
    rungs.check_annualization(declaration)


@pytest.mark.parametrize("declaration", _DECLARATIONS, ids=_IDS)
def test_scaling(declaration: Declaration) -> None:
    """Each homogeneity axis scales every lane by the declared degree."""
    rungs.check_scaling(declaration)


@pytest.mark.parametrize("declaration", _DECLARATIONS, ids=_IDS)
def test_raises(declaration: Declaration) -> None:
    """Each validation counterexample raises its canonical ValueError."""
    rungs.check_raises(declaration)


@pytest.mark.parametrize("declaration", _DECLARATIONS, ids=_IDS)
def test_warmup(declaration: Declaration) -> None:
    """The output carries exactly the declared leading nulls."""
    rungs.check_warmup(declaration)


@pytest.mark.parametrize("declaration", _DECLARATIONS, ids=_IDS)
def test_all_null(declaration: Declaration) -> None:
    """An all-null input yields all-null, or the declared deviant answer."""
    rungs.check_all_null(declaration)


@pytest.mark.parametrize("declaration", _DECLARATIONS, ids=_IDS)
def test_empty(declaration: Declaration) -> None:
    """An empty frame gives zero rows for an elementwise output, one null row for a reduction."""
    rungs.check_empty(declaration)


@pytest.mark.parametrize("declaration", _DECLARATIONS, ids=_IDS)
def test_single_row(declaration: Declaration) -> None:
    """A one-row input does not crash and keeps the declared shape."""
    rungs.check_single_row(declaration)
