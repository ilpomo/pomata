"""
Self-tests of :mod:`tests.support.registry` — auto-registration and the surface bijection guard.

These pin the two ways the registry can fail closed: a duplicate registration must be refused (it would silently shadow
the first), and the ``__all__`` bijection must catch both a gap (an exported name with no declaration) and a stray (a
declaration the package does not export).
"""

import enum
import types

import polars as pl
import pytest

from tests.support.declaration import Declaration, ScaleAxis, Shape
from tests.support.registry import assert_bijection, register, registry_pnl


class _Behavior(enum.Enum):
    PROPAGATES = "propagates"


class _Space(enum.Enum):
    CASH = "cash"


class _Sign(enum.Enum):
    LONG_SHORT = "long_short"


class _NonFinite(enum.Enum):
    IEEE_FLOW = "ieee_flow"


def widget(price: pl.Expr) -> pl.Expr:
    """A stand-in factory whose ``__name__`` keys the registry."""
    return price


def reference_widget(price: list[float | None]) -> list[float | None]:
    """The stand-in oracle for ``widget``."""
    return list(price)


def _declaration(family: str = "pnl") -> Declaration:
    """A minimal valid declaration named ``widget`` in ``family``."""
    return Declaration(
        family=family,
        factory=widget,
        inputs=("price",),
        params={},
        shape=Shape.SERIES,
        behavior_null=_Behavior.PROPAGATES,
        behavior_nan=_Behavior.PROPAGATES,
        space=_Space.CASH,
        sign=_Sign.LONG_SHORT,
        nonfinite=_NonFinite.IEEE_FLOW,
        oracle=reference_widget,
        scaling=(ScaleAxis(roles=("price",), degree=0),),
    )


def _package(names: tuple[str, ...]) -> types.ModuleType:
    """A throwaway module carrying an ``__all__``, standing in for a family package."""
    module = types.ModuleType("fake_family")
    module.__dict__["__all__"] = names
    return module


class TestRegister:
    """Registration returns the declaration, stores it, and refuses duplicates and unknown families."""

    def test_returns_and_stores(self) -> None:
        """Registering returns the same declaration and stores it under its name."""
        snapshot = dict(registry_pnl)
        try:
            declaration = _declaration()
            assert register(declaration) is declaration
            assert registry_pnl["widget"] is declaration
        finally:
            registry_pnl.clear()
            registry_pnl.update(snapshot)

    def test_rejects_duplicate(self) -> None:
        """A second declaration of the same name in the same family is refused."""
        snapshot = dict(registry_pnl)
        try:
            register(_declaration())
            with pytest.raises(ValueError, match=r"duplicate declaration in pnl"):
                register(_declaration())
        finally:
            registry_pnl.clear()
            registry_pnl.update(snapshot)

    def test_rejects_unknown_family(self) -> None:
        """A declaration routed to an unknown family is refused."""
        with pytest.raises(ValueError, match=r"unknown family 'mars'"):
            register(_declaration(family="mars"))


class TestBijection:
    """The bijection guard catches gaps and strays in both directions."""

    def test_passes_on_exact_match(self) -> None:
        """A registry whose names equal ``__all__`` passes."""
        assert_bijection(_package(("widget",)), {"widget": _declaration()})

    def test_detects_gap(self) -> None:
        """An exported name with no declaration fails."""
        with pytest.raises(ValueError, match=r"missing declarations \['gadget'\]"):
            assert_bijection(_package(("widget", "gadget")), {"widget": _declaration()})

    def test_detects_stray(self) -> None:
        """A declaration the package does not export fails."""
        with pytest.raises(ValueError, match=r"stray \['gadget'\]"):
            assert_bijection(_package(("widget",)), {"widget": _declaration(), "gadget": _declaration()})
