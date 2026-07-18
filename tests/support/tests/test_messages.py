"""
Self-tests of :mod:`tests.support.messages` — the single failure formatter.

These pin the failure contract: a derived failure must name the declaration, print the probe, show expected vs observed
with the first divergence pointed at, triage the two suspects (naming the value that would have matched), and carry a
copy-pasteable reproduction. A regression here would silently blunt every derived failure.
"""

import enum

import polars as pl

from tests.support.declaration import Declaration, ScaleAxis, Shape
from tests.support.messages import Disagreement, describe_failure, triage_for_enum, triage_generic
from tests.support.synthesis import frame_nan_interior


class _BehaviorNan(enum.Enum):
    PROPAGATES = "propagates"
    LATCHES = "latches"


class _Sole(enum.Enum):
    ONLY = "only"


class _Space(enum.Enum):
    CASH = "cash"


class _Sign(enum.Enum):
    LONG_SHORT = "long_short"


class _NonFinite(enum.Enum):
    IEEE_FLOW = "ieee_flow"


def gizmo(price: pl.Expr) -> pl.Expr:
    """A single-input stand-in factory."""
    return price


def reference_gizmo(price: list[float | None]) -> list[float | None]:
    """The stand-in oracle for ``gizmo``."""
    return list(price)


_DECL = Declaration(
    family="pnl",
    factory=gizmo,
    inputs=("price",),
    params={},
    shape=Shape.SERIES,
    behavior_null=_BehaviorNan.PROPAGATES,
    behavior_nan=_BehaviorNan.LATCHES,
    space=_Space.CASH,
    sign=_Sign.LONG_SHORT,
    nonfinite=_NonFinite.IEEE_FLOW,
    oracle=reference_gizmo,
    scaling=(ScaleAxis(roles=("price",), degree=0),),
)


class TestTriage:
    """The triage line names the declaration and suggests the value that would have matched."""

    def test_two_member_enum_suggests_the_other(self) -> None:
        """A two-value axis suggests the sibling by name."""
        line = triage_for_enum("gizmo", _BehaviorNan.LATCHES)
        assert "_BehaviorNan.LATCHES" in line
        assert "did you mean _BehaviorNan.PROPAGATES?" in line
        assert "gizmo has a bug" in line

    def test_single_member_enum_offers_no_alternative(self) -> None:
        """A one-value axis states that no other value exists."""
        line = triage_for_enum("gizmo", _Sole.ONLY)
        assert "no other value" in line

    def test_generic_triage(self) -> None:
        """The generic triage poses the two suspects for a non-enum check."""
        assert triage_generic("gizmo", "golden master") == "either the golden master is wrong or gizmo has a bug."


class TestDescribeFailure:
    """The full message carries every pillar-2 element."""

    def test_message_contains_every_element(self) -> None:
        """Header, probe, the two lanes with the first divergence, triage, and the snippet are all present."""
        probe = frame_nan_interior(_DECL)
        disagreement = Disagreement(
            lane="out",
            expected=[1.0, 2.0, float("nan"), 4.0],
            observed=[1.0, 2.0, 3.0, 4.0],
            index=2,
        )
        message = describe_failure(
            declaration=_DECL,
            check="check_behavior_nan",
            probe=probe,
            disagreement=disagreement,
            triage=triage_for_enum("gizmo", _BehaviorNan.LATCHES),
        )
        assert "gizmo (pnl): check_behavior_nan disagreed." in message
        assert "expected (oracle): [1.0, 2.0, nan, 4.0]" in message
        assert "observed (factory): [1.0, 2.0, 3.0, 4.0]" in message
        assert "first divergence at index 2: expected nan vs observed 3.0" in message
        assert "Triage:" in message
        assert "did you mean _BehaviorNan.PROPAGATES?" in message
        assert "import polars as pl" in message
        assert "from pomata.pnl import gizmo" in message
