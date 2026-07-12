"""Contract for ``pomata.pnl.equity_curve`` — the compounding cumulation, bridged nulls, latched NaNs."""

from collections.abc import Mapping
from typing import ClassVar

from tests.pnl.oracles import equity_curve_reference
from tests_new.support import ContractProperties, ContractSeries

from pomata.pnl import equity_curve


class TestEquityCurve(ContractSeries, ContractProperties):
    """Declarations only: every rung is inherited from the composed contracts."""

    factory = staticmethod(equity_curve)
    inputs: ClassVar[tuple[str, ...]] = ("returns",)
    params: ClassVar[Mapping[str, int | float | bool]] = {}

    oracle = staticmethod(equity_curve_reference)
    golden_input: ClassVar[Mapping[str, tuple[float | None, ...]]] = {"returns": (0.1, -0.05, 0.2, 0.1)}
    golden_output: ClassVar[tuple[float | None, ...] | Mapping[str, tuple[float | None, ...]]] = (
        1.1,
        1.045,
        1.254,
        1.3794,
    )
