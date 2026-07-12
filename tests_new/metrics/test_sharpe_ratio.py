"""Contract for ``pomata.metrics.sharpe_ratio`` — reducing, annualized, null-skipping, NaN-poisoning."""

import math
from collections.abc import Mapping
from typing import ClassVar

from tests.metrics.oracles import sharpe_ratio_reference
from tests_new.support import ContractProperties, ContractReducing

from pomata.metrics import sharpe_ratio


class TestSharpeRatio(ContractReducing, ContractProperties):
    """Declarations only: every rung is inherited from the composed contracts."""

    factory = staticmethod(sharpe_ratio)
    inputs: ClassVar[tuple[str, ...]] = ("returns",)
    params: ClassVar[Mapping[str, int | float | bool]] = {"periods_per_year": 252, "risk_free_rate": 0.0}
    raises: ClassVar[tuple[tuple[Mapping[str, int | float | bool], str], ...]] = (
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    )

    oracle = staticmethod(sharpe_ratio_reference)
    golden_input: ClassVar[Mapping[str, tuple[float | None, ...]]] = {
        "returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02),
    }
    golden_output: ClassVar[tuple[float | None, ...] | Mapping[str, tuple[float | None, ...]]] = (2.4285,)
