"""
Naive reference implementations — the correctness oracles, one module per primitive.

Each function recomputes one PnL primitive from scratch in plain Python, sharing no code with the Polars implementation
it checks, so agreement between the two is evidence of correctness rather than coincidence. They target the semantics of
Polars 1.40.0 (the project's floor); each docstring states the definition, the subtle points the reimplementation
must reproduce, and its null / NaN / degeneracy contract. This package re-exports them flat, mirroring ``pomata.pnl``:
``from tests.pnl.oracles import returns_simple_reference``.
"""

from tests.pnl.oracles.cost_borrow import cost_borrow_reference
from tests.pnl.oracles.cost_fixed import cost_fixed_reference
from tests.pnl.oracles.cost_funding import cost_funding_reference
from tests.pnl.oracles.cost_notional import cost_notional_reference
from tests.pnl.oracles.cost_per_share import cost_per_share_reference
from tests.pnl.oracles.cost_proportional import cost_proportional_reference
from tests.pnl.oracles.cost_slippage import cost_slippage_reference
from tests.pnl.oracles.cumulative_pnl import cumulative_pnl_reference
from tests.pnl.oracles.dividend import dividend_reference
from tests.pnl.oracles.equity_curve import equity_curve_reference
from tests.pnl.oracles.pnl_gross import pnl_gross_reference
from tests.pnl.oracles.pnl_gross_inverse import pnl_gross_inverse_reference
from tests.pnl.oracles.pnl_net import pnl_net_reference
from tests.pnl.oracles.returns_gross import returns_gross_reference
from tests.pnl.oracles.returns_log import returns_log_reference
from tests.pnl.oracles.returns_net import returns_net_reference
from tests.pnl.oracles.returns_simple import returns_simple_reference
from tests.pnl.oracles.turnover import turnover_reference

__all__ = (
    "cost_borrow_reference",
    "cost_fixed_reference",
    "cost_funding_reference",
    "cost_notional_reference",
    "cost_per_share_reference",
    "cost_proportional_reference",
    "cost_slippage_reference",
    "cumulative_pnl_reference",
    "dividend_reference",
    "equity_curve_reference",
    "pnl_gross_inverse_reference",
    "pnl_gross_reference",
    "pnl_net_reference",
    "returns_gross_reference",
    "returns_log_reference",
    "returns_net_reference",
    "returns_simple_reference",
    "turnover_reference",
)
