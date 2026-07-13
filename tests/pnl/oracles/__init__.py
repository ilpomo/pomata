"""
Re-export shim: the P&L oracles now live in :mod:`tests_new.pnl.oracles`.

The naive reference implementations were consolidated into the ``tests_new`` home during the contract-suite
redesign; this thin package keeps the legacy ``from tests.pnl.oracles import <name>_reference`` path
working for the existing suite until its cutover. See ``tests_new/DESIGN.md``.
"""

# The oracle *modules* live in the new home; extend this shim package's search path so the legacy per-module
# path ``tests.pnl.oracles.<name>`` (imported dynamically by ``tests.test_policies``) resolves there too.
from tests_new.pnl.oracles import __path__ as _new_oracles_path
from tests_new.pnl.oracles import (
    cost_borrow_reference,
    cost_fixed_reference,
    cost_funding_reference,
    cost_notional_reference,
    cost_per_share_reference,
    cost_proportional_reference,
    cost_slippage_reference,
    cumulative_pnl_reference,
    dividend_reference,
    equity_curve_reference,
    pnl_gross_inverse_reference,
    pnl_gross_reference,
    pnl_net_reference,
    returns_gross_reference,
    returns_log_reference,
    returns_net_reference,
    returns_simple_reference,
    turnover_reference,
)

__path__ = [*__path__, *_new_oracles_path]

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
