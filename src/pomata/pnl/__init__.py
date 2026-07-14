"""
Profit-and-loss accounting — composable, Polars-native, and self-sufficient: it turns the positions you hold into the
exact return and equity series the performance and risk metrics consume, so you never leave the toolkit, hand-roll P&L
(mis-handling ``null`` / ``NaN`` / ``0`` / ``inf``), or wonder whether an outside dependency's output is compatible.

One question (what is my P&L?), answered by TWO flows; pick the one that matches the data you hold:

- **Return flow** — you hold a ``weight`` (a signed fraction of capital) and the asset's fractional ``returns``
  (``0.01`` = 1%); for
  strategy research, portfolios, and cross-asset work:
  ``returns_simple`` / ``returns_log`` -> ``returns_gross`` -> (subtract composable costs) -> ``returns_net`` ->
  ``equity_curve`` (the compounded capital curve).
- **Cash / position flow** — you hold a ``quantity`` of units and a ``price`` (in currency); for instrument-level
  booking with contract multipliers, FX, and crypto:
  ``pnl_gross`` -> (subtract composable costs) -> ``pnl_net`` -> ``cumulative_pnl`` (the additive currency total).

Decision rule: think in **weights + returns** -> the ``returns_*`` flow; think in **quantities + prices** -> the
``pnl_*`` flow. The split is by unit (a fraction vs a currency amount), which is also the flow. The return flow ends in
exactly the series the metrics family consumes — ``returns_net`` / ``equity_curve`` — while the cash flow ends in the
currency series to book and inspect: ``pnl_net`` / ``cumulative_pnl``.

Every function is a free-standing ``pl.Expr`` factory: compose it in ``select`` / ``with_columns``, eager or lazy, on a
single series or a long panel via ``.over(...)``. To express a cost or convert PnL to your account currency, just
compose with arithmetic (e.g. ``pnl_gross(...) * fx_rate``). Source is organized into theme modules for
maintainability; this package re-exports a flat public API.
"""

from pomata.pnl.accounting import (
    cumulative_pnl,
    dividend,
    equity_curve,
    pnl_gross,
    pnl_gross_inverse,
    pnl_net,
    returns_gross,
    returns_net,
    turnover,
)
from pomata.pnl.costs import (
    cost_borrow,
    cost_fixed,
    cost_funding,
    cost_notional,
    cost_per_share,
    cost_proportional,
    cost_slippage,
)
from pomata.pnl.returns import returns_log, returns_simple

__all__ = (
    "cost_borrow",
    "cost_fixed",
    "cost_funding",
    "cost_notional",
    "cost_per_share",
    "cost_proportional",
    "cost_slippage",
    "cumulative_pnl",
    "dividend",
    "equity_curve",
    "pnl_gross",
    "pnl_gross_inverse",
    "pnl_net",
    "returns_gross",
    "returns_log",
    "returns_net",
    "returns_simple",
    "turnover",
)
