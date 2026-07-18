"""
The pnl-family import side-effect aggregator: importing this imports every pnl declaration module, so each
``suite_pnl(...)`` call runs and registers its declaration in ``registry_pnl``.

``tests_new.all_declarations`` imports this once so the collectible rung and registry modules see a populated pnl
registry before they parametrize over it. The declarations are gathered into ``PNL_DECLARATIONS`` so the imports are
referenced (not dead), mirroring the public surface order of ``pomata.pnl.__all__``.
"""

from tests_new.pnl.cost_borrow import COST_BORROW
from tests_new.pnl.cost_fixed import COST_FIXED
from tests_new.pnl.cost_funding import COST_FUNDING
from tests_new.pnl.cost_notional import COST_NOTIONAL
from tests_new.pnl.cost_per_share import COST_PER_SHARE
from tests_new.pnl.cost_proportional import COST_PROPORTIONAL
from tests_new.pnl.cost_slippage import COST_SLIPPAGE
from tests_new.pnl.cumulative_pnl import CUMULATIVE_PNL
from tests_new.pnl.dividend import DIVIDEND
from tests_new.pnl.equity_curve import EQUITY_CURVE
from tests_new.pnl.pnl_gross import PNL_GROSS
from tests_new.pnl.pnl_gross_inverse import PNL_GROSS_INVERSE
from tests_new.pnl.pnl_net import PNL_NET
from tests_new.pnl.returns_gross import RETURNS_GROSS
from tests_new.pnl.returns_log import RETURNS_LOG
from tests_new.pnl.returns_net import RETURNS_NET
from tests_new.pnl.returns_simple import RETURNS_SIMPLE
from tests_new.pnl.turnover import TURNOVER
from tests_new.support.declaration import Declaration

PNL_DECLARATIONS: tuple[Declaration, ...] = (
    COST_BORROW,
    COST_FIXED,
    COST_FUNDING,
    COST_NOTIONAL,
    COST_PER_SHARE,
    COST_PROPORTIONAL,
    COST_SLIPPAGE,
    CUMULATIVE_PNL,
    DIVIDEND,
    EQUITY_CURVE,
    PNL_GROSS,
    PNL_GROSS_INVERSE,
    PNL_NET,
    RETURNS_GROSS,
    RETURNS_LOG,
    RETURNS_NET,
    RETURNS_SIMPLE,
    TURNOVER,
)
