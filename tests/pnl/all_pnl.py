"""
The pnl-family import side-effect aggregator: importing this imports every pnl declaration module, so each
``suite_pnl(...)`` call runs and registers its declaration in ``registry_pnl``.

``tests.all_declarations`` imports this once so the collectible rung and registry modules see a populated pnl
registry before they parametrize over it. The declarations are gathered into ``PNL_DECLARATIONS`` so the imports are
referenced (not dead), mirroring the public surface order of ``pomata.pnl.__all__``.
"""

from tests.pnl.cost_borrow import COST_BORROW
from tests.pnl.cost_fixed import COST_FIXED
from tests.pnl.cost_funding import COST_FUNDING
from tests.pnl.cost_notional import COST_NOTIONAL
from tests.pnl.cost_per_share import COST_PER_SHARE
from tests.pnl.cost_proportional import COST_PROPORTIONAL
from tests.pnl.cost_slippage import COST_SLIPPAGE
from tests.pnl.cumulative_pnl import CUMULATIVE_PNL
from tests.pnl.dividend import DIVIDEND
from tests.pnl.equity_curve import EQUITY_CURVE
from tests.pnl.pnl_gross import PNL_GROSS
from tests.pnl.pnl_gross_inverse import PNL_GROSS_INVERSE
from tests.pnl.pnl_net import PNL_NET
from tests.pnl.returns_gross import RETURNS_GROSS
from tests.pnl.returns_log import RETURNS_LOG
from tests.pnl.returns_net import RETURNS_NET
from tests.pnl.returns_simple import RETURNS_SIMPLE
from tests.pnl.turnover import TURNOVER
from tests.support.declaration import Declaration

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
