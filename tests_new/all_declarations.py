"""
The aggregator: importing this imports every family's declaration modules, so their ``suite_*`` calls run and register
each declaration. The collectible modules (``test_registry`` / ``test_rungs``) import this once to populate the registry
before they parametrize over it.

The family declaration imports land here as each family is built; the pnl and metrics families are registered. The
remaining family's aggregator joins ``ALL_DECLARATIONS`` as it is ported.
"""

from tests_new.metrics import all_metrics
from tests_new.pnl import all_pnl
from tests_new.support.declaration import Declaration

# Every registered declaration, in family then declaration order. Importing the family aggregators above is what runs
# their ``suite_*(...)`` registrations; this handle also keeps those side-effect imports referenced (pnl and metrics).
ALL_DECLARATIONS: tuple[Declaration, ...] = all_pnl.PNL_DECLARATIONS + all_metrics.METRIC_DECLARATIONS
