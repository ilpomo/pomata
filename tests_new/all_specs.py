"""
The aggregator: one explicit import per migrated function, the per-family tuples, and the surface guards.

A forgotten import is a red build. The guards run at import (so any ``pytest tests_new`` collection enforces them):
the per-family tuple must be in exact two-way correspondence with ``MIGRATED`` — a stray spec (one not listed) fails
as loudly as a gap (a listed name with no spec) — every migrated name must be in its family's public ``__all__``, and
the names must be unique. At cutover ``MIGRATED`` is replaced by the ``__all__`` tuples themselves and this becomes
the bijection guard of the whole suite.
"""

from collections import Counter

from tests_new.indicators.ichimoku import ICHIMOKU
from tests_new.indicators.mama import MAMA
from tests_new.metrics.sharpe_ratio import SHARPE_RATIO
from tests_new.pnl.equity_curve import EQUITY_CURVE
from tests_new.support.spec import Spec

import pomata.indicators
import pomata.metrics
import pomata.pnl

INDICATORS_SPECS: tuple[Spec, ...] = (ICHIMOKU, MAMA)
METRICS_SPECS: tuple[Spec, ...] = (SHARPE_RATIO,)
PNL_SPECS: tuple[Spec, ...] = (EQUITY_CURVE,)
ALL_SPECS: tuple[Spec, ...] = (*INDICATORS_SPECS, *METRICS_SPECS, *PNL_SPECS)

# The functions whose spec has landed, per family; each family extends both a tuple above and the matching set here.
MIGRATED: dict[str, frozenset[str]] = {
    "indicators": frozenset({"ichimoku", "mama"}),
    "metrics": frozenset({"sharpe_ratio"}),
    "pnl": frozenset({"equity_curve"}),
}

_FAMILY_ALL = {
    "indicators": pomata.indicators.__all__,
    "metrics": pomata.metrics.__all__,
    "pnl": pomata.pnl.__all__,
}
_FAMILY_SPECS = {"indicators": INDICATORS_SPECS, "metrics": METRICS_SPECS, "pnl": PNL_SPECS}


def _check_surface() -> None:
    """The two-way bijection, the public-name subset, and the uniqueness guard — born red, run at import."""
    duplicates = sorted(name for name, count in Counter(spec.name for spec in ALL_SPECS).items() if count > 1)
    if duplicates:
        msg = f"duplicate spec names: {duplicates}"
        raise ValueError(msg)
    for family, specs in _FAMILY_SPECS.items():
        declared = {spec.name for spec in specs}
        if declared != set(MIGRATED[family]):
            msg = f"{family}: declared specs {sorted(declared)} disagree with MIGRATED {sorted(MIGRATED[family])}"
            raise ValueError(msg)
        stray = MIGRATED[family] - set(_FAMILY_ALL[family])
        if stray:
            msg = f"{family}: migrated names outside the public __all__: {sorted(stray)}"
            raise ValueError(msg)
        misfiled = sorted(spec.name for spec in specs if spec.family != family)
        if misfiled:
            msg = f"{family}: specs whose derived family is not {family}: {misfiled}"
            raise ValueError(msg)


_check_surface()
