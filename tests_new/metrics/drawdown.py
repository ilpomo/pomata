"""Declaration for ``pomata.metrics.drawdown`` — the running fractional decline from a prior peak, scale-invariant."""

import math

from pomata.metrics import drawdown
from tests_new.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests_new.metrics.harness import suite_metrics
from tests_new.metrics.oracles import reference_drawdown
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

DRAWDOWN = suite_metrics(
    factory=drawdown,
    inputs=("equity_curve",),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    annualization=Annualization.NONE,
    shape=Shape.SERIES,
    oracle=reference_drawdown,
    scaling=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden=Golden(
        inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2, 0.9, 1.0)}, output=(0.0, 0.0, -0.0455, 0.0, -0.25, -0.1667)
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a one-element series is at its own peak, so the drawdown is 0",
        ),
        Pin(
            label="leading_null",
            inputs={"equity_curve": (None, 1.0, 1.1, 0.99, 1.2)},
            expected=(None, 0.0, 0.0, -0.10000000000000009, 0.0),
            reason="a leading warm-up null stays null and the curve begins at the first defined equity",
        ),
        Pin(
            label="interior_null_carries_peak",
            inputs={"equity_curve": (1.0, 1.2, None, 1.1, 1.3)},
            expected=(0.0, 0.0, None, -0.08333333333333326, 0.0),
            reason="an interior null yields null at that row while the running peak carries across it",
        ),
        Pin(
            label="interior_nan_row_propagates_and_peak_carries",
            inputs={"equity_curve": (1.0, 1.1, math.nan, 0.9, 1.2)},
            expected=(0.0, 0.0, math.nan, -0.18181818181818188, 0.0),
            reason="a NaN equity yields NaN at that row while the running peak ignores it",
        ),
    ),
)
