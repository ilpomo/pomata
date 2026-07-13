"""Spec for ``pomata.metrics.drawdown`` — the running fractional decline from a prior peak, scale-invariant."""

import math

from tests.metrics.oracles import drawdown_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import drawdown

DRAWDOWN = Spec(
    factory=drawdown,
    inputs=("equity_curve",),
    params={},
    shape=Shape.SERIES,
    oracle=drawdown_reference,
    # Invariant under a positive rescale (the peak ratio cancels) — tests/metrics/test_drawdown.py
    # ::TestDrawdownProperties::test_scale_invariance.
    scale=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden_input={"equity_curve": (1.0, 1.1, 1.05, 1.2, 0.9, 1.0)},
    golden_output=(0.0, 0.0, -0.0455, 0.0, -0.25, -0.1667),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a one-element series is at its own peak, so the drawdown is 0 "
            "(tests/metrics/test_drawdown.py::TestDrawdownEdge::test_single_row)",
        ),
        SpecPin(
            label="leading_null",
            inputs={"equity_curve": (None, 1.0, 1.1, 0.99, 1.2)},
            expected=(None, 0.0, 0.0, 0.99 / 1.1 - 1.0, 0.0),
            reason="a leading warm-up null stays null and the curve begins at the first defined equity "
            "(tests/metrics/test_drawdown.py::TestDrawdownEdge::test_leading_null)",
        ),
        SpecPin(
            label="interior_null_carries_peak",
            inputs={"equity_curve": (1.0, 1.2, None, 1.1, 1.3)},
            expected=(0.0, 0.0, None, 1.1 / 1.2 - 1.0, 0.0),
            reason="an interior null yields null at that row while the running peak carries across it "
            "(tests/metrics/test_drawdown.py::TestDrawdownEdge::test_interior_null_carries_peak)",
        ),
        SpecPin(
            label="interior_nan_row_propagates_and_peak_carries",
            inputs={"equity_curve": (1.0, 1.1, math.nan, 0.9, 1.2)},
            expected=(0.0, 0.0, math.nan, 0.9 / 1.1 - 1.0, 0.0),
            reason="a NaN equity yields NaN at that row while the running peak ignores it "
            "(tests/metrics/test_drawdown.py::TestDrawdownEdge::test_nan_row)",
        ),
    ),
)
