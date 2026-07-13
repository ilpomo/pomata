"""Spec for ``pomata.metrics.max_drawdown`` — reducing, the deepest peak-to-trough decline, scale-invariant."""

from tests.metrics.oracles import max_drawdown_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import max_drawdown

MAX_DRAWDOWN = Spec(
    factory=max_drawdown,
    inputs=("equity_curve",),
    params={},
    shape=Shape.REDUCING,
    oracle=max_drawdown_reference,
    # Invariant under a positive rescale (the peak ratio cancels) — tests/metrics/test_max_drawdown.py
    # ::TestMaxDrawdownProperties::test_scale_invariance.
    scale=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden_input={"equity_curve": (1.0, 1.1, 1.05, 1.2, 0.9, 1.0)},
    golden_output=(-0.25,),
    pins=(
        SpecPin(
            label="single_row_is_zero",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a one-element series is at its own peak, so the maximum drawdown is 0 "
            "(tests/metrics/test_max_drawdown.py::TestMaxDrawdownEdge::test_single_row)",
        ),
        SpecPin(
            label="monotonic_rise_is_zero",
            inputs={"equity_curve": (1.0, 1.1, 1.2, 1.3)},
            expected=(0.0,),
            reason="a never-declining curve has zero drawdown "
            "(tests/metrics/test_max_drawdown.py::TestMaxDrawdownEdge::test_monotonic_rise_is_zero)",
        ),
    ),
)
