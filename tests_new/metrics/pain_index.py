"""Spec for ``pomata.metrics.pain_index`` — reducing, the mean absolute drawdown, scale-invariant."""

from tests.metrics.oracles import pain_index_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import pain_index

PAIN_INDEX = Spec(
    factory=pain_index,
    inputs=("equity_curve",),
    params={},
    shape=Shape.REDUCING,
    oracle=pain_index_reference,
    # A mean of drawdowns is invariant under a positive rescale (the peak ratio cancels) — tests/metrics/
    # test_pain_index.py::TestPainIndexProperties::test_scale_invariance.
    scale=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden_input={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)},
    golden_output=(0.0179,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a one-element series is at its own peak, so the pain index is exactly 0 "
            "(tests/metrics/test_pain_index.py::TestPainIndexEdge::test_single_row)",
        ),
        SpecPin(
            label="no_drawdown_is_zero",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(0.0,),
            reason="a monotonically rising curve is never below its running peak, so the mean drawdown is 0 "
            "(tests/metrics/test_pain_index.py::TestPainIndexEdge::test_no_drawdown_is_zero)",
        ),
    ),
)
