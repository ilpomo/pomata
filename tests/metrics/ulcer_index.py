"""Spec for ``pomata.metrics.ulcer_index`` — reducing, the RMS drawdown, scale-invariant."""

from tests.metrics.oracles import ulcer_index_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import ulcer_index

ULCER_INDEX = Spec(
    factory=ulcer_index,
    inputs=("equity_curve",),
    params={},
    shape=Shape.REDUCING,
    oracle=ulcer_index_reference,
    # Invariant under a positive rescale (the peak ratio cancels) — tests/metrics/test_ulcer_index.py
    # ::TestUlcerIndexProperties::test_scale_invariance.
    scale=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden_input={"equity_curve": (1.0, 1.1, 1.05, 1.2, 0.9, 1.0)},
    golden_output=(0.1241,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a one-element series has no drawdown, so the Ulcer Index is 0 "
            "(tests/metrics/test_ulcer_index.py::TestUlcerIndexEdge::test_single_row)",
        ),
        SpecPin(
            label="monotonic_rise_is_zero",
            inputs={"equity_curve": (1.0, 1.1, 1.2, 1.3)},
            expected=(0.0,),
            reason="a never-declining curve has all-zero drawdowns, so the Ulcer Index is exactly 0 "
            "(tests/metrics/test_ulcer_index.py::TestUlcerIndexEdge::test_monotonic_rise_is_zero)",
        ),
    ),
)
