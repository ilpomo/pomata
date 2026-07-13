"""Spec for ``pomata.metrics.max_drawdown_duration`` — reducing, the longest underwater run, scale-invariant."""

from tests_new.metrics.oracles import max_drawdown_duration_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import max_drawdown_duration

MAX_DRAWDOWN_DURATION = Spec(
    factory=max_drawdown_duration,
    inputs=("equity_curve",),
    params={},
    shape=Shape.REDUCING,
    oracle=max_drawdown_duration_reference,
    # A count of bars underwater is unchanged by a positive rescale (the peak ratio cancels) — tests/metrics/
    # test_max_drawdown_duration.py::TestMaxDrawdownDurationProperties::test_scale_invariance.
    scale=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden_input={"equity_curve": (1.0, 0.9, 0.8, 0.85, 1.1, 1.05)},
    golden_output=(3.0,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a one-element series is never underwater, so the duration is 0 "
            "(tests/metrics/test_max_drawdown_duration.py::test_single_row)",
        ),
        SpecPin(
            label="no_drawdown_is_zero",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(0.0,),
            reason="a monotonically rising curve is never underwater, so the duration is 0 "
            "(tests/metrics/test_max_drawdown_duration.py::test_no_drawdown_is_zero)",
        ),
    ),
)
