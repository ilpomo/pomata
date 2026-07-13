"""Spec for ``pomata.metrics.drawdown_rolling`` — the decline from the trailing-window peak, scale-invariant."""

from tests.metrics.oracles import drawdown_rolling_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import drawdown_rolling

DRAWDOWN_ROLLING = Spec(
    factory=drawdown_rolling,
    inputs=("equity_curve",),
    params={"window": 3},
    shape=Shape.SERIES,
    warmup=2,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=drawdown_rolling_reference,
    # A ratio (current / trailing-window peak - 1) is invariant under a positive rescale — tests/metrics/
    # test_drawdown_rolling.py::TestDrawdownRollingProperties::test_scale_invariance.
    scale=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden_input={"equity_curve": (1.0, 1.1, 1.05, 1.2, 1.15, 1.3, 1.25)},
    golden_output=(None, None, -0.0455, 0.0, -0.0417, 0.0, -0.0385),
    pins=(
        SpecPin(
            label="window_equals_length",
            inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2)},
            expected=(None, None, None, 0.0),
            reason="when the window exactly equals the series length only the last row is defined "
            "(tests/metrics/test_drawdown_rolling.py::TestDrawdownRollingEdge::test_window_equals_length)",
            params_override={"window": 4},
        ),
        SpecPin(
            label="window_peak_is_zero",
            inputs={"equity_curve": (1.0, 1.1, 1.2, 1.3)},
            expected=(None, None, 0.0, 0.0),
            reason="at a monotonically rising window's peak the drawdown is exactly 0 "
            "(tests/metrics/test_drawdown_rolling.py::TestDrawdownRollingEdge::test_window_peak_is_zero)",
        ),
    ),
)
