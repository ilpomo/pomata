"""Spec for ``pomata.metrics.conditional_drawdown_at_risk`` — reducing, the mean of the worst drawdowns,
scale-invariant.
"""

from tests.metrics.oracles import conditional_drawdown_at_risk_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import conditional_drawdown_at_risk

CONDITIONAL_DRAWDOWN_AT_RISK = Spec(
    factory=conditional_drawdown_at_risk,
    inputs=("equity_curve",),
    params={"confidence": 0.95},
    shape=Shape.REDUCING,
    raises=(
        ({"confidence": 0.0}, r"confidence must be in the open interval"),
        ({"confidence": 1.0}, r"confidence must be in the open interval"),
        ({"confidence": -0.1}, r"confidence must be in the open interval"),
        ({"confidence": 1.5}, r"confidence must be in the open interval"),
    ),
    oracle=conditional_drawdown_at_risk_reference,
    # Scale-invariant: scaling every equity value by a constant leaves the drawdown series unchanged — tests/metrics/
    # test_conditional_drawdown_at_risk.py::test_scale_invariance.
    scale=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden_input={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)},
    golden_output=(-0.0455,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a one-element series is at its own peak, so CDaR is exactly 0 "
            "(tests/metrics/test_conditional_drawdown_at_risk.py::test_single_row)",
        ),
        SpecPin(
            label="no_drawdown_is_zero",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(0.0,),
            reason="a monotonically rising curve has an all-zero drawdown series, so CDaR is 0 "
            "(tests/metrics/test_conditional_drawdown_at_risk.py::test_no_drawdown_is_zero)",
        ),
        SpecPin(
            label="fractional_weight_golden",
            inputs={"equity_curve": (1.0, 0.8, 1.0, 0.9, 0.7, 1.0)},
            expected=(-0.26666666666666666,),
            reason="the Rockafellar-Uryasev fractional boundary-weight case at confidence=0.75 (worst averaged in "
            "full, second-worst at weight 0.5) (tests/metrics/test_conditional_drawdown_at_risk.py"
            "::test_fractional_weight_golden)",
            params_override={"confidence": 0.75},
        ),
    ),
)
