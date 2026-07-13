"""Spec for ``pomata.metrics.payoff_ratio`` — reducing, average win over average loss magnitude, scale-invariant."""

from tests_new.metrics.oracles import payoff_ratio_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import payoff_ratio

PAYOFF_RATIO = Spec(
    factory=payoff_ratio,
    inputs=("returns",),
    params={},
    shape=Shape.REDUCING,
    oracle=payoff_ratio_reference,
    # A ratio of two means: scaling returns scales both by k, ratio invariant
    # (test_payoff_ratio.py::test_scale_invariance).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)},
    golden_output=(1.0833,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(None,),
            reason="a one-element series leaves one side empty, so the ratio is null "
            "(test_payoff_ratio.py::test_single_row)",
        ),
        SpecPin(
            label="no_losses_is_null",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(None,),
            reason="an all-positive series has no losing side, so the ratio is null "
            "(test_payoff_ratio.py::test_no_losses_is_null)",
        ),
        SpecPin(
            label="no_gains_is_null",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(None,),
            reason="an all-negative series has no winning side, so the ratio is null "
            "(test_payoff_ratio.py::test_no_gains_is_null)",
        ),
    ),
)
