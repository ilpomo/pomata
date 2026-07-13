"""Spec for ``pomata.metrics.total_return`` — reducing, the final growth factor minus one, scale-exempt."""

from tests.metrics.oracles import total_return_reference
from tests.support.spec import ScaleExempt, Shape, Spec, SpecPin

from pomata.metrics import total_return

TOTAL_RETURN = Spec(
    factory=total_return,
    inputs=("equity_curve",),
    params={},
    shape=Shape.REDUCING,
    oracle=total_return_reference,
    # A growth-factor series normalized to a unit start (the result is the final value minus one), neither
    # scale-homogeneous nor scale-invariant
    scale=ScaleExempt(
        reason="a growth-factor series normalized to a unit start (the result is the final value minus one), "
        "neither scale-homogeneous nor scale-invariant"
    ),
    golden_input={"equity_curve": (1.1, 1.045, 1.254, 1.3794)},
    golden_output=(0.3794,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"equity_curve": (1.21,)},
            expected=(0.21,),
            reason="a one-element series resolves to the final growth minus one ",
        ),
    ),
)
