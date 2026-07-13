"""Spec for ``pomata.metrics.kelly_criterion`` — reducing, the optimal betting fraction, scale-invariant."""

import polars as pl
from tests.metrics.oracles import kelly_criterion_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import kelly_criterion, payoff_ratio, win_rate


def _kelly_component() -> pl.Expr:
    """Kelly recomposed from the public ``win_rate`` and ``payoff_ratio`` factories: p - (1 - p) / W."""
    probability = win_rate(pl.col("returns"))
    return probability - (1.0 - probability) / payoff_ratio(pl.col("returns"))


KELLY_CRITERION = Spec(
    factory=kelly_criterion,
    inputs=("returns",),
    params={},
    shape=Shape.REDUCING,
    oracle=kelly_criterion_reference,
    # Win rate and payoff ratio are both scale-invariant (test_kelly_criterion.py::test_scale_invariance).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)},
    golden_output=(0.1758,),
    component_expr=_kelly_component,
    pins=(
        SpecPin(
            label="single_row_one_sided",
            inputs={"returns": (0.02,)},
            expected=(None,),
            reason="a one-element series is one-sided, so the payoff ratio is undefined and the fraction is null "
            "(test_kelly_criterion.py::test_single_row)",
        ),
        SpecPin(
            label="no_losses_is_null",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(None,),
            reason="an all-positive series has an undefined payoff ratio, so the fraction is null "
            "(test_kelly_criterion.py::test_no_losses_is_null)",
        ),
        SpecPin(
            label="no_wins_is_null",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(None,),
            reason="an all-negative series has an undefined payoff ratio, so the fraction is null "
            "(test_kelly_criterion.py::test_no_wins_is_null)",
        ),
    ),
)
