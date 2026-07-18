"""Declaration for ``pomata.metrics.kelly_criterion`` — reducing, the optimal betting fraction, scale-invariant."""

import polars as pl

from pomata.metrics import kelly_criterion, payoff_ratio, win_rate
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_kelly_criterion
from tests.support.declaration import Golden, Pin, ScaleAxis


def _kelly_component() -> pl.Expr:
    """Kelly recomposed from the public ``win_rate`` and ``payoff_ratio`` factories: p - (1 - p) / W."""
    probability = win_rate(pl.col("returns"))
    return probability - (1.0 - probability) / payoff_ratio(pl.col("returns"))


KELLY_CRITERION = suite_metrics(
    factory=kelly_criterion,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_kelly_criterion,
    recomposition=_kelly_component,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    golden=Golden(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, output=(0.1758,)),
    pins=(
        Pin(
            label="single_row_one_sided",
            inputs={"returns": (0.02,)},
            expected=(None,),
            reason="a one-element series is one-sided, so the payoff ratio is undefined and the fraction is null",
        ),
        Pin(
            label="no_losses_is_null",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(None,),
            reason="an all-positive series has an undefined payoff ratio, so the fraction is null ",
        ),
        Pin(
            label="no_wins_is_null",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(None,),
            reason="an all-negative series has an undefined payoff ratio, so the fraction is null ",
        ),
    ),
)
