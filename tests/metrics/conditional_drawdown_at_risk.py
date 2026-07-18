"""Declaration for ``pomata.metrics.conditional_drawdown_at_risk`` — reducing, the mean of the worst drawdowns."""

from pomata.metrics import conditional_drawdown_at_risk
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_conditional_drawdown_at_risk
from tests.support.declaration import Golden, Pin, ScaleAxis

CONDITIONAL_DRAWDOWN_AT_RISK = suite_metrics(
    factory=conditional_drawdown_at_risk,
    inputs=("equity_curve",),
    params={"confidence": 0.95},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    oracle=reference_conditional_drawdown_at_risk,
    scaling=(ScaleAxis(roles=("equity_curve",), degree=0),),
    raises=(
        ({"confidence": 0.0}, r"confidence must be in the open interval"),
        ({"confidence": 1.0}, r"confidence must be in the open interval"),
        ({"confidence": -0.1}, r"confidence must be in the open interval"),
        ({"confidence": 1.5}, r"confidence must be in the open interval"),
    ),
    golden=Golden(inputs={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)}, output=(-0.0455,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a one-element series is at its own peak, so CDaR is exactly 0",
        ),
        Pin(
            label="no_drawdown_is_zero",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(0.0,),
            reason="a monotonically rising curve has an all-zero drawdown series, so CDaR is 0",
        ),
        Pin(
            label="fractional_weight_golden",
            inputs={"equity_curve": (1.0, 0.8, 1.0, 0.9, 0.7, 1.0)},
            expected=(-0.26666666666666666,),
            reason="the Rockafellar-Uryasev fractional boundary-weight case at confidence=0.75 (worst "
            "averaged in full, second-worst at weight 0.5)",
            params_override={"confidence": 0.75},
        ),
    ),
)
