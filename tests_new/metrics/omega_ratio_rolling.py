"""Declaration for ``pomata.metrics.omega_ratio_rolling`` — the rolling mean gain over mean loss about a threshold."""

import math

from pomata.metrics import omega_ratio_rolling
from tests_new.metrics.enums import BehaviorNan, BehaviorNull
from tests_new.metrics.harness import suite_metrics
from tests_new.metrics.omega_ratio import OMEGA_RATIO
from tests_new.metrics.oracles import reference_omega_ratio_rolling
from tests_new.support.declaration import Golden, Pin, ScaleAxis

OMEGA_RATIO_ROLLING = suite_metrics(
    factory=omega_ratio_rolling,
    inputs=("returns",),
    params={"window": 3, "threshold": 0.0},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    rolling_of=OMEGA_RATIO,
    window="window",
    warmup=2,
    oracle=reference_omega_ratio_rolling,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"threshold": math.nan}, r"threshold must be a finite number"),
        ({"threshold": math.inf}, r"threshold must be a finite number"),
        ({"threshold": -math.inf}, r"threshold must be a finite number"),
    ),
    golden=Golden(
        inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)},
        output=(None, None, 2.0, 1.0, 5.0, 2.0, 1.3333),
    ),
    pins=(
        Pin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(None, None, None, None, 2.0),
            reason="when the window equals the series length only the last row is defined",
            params_override={"window": 5},
        ),
        Pin(
            label="matches_reference_with_threshold",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(None, None, 0.6666666666666665, 0.3999999999999999, 1.5),
            reason="agreement at a non-default threshold",
            params_override={"window": 3, "threshold": 0.01},
        ),
        Pin(
            label="no_downside_window_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(None, None, math.inf),
            reason="a window with no return below the threshold has zero mean loss, so +inf",
        ),
        Pin(
            label="no_activity_window_is_nan",
            inputs={"returns": (1000000000.0, 0.01, 1e-09, 0.0, 0.0)},
            expected=(None, math.inf, math.inf, math.inf, math.nan),
            reason="a large-magnitude corner guarding a spurious +inf residue against a correct 0/0=NaN",
            params_override={"window": 2},
        ),
        Pin(
            label="tiny_loss_window_matches_reference",
            inputs={"returns": (-0.01, 0.5, 0.5)},
            expected=(None, None, 99.99999999999999),
            reason="the smallest mean loss the fuzz domain can put in a window (one loss at the |r| >= 0.01 "
            "floor against gains at the 0.5 cap): even here the plain sum ratio matches the oracle "
            "exactly, so no conditioning filter is declared",
        ),
    ),
)
