"""Spec for ``pomata.metrics.omega_ratio_rolling`` — the rolling mean gain over mean loss about a threshold,
scale-invariant.
"""

import math

from tests.metrics.oracles import omega_ratio_rolling_reference
from tests.support import RELATIVE_TOLERANCE_ROLLING_ORACLE
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import omega_ratio_rolling

OMEGA_RATIO_ROLLING = Spec(
    factory=omega_ratio_rolling,
    inputs=("returns",),
    params={"window": 3, "threshold": 0.0},
    shape=Shape.SERIES,
    warmup=2,
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"threshold": math.nan}, r"threshold must be a finite number"),
        ({"threshold": math.inf}, r"threshold must be a finite number"),
        ({"threshold": -math.inf}, r"threshold must be a finite number"),
    ),
    oracle=omega_ratio_rolling_reference,
    oracle_rel_tol=RELATIVE_TOLERANCE_ROLLING_ORACLE,
    # A ratio of a rolling mean gain to a rolling mean loss is scale-invariant (by analogy to the reducing omega).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)},
    golden_output=(None, None, 2.0, 1.0, 5.0, 2.0, 1.3333),
    pins=(
        SpecPin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(None, None, None, None, 2.0),
            reason="when the window equals the series length only the last row is defined",
            params_override={"window": 5},
        ),
        SpecPin(
            label="matches_reference_with_threshold",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(None, None, 0.6666666666666665, 0.3999999999999999, 1.5),
            reason="agreement at a non-default threshold",
            params_override={"window": 3, "threshold": 0.01},
        ),
        SpecPin(
            label="no_downside_window_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(None, None, math.inf),
            reason="a window with no return below the threshold has zero mean loss, so +inf",
        ),
        SpecPin(
            label="no_activity_window_is_nan",
            inputs={"returns": (1000000000.0, 0.01, 1e-09, 0.0, 0.0)},
            expected=(None, math.inf, math.inf, math.inf, math.nan),
            reason="a large-magnitude corner guarding a spurious +inf residue against a correct 0/0=NaN",
            params_override={"window": 2},
        ),
        SpecPin(
            label="tiny_loss_window_matches_reference",
            inputs={"returns": (-0.01, 0.5, 0.5)},
            expected=(None, None, 99.99999999999999),
            reason="the smallest mean loss the fuzz domain can put in a window (one loss at the |r| >= 0.01 floor "
            "against gains at the 0.5 cap): even here the plain sum ratio matches the oracle exactly, so no "
            "conditioning filter is declared",
        ),
    ),
)
