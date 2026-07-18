"""Declaration for ``pomata.metrics.downside_deviation_rolling`` — the rolling shortfall RMS, degree-1 homogeneous."""

import math

from pomata.metrics import downside_deviation_rolling
from tests_new.metrics.downside_deviation import DOWNSIDE_DEVIATION
from tests_new.metrics.enums import BehaviorNan, BehaviorNull
from tests_new.metrics.harness import suite_metrics
from tests_new.metrics.oracles import reference_downside_deviation_rolling
from tests_new.support.declaration import Golden, Pin, ScaleAxis

DOWNSIDE_DEVIATION_ROLLING = suite_metrics(
    factory=downside_deviation_rolling,
    inputs=("returns",),
    params={"window": 3, "periods_per_year": 252, "threshold": 0.0},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    rolling_of=DOWNSIDE_DEVIATION,
    window="window",
    warmup=2,
    oracle=reference_downside_deviation_rolling,
    scaling=(ScaleAxis(roles=("returns",), degree=1),),
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"threshold": math.nan}, r"threshold must be a finite number"),
        ({"threshold": math.inf}, r"threshold must be a finite number"),
        ({"threshold": -math.inf}, r"threshold must be a finite number"),
    ),
    golden=Golden(
        inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)},
        output=(None, None, 0.1833, 0.2049, 0.0917, 0.0917, 0.1375),
    ),
    pins=(
        Pin(
            label="no_downside_window_is_zero",
            inputs={"returns": (0.01, 0.02, 0.03, 0.04)},
            expected=(None, None, 0.0, 0.0),
            reason="a window with every return at or above the threshold has zero shortfall",
        ),
        Pin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03)},
            expected=(None, None, 0.18330302779823363),
            reason="when the window equals the series length only the last row is defined",
        ),
        Pin(
            label="constant_downside_window_matches_reference",
            inputs={"returns": (-0.01, -0.01, -0.01)},
            expected=(None, None, 0.15874507866387544),
            reason="a bit-constant all-downside window — the near-constant regime; no conditioning filter is "
            "declared: the shortfall RMS sums non-negative squares with NO mean subtraction, so there "
            "is no cancellation to round apart and impl and oracle agree to the ULP",
        ),
        Pin(
            label="threshold_nonzero",
            inputs={"returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018)},
            expected=(
                None,
                None,
                0.16497272501841026,
                0.28234022030167794,
                0.23366642891095848,
                0.2509980079602227,
                0.2934280150224242,
                0.28982753492378877,
            ),
            reason="a non-zero threshold shifts the shortfall per window — the branch every other tier "
            "leaves at the 0.0 default, mirroring the reducing twin's pin",
            params_override={"threshold": 0.01},
        ),
    ),
)
