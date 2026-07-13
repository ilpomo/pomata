"""Spec for ``pomata.metrics.downside_deviation_rolling`` — the rolling shortfall RMS, degree-1 homogeneous."""

import math

from tests.metrics.oracles import downside_deviation_rolling_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import downside_deviation_rolling

DOWNSIDE_DEVIATION_ROLLING = Spec(
    factory=downside_deviation_rolling,
    inputs=("returns",),
    params={"window": 3, "periods_per_year": 252, "threshold": 0.0},
    shape=Shape.SERIES,
    warmup=2,
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"threshold": math.nan}, r"threshold must be a finite number"),
        ({"threshold": math.inf}, r"threshold must be a finite number"),
        ({"threshold": -math.inf}, r"threshold must be a finite number"),
    ),
    oracle=downside_deviation_rolling_reference,
    # Degree-1 homogeneous per window at threshold=0 (by analogy to the reducing downside_deviation).
    scale=(ScaleAxis(roles=("returns",), degree=1),),
    golden_input={"returns": (0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015)},
    golden_output=(None, None, 0.1833, 0.2049, 0.0917, 0.0917, 0.1375),
    pins=(
        SpecPin(
            label="no_downside_window_is_zero",
            inputs={"returns": (0.01, 0.02, 0.03, 0.04)},
            expected=(None, None, 0.0, 0.0),
            reason="a window with every return at or above the threshold has zero shortfall",
        ),
        SpecPin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03)},
            expected=(None, None, 0.18330302779823363),
            reason="when the window equals the series length only the last row is defined",
        ),
        SpecPin(
            label="constant_downside_window_matches_reference",
            inputs={"returns": (-0.01, -0.01, -0.01)},
            expected=(None, None, 0.15874507866387544),
            reason="a bit-constant all-downside window — the near-constant regime; no conditioning filter is "
            "declared: the shortfall RMS sums non-negative squares with NO mean subtraction, so there is no "
            "cancellation to round apart and impl and oracle agree to the ULP",
        ),
    ),
)
