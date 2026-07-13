"""Spec for ``pomata.metrics.sortino_ratio_rolling`` — the Sortino over a trailing window, scale-invariant."""

import math

from tests_new.metrics.oracles import sortino_ratio_rolling_reference
from tests_new.support import RELATIVE_TOLERANCE_SCALE
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import sortino_ratio_rolling

SORTINO_RATIO_ROLLING = Spec(
    factory=sortino_ratio_rolling,
    inputs=("returns",),
    params={"window": 3, "periods_per_year": 252, "risk_free_rate": 0.0},
    shape=Shape.SERIES,
    warmup=2,
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    oracle=sortino_ratio_rolling_reference,
    oracle_rel_tol=RELATIVE_TOLERANCE_SCALE,
    # A ratio of a rolling mean to a rolling downside deviation is scale-invariant (by analogy to the reducing sortino).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.03, -0.01, 0.02, -0.015, 0.025, -0.005, 0.02)},
    golden_output=(None, None, 36.6606, -2.542, 18.3303, 2.8983, 73.3212),
    pins=(
        SpecPin(
            label="no_downside_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03, 0.04)},
            expected=(None, None, math.inf, math.inf),
            reason="a window with no downside has zero downside deviation with a positive mean, so +inf "
            "(test_sortino_ratio_rolling.py::test_no_downside_is_inf)",
        ),
        SpecPin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(None, None, None, None, 9.524704719832526),
            reason="when the window equals the series length only the last row is defined "
            "(test_sortino_ratio_rolling.py::test_window_equals_length)",
            params_override={"window": 5},
        ),
        SpecPin(
            label="matches_reference_nonzero_risk_free_rate",
            inputs={"returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.01, -0.02, 0.018)},
            expected=(
                None,
                None,
                None,
                4.030099066262739,
                0.78213717158662,
                10.362383785058274,
                -6.421342897405959,
                5.014761093125949,
            ),
            reason="reference agreement at a non-default risk-free rate "
            "(test_sortino_ratio_rolling.py::test_matches_reference)",
            params_override={"window": 4, "risk_free_rate": 0.02},
        ),
        SpecPin(
            label="tiny_downside_window_matches_reference",
            inputs={"returns": (-0.01, 0.5, 0.5)},
            expected=(None, None, 907.3499876012563),
            reason="the smallest downside deviation the fuzz domain can put in a window (one loss at the "
            "|r| >= 0.01 floor against gains at the 0.5 cap) — at the fixed window of 3 this sits ABOVE the "
            "downside floor the old suite's filter cut at, so that filter was dead code and is not declared; the "
            "ratio matches the oracle to the ULP even here",
        ),
    ),
)
