"""Spec for ``pomata.metrics.sortino_ratio_rolling`` — the Sortino over a trailing window, scale-invariant."""

import math

from tests.metrics.oracles import sortino_ratio_rolling_reference
from tests.support import RELATIVE_TOLERANCE_ROLLING_ORACLE
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

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
    # A one-pass rolling mean over downside deviation against a recompute-per-window two-pass oracle.
    oracle_rel_tol=RELATIVE_TOLERANCE_ROLLING_ORACLE,
    # A ratio of a rolling mean to a rolling downside deviation is scale-invariant (by analogy to the reducing sortino).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.03, -0.01, 0.02, -0.015, 0.025, -0.005, 0.02)},
    golden_output=(None, None, 36.6606, -2.542, 18.3303, 2.8983, 73.3212),
    pins=(
        SpecPin(
            label="no_downside_is_inf",
            inputs={"returns": (0.01, 0.02, 0.03, 0.04)},
            expected=(None, None, math.inf, math.inf),
            reason="a window with no downside has zero downside deviation with a positive mean, so +inf",
        ),
        SpecPin(
            label="window_equals_length",
            inputs={"returns": (0.01, -0.02, 0.03, -0.01, 0.02)},
            expected=(None, None, None, None, 9.524704719832526),
            reason="when the window equals the series length only the last row is defined",
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
            reason="reference agreement at a non-default risk-free rate",
            params_override={"window": 4, "risk_free_rate": 0.02},
        ),
        SpecPin(
            label="tiny_downside_window_matches_reference",
            inputs={"returns": (-0.01, 0.5, 0.5)},
            expected=(None, None, 907.3499876012563),
            reason="the smallest downside deviation the fuzz domain can put in a window (one loss at the "
            "|r| >= 0.01 floor against gains at the 0.5 cap): even here the ratio matches the oracle to the ULP, "
            "so no conditioning filter is declared",
        ),
        SpecPin(
            label="flat_zero_excess_window_by_slide_is_nan",
            inputs={"returns": (-0.3233, -0.6457, 0.0, 0.4404, 0.0, 0.0, 0.0, 0.0)},
            expected=(
                None,
                None,
                None,
                -5.810192520420878,
                -2.5236460159279557,
                math.inf,
                math.inf,
                math.nan,
            ),
            reason="a window whose excess returns are all exactly zero degenerates to 0/0 -> NaN even after larger "
            "values slid out: the exact rolling mean pins the numerator to zero, so the incremental running-sum "
            "residue cannot ride above the exactly-zero downside deviation as a spurious inf; the residue is "
            "bit-sensitive, so the prefix must reach the kernel with exactly these bits — the two interior +inf "
            "rows are genuine no-downside windows with positive mean",
            params_override={"window": 4},
        ),
    ),
)
