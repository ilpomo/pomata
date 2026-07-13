"""Spec for ``pomata.metrics.probabilistic_sharpe_ratio`` — reducing, P(true Sharpe > benchmark), scale-invariant."""

import math

import polars as pl
from tests_new.metrics.oracles import probabilistic_sharpe_ratio_reference
from tests_new.support import well_spread
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import probabilistic_sharpe_ratio


def _well_spread(frame: pl.DataFrame) -> bool:
    """Reject a near-constant sample: the embedded sample Sharpe and higher moments are a 0/0 there."""
    return well_spread(frame.to_series(0).to_list())


PROBABILISTIC_SHARPE_RATIO = Spec(
    factory=probabilistic_sharpe_ratio,
    inputs=("returns",),
    params={"periods_per_year": 252, "benchmark_sharpe": 0.0, "risk_free_rate": 0.0},
    shape=Shape.REDUCING,
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"benchmark_sharpe": math.nan}, r"benchmark_sharpe must be a finite number"),
        ({"benchmark_sharpe": math.inf}, r"benchmark_sharpe must be a finite number"),
        ({"benchmark_sharpe": -math.inf}, r"benchmark_sharpe must be a finite number"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    oracle=probabilistic_sharpe_ratio_reference,
    conditioning=_well_spread,
    # A standardized statistic, scale-invariant, degree 0 (test_probabilistic_sharpe_ratio.py::test_scale_invariance).
    scale=(ScaleAxis(roles=("returns",), degree=0),),
    golden_input={"returns": (0.012, 0.008, 0.015, -0.004, 0.02, 0.006, 0.011, -0.003, 0.014, 0.009)},
    golden_output=(0.9922,),
    pins=(
        SpecPin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(None,),
            reason="one observation has no sample dispersion, so the statistic is null "
            "(test_probabilistic_sharpe_ratio.py::test_single_row)",
        ),
        SpecPin(
            label="zero_volatility",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            expected=(math.nan,),
            reason="a constant series has zero dispersion, so the Sharpe and higher moments are undefined, yielding "
            "NaN (test_probabilistic_sharpe_ratio.py::test_zero_volatility_is_nan)",
        ),
        SpecPin(
            label="null_skipped_benchmark_offset",
            inputs={"returns": (0.012, -0.008, 0.02, None, 0.005, 0.0, -0.02, 0.018, 0.01, -0.004)},
            expected=(0.729973707391394,),
            reason="a null is skipped under a non-default benchmark Sharpe "
            "(test_probabilistic_sharpe_ratio.py::test_null_skipped)",
            params_override={"benchmark_sharpe": 0.05},
        ),
        SpecPin(
            label="matches_reference_benchmark_offset",
            inputs={"returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018, 0.01, -0.004)},
            expected=(0.5961103866888193,),
            reason="reference agreement under a non-default benchmark Sharpe "
            "(test_probabilistic_sharpe_ratio.py::test_matches_reference)",
            params_override={"benchmark_sharpe": 0.05},
        ),
    ),
)
