"""
Declaration for ``pomata.metrics.probabilistic_sharpe_ratio`` — reducing, P(true Sharpe > benchmark), scale-
invariant.
"""

import math

import polars as pl

from pomata.metrics import probabilistic_sharpe_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_probabilistic_sharpe_ratio
from tests.support.declaration import Golden, Pin, ScaleAxis
from tests.support.strategies import well_spread


def _well_spread(frame: pl.DataFrame) -> bool:
    """
    Reject a near-constant sample: the embedded sample Sharpe and higher moments are a 0/0 there. KEPT deliberately
    over-wide: this statistic's own divergence onset sits at var_rel ~2.5e-19, far below the shared cut of 1e-9,
    but the cut is sized on the worst family member (kurtosis) and a spec-local narrowing would buy back a
    negligible slice of draws at the price of one more magic constant — over-width here is a safe, conservative
    guard, not a hazard.
    """
    return well_spread(frame.to_series(0).to_list())


PROBABILISTIC_SHARPE_RATIO = suite_metrics(
    factory=probabilistic_sharpe_ratio,
    inputs=("returns",),
    params={"periods_per_year": 252, "benchmark_sharpe": 0.0, "risk_free_rate": 0.0},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.GEOMETRIC,
    degenerate=Degenerate.ZERO_DISPERSION_IS_NAN,
    oracle=reference_probabilistic_sharpe_ratio,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    raises=(
        ({"periods_per_year": 0}, r"periods_per_year must be >= 1"),
        ({"benchmark_sharpe": math.nan}, r"benchmark_sharpe must be a finite number"),
        ({"benchmark_sharpe": math.inf}, r"benchmark_sharpe must be a finite number"),
        ({"benchmark_sharpe": -math.inf}, r"benchmark_sharpe must be a finite number"),
        ({"risk_free_rate": math.nan}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -2.0}, r"risk_free_rate must be >= -1"),
        ({"risk_free_rate": math.inf}, r"risk_free_rate must be a finite number"),
        ({"risk_free_rate": -math.inf}, r"risk_free_rate must be a finite number"),
    ),
    conditioning=_well_spread,
    golden=Golden(
        inputs={"returns": (0.012, 0.008, 0.015, -0.004, 0.02, 0.006, 0.011, -0.003, 0.014, 0.009)}, output=(0.9922,)
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"returns": (0.05,)},
            expected=(None,),
            reason="one observation has no sample dispersion, so the statistic is null ",
        ),
        Pin(
            label="zero_volatility",
            inputs={"returns": (0.01, 0.01, 0.01, 0.01)},
            expected=(math.nan,),
            reason="a constant series has zero dispersion, so the Sharpe and higher moments are undefined, "
            "yielding NaN — the exact core of the near-constant regime the conditioning filter "
            "excludes from the property tiers",
            covers_conditioning=True,
        ),
        Pin(
            label="null_skipped_benchmark_offset",
            inputs={"returns": (0.012, -0.008, 0.02, None, 0.005, 0.0, -0.02, 0.018, 0.01, -0.004)},
            expected=(0.729973707391394,),
            reason="a null is skipped under a non-default benchmark Sharpe ",
            params_override={"benchmark_sharpe": 0.05},
        ),
        Pin(
            label="matches_reference_benchmark_offset",
            inputs={"returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018, 0.01, -0.004)},
            expected=(0.5961103866888193,),
            reason="reference agreement under a non-default benchmark Sharpe ",
            params_override={"benchmark_sharpe": 0.05},
        ),
    ),
)
