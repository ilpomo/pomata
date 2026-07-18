"""Declaration for ``pomata.metrics.capture_upside_ratio`` — reducing, up-market capture, scale-exempt."""

import math

from pomata.metrics import capture_upside_ratio
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_capture_upside_ratio
from tests.support.declaration import Golden, Pin, ScaleExempt

CAPTURE_UPSIDE_RATIO = suite_metrics(
    factory=capture_upside_ratio,
    inputs=("returns", "benchmark"),
    params={"periods_per_year": 252},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.GEOMETRIC,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_capture_upside_ratio,
    scaling=ScaleExempt(reason="a ratio of two annualized geometric returns — neither scale-invariant nor homogeneous"),
    raises=(({"periods_per_year": 0}, r"periods_per_year must be >= 1"),),
    golden=Golden(
        inputs={
            "returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018),
            "benchmark": (0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015),
        },
        output=(1.3781,),
    ),
    pins=(
        Pin(
            label="null_misalignment_drops_pair",
            inputs={"returns": (0.01, None, 0.03, -0.01, 0.02), "benchmark": (0.008, -0.01, None, -0.005, 0.018)},
            expected=(1.669794280979366,),
            reason="a null in returns on one row and a null in benchmark on a different row each drop their pair",
        ),
        Pin(
            label="nan_poisons_single_leg",
            inputs={"returns": (0.02, math.nan, 0.03, -0.01), "benchmark": (0.015, 0.01, 0.025, -0.008)},
            expected=(math.nan,),
            reason="a NaN in only one leg of a retained up-market pair poisons the whole scalar to NaN",
        ),
        Pin(
            label="no_up_market_is_null",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (-0.01, -0.02, -0.03)},
            expected=(None,),
            reason="every pair is complete but no benchmark period is positive, so there is no up-market subset, null",
        ),
        Pin(
            label="return_below_negative_one_is_nan",
            inputs={"returns": (0.02, -1.5, 0.01), "benchmark": (0.01, 0.02, 0.03)},
            expected=(math.nan,),
            reason="a selected up-market return <= -1 is outside the geometric-growth domain, a loud NaN",
        ),
        Pin(
            label="flat_benchmark_day_is_excluded",
            inputs={"returns": (0.02, -0.01, 0.03, 0.015), "benchmark": (0.01, -0.02, 0.0, 0.02)},
            expected=(1.8838761627776746,),
            reason="an exactly-flat benchmark day (0.0) belongs to NEITHER market leg: the up-market subset "
            "takes strictly positive benchmark periods only — letting the flat day leak into the "
            "upside leg would distort this scalar to ~19.67; the boundary is held by this fixed case",
        ),
    ),
)
