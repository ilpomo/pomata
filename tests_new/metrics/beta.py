"""Spec for ``pomata.metrics.beta`` — reducing, the regression slope of returns on the benchmark, scale-invariant."""

import math

import polars as pl
from tests.metrics.oracles import beta_reference
from tests.support import complete_benchmark, well_spread
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import beta


def _well_spread_benchmark(frame: pl.DataFrame) -> bool:
    """Reject a near-constant benchmark: the regression slope is ill-conditioned when its denominator vanishes."""
    returns = frame["returns"].to_list()
    benchmark = frame["benchmark"].to_list()
    return well_spread(complete_benchmark(returns, benchmark))


BETA = Spec(
    factory=beta,
    inputs=("returns", "benchmark"),
    params={},
    shape=Shape.REDUCING,
    oracle=beta_reference,
    conditioning=_well_spread_benchmark,
    # A ratio of two degree-2 moments: a joint rescale of both legs by the same k leaves the slope unchanged
    # (tests/metrics/test_beta.py::TestBetaProperties::test_scale_invariance).
    scale=(ScaleAxis(roles=("returns", "benchmark"), degree=0),),
    golden_input={
        "returns": (0.012, -0.008, 0.02, -0.015, 0.005, 0.0, -0.02, 0.018),
        "benchmark": (0.01, -0.006, 0.018, -0.012, 0.004, 0.002, -0.018, 0.015),
    },
    golden_output=(1.162,),
    pins=(
        SpecPin(
            label="null_misalignment_drops_pair",
            inputs={
                "returns": (0.01, None, 0.03, -0.01, 0.02),
                "benchmark": (0.008, -0.01, None, -0.005, 0.018),
            },
            expected=(1.3157894736842104,),
            reason="a null in returns at one row and a null in benchmark at a different row each drop their pair "
            "independently (tests/metrics/test_beta.py::TestBetaEdge::test_null_misalignment_drops_pair)",
        ),
        SpecPin(
            label="nan_poisons",
            inputs={"returns": (0.01, math.nan, 0.03, -0.01), "benchmark": (0.008, -0.01, 0.025, -0.005)},
            expected=(math.nan,),
            reason="a NaN in only the returns leg still poisons the whole reduction to NaN "
            "(tests/metrics/test_beta.py::TestBetaEdge::test_nan_poisons)",
        ),
        SpecPin(
            label="single_pair",
            inputs={"returns": (0.05,), "benchmark": (0.04,)},
            expected=(None,),
            reason="one complete pair has no regression slope (needs >= 2 observations), so the result is null "
            "(tests/metrics/test_beta.py::TestBetaEdge::test_single_pair)",
        ),
        SpecPin(
            label="constant_benchmark_0_1",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (0.1, 0.1, 0.1)},
            expected=(math.nan,),
            reason="a constant (zero-variance) benchmark gives 0/0, reported as NaN via an exact max==min guard "
            "(tests/metrics/test_beta.py::TestBetaEdge::test_constant_benchmark_is_nan)",
        ),
        SpecPin(
            label="constant_benchmark_one_third",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)},
            expected=(math.nan,),
            reason="the same guard at a constant not exactly representable in float, proving it is an exact equality "
            "check, not a rounding one (tests/metrics/test_beta.py::TestBetaEdge::test_constant_benchmark_is_nan)",
        ),
        SpecPin(
            label="constant_benchmark_many_digits",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (0.123456789, 0.123456789, 0.123456789)},
            expected=(math.nan,),
            reason="the same guard at a third, many-digit constant magnitude "
            "(tests/metrics/test_beta.py::TestBetaEdge::test_constant_benchmark_is_nan)",
        ),
    ),
)
