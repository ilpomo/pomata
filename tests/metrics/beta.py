"""Spec for ``pomata.metrics.beta`` — reducing, the regression slope of returns on the benchmark, scale-invariant."""

import math

from tests.metrics.oracles import beta_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.metrics import beta

BETA = Spec(
    factory=beta,
    inputs=("returns", "benchmark"),
    params={},
    shape=Shape.REDUCING,
    oracle=beta_reference,
    # A ratio of two degree-2 moments: a joint rescale of both legs by the same k leaves the slope unchanged.
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
            "independently",
        ),
        SpecPin(
            label="nan_poisons",
            inputs={"returns": (0.01, math.nan, 0.03, -0.01), "benchmark": (0.008, -0.01, 0.025, -0.005)},
            expected=(math.nan,),
            reason="a NaN in only the returns leg still poisons the whole reduction to NaN",
        ),
        SpecPin(
            label="single_pair",
            inputs={"returns": (0.05,), "benchmark": (0.04,)},
            expected=(None,),
            reason="one complete pair has no regression slope (needs >= 2 observations), so the result is null",
        ),
        SpecPin(
            label="constant_benchmark_0_1",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (0.1, 0.1, 0.1)},
            expected=(math.nan,),
            reason="a constant (zero-variance) benchmark gives 0/0, reported as NaN via an exact max==min guard — "
            "the exact-zero core of the near-constant regime; no conditioning filter is declared: the cov/var "
            "slope matches the oracle within one ULP even at ULP-adjacent benchmark spreads (measured down to a "
            "1e-15 spread on base 0.1)",
        ),
        SpecPin(
            label="constant_benchmark_one_third",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)},
            expected=(math.nan,),
            reason="the same guard at a constant not exactly representable in float, proving it is an exact equality "
            "check, not a rounding one",
        ),
        SpecPin(
            label="constant_benchmark_many_digits",
            inputs={"returns": (0.01, -0.02, 0.03), "benchmark": (0.123456789, 0.123456789, 0.123456789)},
            expected=(math.nan,),
            reason="the same guard at a third, many-digit constant magnitude",
        ),
    ),
)
