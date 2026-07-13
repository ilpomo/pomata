"""Spec for ``pomata.indicators.accumulation_distribution`` — the running money-flow-volume total, gap-bridging."""

from tests_new.indicators.oracles import accumulation_distribution_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import accumulation_distribution

ACCUMULATION_DISTRIBUTION = Spec(
    factory=accumulation_distribution,
    inputs=("high", "low", "close", "volume"),
    params={},
    shape=Shape.SERIES,
    warmup=None,
    lands_on="close",
    oracle=accumulation_distribution_reference,
    # The running total of the money-flow multiplier times volume: homogeneous of degree 1 in volume (the multiplier
    # is a price ratio, hence invariant to a rescaling of the price legs, degree 0) (tests/indicators/
    # test_accumulation_distribution.py::TestAccumulationDistributionProperties::test_volume_scale_homogeneity).
    scale=(
        ScaleAxis(roles=("volume",), degree=1),
        ScaleAxis(roles=("high", "low", "close"), degree=0),
    ),
    golden_input={
        "high": (10.0, 11.0, 12.0, 13.0, 14.0),
        "low": (8.0, 9.0, 10.0, 11.0, 12.0),
        "close": (9.0, 10.5, 10.0, 13.0, 12.5),
        "volume": (100.0, 200.0, 300.0, 400.0, 500.0),
    },
    golden_output=(0.0, 100.0, -200.0, 200.0, -50.0),
)
