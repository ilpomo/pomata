"""Spec for ``pomata.indicators.obv`` — On-Balance Volume, the signed-volume running total, gap-bridging."""

from tests.indicators.oracles import obv_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import obv

OBV = Spec(
    factory=obv,
    inputs=("price", "volume"),
    params={},
    shape=Shape.SERIES,
    warmup=None,
    oracle=obv_reference,
    # Each bar adds or subtracts its whole volume by the sign of the price change: homogeneous of degree 1 in volume,
    # and invariant to a positive rescaling of the price (the sign is preserved), degree 0 (tests/indicators/
    # test_obv.py::TestObvProperties::test_volume_scale_homogeneity).
    scale=(
        ScaleAxis(roles=("volume",), degree=1),
        ScaleAxis(roles=("price",), degree=0),
    ),
    golden_input={
        "price": (10.0, 12.0, 11.0, 11.0, 13.0, 9.0, 9.0, 14.0),
        "volume": (100.0, 200.0, 150.0, 80.0, 300.0, 250.0, 90.0, 400.0),
    },
    golden_output=(0.0, 200.0, 50.0, 50.0, 350.0, 100.0, 100.0, 500.0),
    pins=(
        SpecPin(
            label="flat_price_never_moves_the_total",
            inputs={"price": (5.0, 5.0, 5.0, 5.0), "volume": (10.0, 20.0, 30.0, 40.0)},
            expected=(0.0, 0.0, 0.0, 0.0),
            reason="an unchanged price contributes no signed volume, so the running total stays at the seed 0 "
            "(test_obv.py::TestObvEdge::test_flat_price_no_change)",
        ),
    ),
)
