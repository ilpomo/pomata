"""Spec for ``pomata.indicators.adxr`` — Wilder's average directional index rating, gap-bridging, scale-invariant."""

from tests.indicators.oracles import adxr_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import adxr

ADXR = Spec(
    factory=adxr,
    inputs=("high", "low", "close"),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=40,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=adxr_reference,
    # The mean of the current ADX and the ADX one window back, bounded in [0, 100] and scale-INVARIANT, degree 0
    # (tests/indicators/test_adxr.py::TestAdxrProperties::test_scale_invariance).
    scale=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    golden_params={"window": 2},
    golden_input={
        "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5),
        "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5),
        "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0),
    },
    golden_output=(None, None, None, None, 84.1176, 52.0588, 63.2977, 41.6489, 56.9044, 38.4522),
)
