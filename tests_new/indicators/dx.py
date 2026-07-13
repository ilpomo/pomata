"""Spec for ``pomata.indicators.dx`` — Wilder's directional movement index, gap-bridging, scale-invariant."""

from tests.indicators.oracles import dx_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import dx

DX = Spec(
    factory=dx,
    inputs=("high", "low", "close"),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=13,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=dx_reference,
    # The normalized spread of the two directional indicators, bounded in [0, 100] and scale-INVARIANT, degree 0
    # (tests/indicators/test_dx.py::TestDxProperties::test_scale_invariance).
    scale=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    golden_params={"window": 2},
    golden_input={
        "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0),
        "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0),
        "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5),
    },
    golden_output=(None, 100.0, 100.0, 20.0, 76.4706, 20.0, 72.6027),
)
