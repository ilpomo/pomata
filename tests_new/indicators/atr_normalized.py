"""Spec for ``pomata.indicators.atr_normalized`` — the ATR as a percentage of close, gap-bridging, scale-invariant."""

from tests_new.indicators.oracles import atr_normalized_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import atr_normalized

ATR_NORMALIZED = Spec(
    factory=atr_normalized,
    inputs=("high", "low", "close"),
    params={"window": 14},
    shape=Shape.SERIES,
    warmup=13,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=atr_normalized_reference,
    # The ATR divided by the close, a percentage that is scale-INVARIANT, degree 0 (tests/indicators/
    # test_atr_normalized.py::TestAtrNormalizedProperties::test_scale_invariance).
    scale=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    golden_params={"window": 2},
    golden_input={
        "high": (10.2, 10.5, 10.7, 10.3, 10.8),
        "low": (9.8, 10.0, 10.2, 9.9, 10.3),
        "close": (10.0, 10.3, 10.5, 10.1, 10.6),
    },
    golden_output=(None, 4.3689, 4.5238, 5.3218, 5.8373),
)
