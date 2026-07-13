"""Spec for ``pomata.indicators.aroon`` — the time-since-extreme struct (up, down), window-nulling, scale-invariant."""

from tests_new.indicators.oracles import aroon_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import aroon

AROON = Spec(
    factory=aroon,
    inputs=("high", "low"),
    params={"window": 25},
    shape=Shape.STRUCT,
    fields=("up", "down"),
    warmup={"up": 25, "down": 25},
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=aroon_reference,
    # Each line depends only on the position of the window extreme, so it is scale-INVARIANT, degree 0
    # (tests/indicators/test_aroon.py::test_scale_invariance).
    scale=(ScaleAxis(roles=("high", "low"), degree={"up": 0, "down": 0}),),
    golden_params={"window": 3},
    golden_input={
        "high": (10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0),
        "low": (9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0),
    },
    golden_output={
        "up": (None, None, None, 66.6667, 100.0, 66.6667, 100.0, 66.6667),
        "down": (None, None, None, 0.0, 66.6667, 33.3333, 0.0, 33.3333),
    },
    pins=(
        SpecPin(
            label="current_extreme_reads_100",
            inputs={"high": (1.0, 2.0, 3.0), "low": (3.0, 2.0, 1.0)},
            params_override={"window": 2},
            expected={"up": (None, None, 100.0), "down": (None, None, 100.0)},
            reason="when the current bar holds the look-back high (low) the up (down) line reads 100 "
            "(test_aroon.py::TestAroonEdge::test_current_extreme_reads_100)",
        ),
        SpecPin(
            label="ties_use_most_recent_extreme",
            inputs={"high": (5.0, 5.0, 3.0), "low": (1.0, 2.0, 3.0)},
            params_override={"window": 2},
            expected={"up": (None, None, 50.0), "down": (None, None, 0.0)},
            reason="a repeated high resolves to the most recent occurrence (one bar back, up=50) "
            "(test_aroon.py::TestAroonEdge::test_ties_use_most_recent_extreme)",
        ),
    ),
)
