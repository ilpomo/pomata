"""Spec for ``pomata.indicators.vortex`` — the plus/minus vortex movement pair, window-nulling, scale-invariant."""

from tests.indicators.oracles import vortex_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import vortex

VORTEX = Spec(
    factory=vortex,
    inputs=("high", "low", "close"),
    params={"window": 14},
    shape=Shape.STRUCT,
    fields=("plus", "minus"),
    warmup=14,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=vortex_reference,
    # Each line is a ratio of a rolling vortex movement to the rolling true range, scale-INVARIANT, degree 0
    # (tests/indicators/test_vortex.py::TestVortexProperties::test_scale_invariance).
    scale=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    golden_params={"window": 2},
    golden_input={
        "high": (2.0, 4.0, 6.0, 5.0, 7.0),
        "low": (1.0, 3.0, 4.0, 4.0, 5.0),
        "close": (1.5, 3.5, 5.0, 4.5, 6.0),
    },
    golden_output={
        "plus": (None, None, 1.2, 1.1429, 1.1429),
        "minus": (None, None, 0.2, 0.5714, 0.5714),
    },
    pins=(
        SpecPin(
            label="window_one_single_bar_ratio",
            inputs={"high": (2.0, 4.0, 6.0), "low": (1.0, 3.0, 4.0), "close": (1.5, 3.5, 5.0)},
            params_override={"window": 1},
            expected={"plus": (None, 1.2, 1.2), "minus": (None, 0.4, 0.0)},
            reason="window=1 reduces each line to a single bar's vortex movement over its true range, the first bar "
            "warm-up (no prior bar) (tests/indicators/test_vortex.py::TestVortexEdge::test_window_one)",
        ),
    ),
)
