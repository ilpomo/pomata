"""Spec for ``pomata.indicators.fisher_transform`` — the Gaussianized channel struct (fisher, signal)."""

import math

from tests.indicators.oracles import fisher_transform_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import fisher_transform

FISHER_TRANSFORM = Spec(
    factory=fisher_transform,
    inputs=("high", "low"),
    params={"window": 10},
    shape=Shape.STRUCT,
    fields=("fisher", "signal"),
    warmup={"fisher": 9, "signal": 10},
    raises=(({"window": 0}, r"window must be >= 1"),),
    # The fisher line is a half-weight recursion, so an interior null perturbs the state and decays back to the clean
    # baseline only geometrically (0.5 per bar); the flow rung's return-to-baseline check must look far enough past the
    # missing bar for that decay to fall inside the reference tolerance.
    flow_horizon=60,
    oracle=fisher_transform_reference,
    # A rolling-channel normalization, scale-INVARIANT, degree 0 (tests/indicators/test_fisher_transform.py
    # ::TestFisherTransformProperties::test_scale_invariance).
    scale=(ScaleAxis(roles=("high", "low"), degree={"fisher": 0, "signal": 0}),),
    golden_params={"window": 2},
    golden_input={"high": (2.0, 4.0, 3.0), "low": (0.0, 2.0, 1.0)},
    golden_output={
        "fisher": (None, 0.3428, 0.0621),
        "signal": (None, None, 0.3428),
    },
    pins=(
        SpecPin(
            label="window_one_single_row_is_flat_nan",
            inputs={"high": (11.0,), "low": (9.0,)},
            params_override={"window": 1},
            expected={"fisher": (math.nan,), "signal": (None,)},
            reason="window=1 is flat by construction (max == min), so fisher is NaN from the first row while signal "
            "is still warm-up null (test_fisher_transform.py::TestFisherTransformEdge::test_single_row)",
        ),
        SpecPin(
            label="flat_window_is_nan",
            inputs={"high": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0)},
            params_override={"window": 3},
            expected={
                "fisher": (None, None, math.nan, math.nan, math.nan, math.nan),
                "signal": (None, None, None, math.nan, math.nan, math.nan),
            },
            reason="a constant series has max == min over every window: the channel normalization is 0/0 NaN, which "
            "bridges through the recursion "
            "(test_fisher_transform.py::TestFisherTransformEdge::test_flat_window_is_nan)",
        ),
    ),
)
