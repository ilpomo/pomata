"""Spec for ``pomata.indicators.parabolic_sar`` — Wilder's stop-and-reverse, propagating, degree-1 homogeneous."""

from tests.indicators.oracles import parabolic_sar_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import parabolic_sar

PARABOLIC_SAR = Spec(
    factory=parabolic_sar,
    inputs=("high", "low"),
    params={"acceleration": 0.02, "maximum": 0.2},
    shape=Shape.SERIES,
    warmup=1,
    raises=(
        ({"acceleration": 0.0}, r"acceleration must be in the half-open interval \(0, 1\]"),
        ({"maximum": 0.0}, r"maximum must be in the half-open interval \(0, 1\]"),
        ({"acceleration": 1.5, "maximum": 2.0}, r"acceleration must be in the half-open interval \(0, 1\]"),
        ({"maximum": 1.5}, r"maximum must be in the half-open interval \(0, 1\]"),
        ({"acceleration": 0.5, "maximum": 0.3}, r"acceleration must be <= maximum"),
    ),
    # The stop is a recursion whose step shrinks by (1 - acceleration_factor); after an interior null the state lags the
    # clean baseline by one bar and reconverges only geometrically (factor 0.8 once the factor caps at maximum=0.2), so
    # the flow rung's return-to-baseline check must look far past the missing bar for that decay to clear the tolerance.
    flow_horizon=130,
    oracle=parabolic_sar_reference,
    # Homogeneous of degree 1 under a positive common rescaling of high and low.
    scale=(ScaleAxis(roles=("high", "low"), degree=1),),
    golden_input={
        "high": (10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0, 10.0, 11.0),
        "low": (9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0, 9.0, 10.0),
    },
    golden_output=(None, 9.0, 9.0, 9.12, 9.3528, 9.7246, 10.0666, 14.0, 13.92, 13.7232),
    pins=(
        SpecPin(
            label="short_seed_golden",
            inputs={"high": (10.0, 9.0, 8.0, 7.0, 8.5), "low": (9.0, 8.0, 7.0, 6.0, 7.5)},
            expected=(None, 10.0, 10.0, 9.88, 9.647200000000002),
            reason="a hand-derived golden whose first bar pair falls, so the trend seeds short — the branch the "
            "long-seeding top-level golden never reaches",
        ),
        SpecPin(
            label="null_bridging_golden",
            inputs={"high": (10.0, 11.0, 12.0, None, 13.0, 14.0), "low": (9.0, 10.0, 11.0, 11.5, 12.0, 13.0)},
            expected=(None, 9.0, 9.0, None, 9.12, 9.352799999999998),
            reason="the documented null-bridging behavior: the null row emits null while the running trend state is "
            "untouched and resumes on the next complete bar",
        ),
    ),
)
