"""Spec for ``pomata.indicators.donchian_channels`` — the rolling high/low channel struct, window-nulling, degree-1."""

from tests.indicators.oracles import donchian_channels_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import donchian_channels

DONCHIAN_CHANNELS = Spec(
    factory=donchian_channels,
    inputs=("high", "low"),
    params={"window": 20},
    shape=Shape.STRUCT,
    fields=("lower", "middle", "upper"),
    warmup={"lower": 19, "middle": 19, "upper": 19},
    lands_on="low",
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=donchian_channels_reference,
    # Every band is a price extreme (or their mean), homogeneous of degree 1
    scale=(ScaleAxis(roles=("high", "low"), degree={"lower": 1, "middle": 1, "upper": 1}),),
    golden_params={"window": 3},
    golden_input={
        "high": (11.0, 12.0, 13.0, 12.5, 14.0),
        "low": (9.0, 10.0, 11.0, 11.0, 12.0),
    },
    golden_output={
        "lower": (None, None, 9.0, 10.0, 11.0),
        "middle": (None, None, 11.0, 11.5, 12.5),
        "upper": (None, None, 13.0, 13.0, 14.0),
    },
    pins=(
        SpecPin(
            label="window_one_tracks_each_bar",
            inputs={"high": (11.0, 12.0, 13.0), "low": (9.0, 10.0, 11.0)},
            params_override={"window": 1},
            expected={
                "lower": (9.0, 10.0, 11.0),
                "middle": (10.0, 11.0, 12.0),
                "upper": (11.0, 12.0, 13.0),
            },
            reason="window=1 makes the upper/lower channel the bar's own high/low and the middle their mean, with no "
            "warm-up",
        ),
    ),
)
