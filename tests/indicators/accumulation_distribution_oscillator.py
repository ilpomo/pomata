"""Spec for ``pomata.indicators.accumulation_distribution_oscillator`` — the Chaikin A/D oscillator, gap-bridging."""

from tests.indicators.oracles import accumulation_distribution_oscillator_reference
from tests.support.spec import ScaleAxis, Shape, Spec

from pomata.indicators import accumulation_distribution_oscillator

ACCUMULATION_DISTRIBUTION_OSCILLATOR = Spec(
    factory=accumulation_distribution_oscillator,
    inputs=("high", "low", "close", "volume"),
    params={"window_fast": 3, "window_slow": 10},
    shape=Shape.SERIES,
    warmup=9,
    lands_on="close",
    raises=(
        ({"window_fast": 0}, r"window_fast must be >= 1"),
        ({"window_slow": 0}, r"window_slow must be >= 1"),
        ({"window_fast": 10, "window_slow": 3}, r"windows must be ordered window_fast <= window_slow"),
    ),
    oracle=accumulation_distribution_oscillator_reference,
    # The difference of two EMAs of the A/D line, homogeneous of degree 1 in volume (the line scales with volume while
    # the multiplier is price-invariant, degree 0).
    scale=(
        ScaleAxis(roles=("volume",), degree=1),
        ScaleAxis(roles=("high", "low", "close"), degree=0),
    ),
    golden_params={"window_fast": 2, "window_slow": 3},
    golden_input={
        "high": (10.2, 10.5, 10.7, 10.3, 10.8),
        "low": (9.8, 10.0, 10.2, 9.9, 10.3),
        "close": (10.0, 10.3, 10.5, 10.1, 10.6),
        "volume": (100.0, 150.0, 120.0, 200.0, 180.0),
    },
    golden_output=(None, None, 13.0, 8.6667, 11.0556),
)
