"""Spec for ``pomata.indicators.bollinger_bands`` — the SMA-and-deviation band struct, window-nulling, degree-1."""

import math

from tests_new.indicators.oracles import bollinger_bands_reference
from tests_new.support import ABSOLUTE_TOLERANCE_SCALE, RELATIVE_TOLERANCE_SCALE
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import bollinger_bands

BOLLINGER_BANDS = Spec(
    factory=bollinger_bands,
    inputs=("price",),
    params={"window": 20, "multiplier": 2.0},
    shape=Shape.STRUCT,
    fields=("lower", "middle", "upper"),
    warmup={"lower": 19, "middle": 19, "upper": 19},
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"multiplier": 0.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -1.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.nan}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.inf}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -math.inf}, r"multiplier must be a finite number > 0"),
    ),
    oracle=bollinger_bands_reference,
    # The bands ride a one-pass rolling deviation against a two-pass oracle: the fixed streaming band over the
    # well-conditioned domain, matching every other one-pass moment family.
    oracle_rel_tol=RELATIVE_TOLERANCE_SCALE,
    oracle_abs_tol=ABSOLUTE_TOLERANCE_SCALE,
    # Every band is a price level (mean plus/minus a dispersion), homogeneous of degree 1 (tests/indicators/
    # test_bollinger_bands.py::TestBollingerBandsProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("price",), degree={"lower": 1, "middle": 1, "upper": 1}),),
    golden_params={"window": 2},
    golden_input={"price": (2.0, 4.0, 4.0, 8.0)},
    golden_output={
        "lower": (None, 1.0, 4.0, 2.0),
        "middle": (None, 3.0, 4.0, 6.0),
        "upper": (None, 5.0, 4.0, 10.0),
    },
    pins=(
        SpecPin(
            label="multiplier_one_halves_the_band_width",
            inputs={"price": (2.0, 4.0, 4.0, 8.0)},
            params_override={"window": 2, "multiplier": 1.0},
            expected={
                "lower": (None, 2.0, 4.0, 4.0),
                "middle": (None, 3.0, 4.0, 6.0),
                "upper": (None, 4.0, 4.0, 8.0),
            },
            reason="the band half-width is linear in the multiplier: at multiplier=1 it is half of the default, the "
            "arithmetic a single default-multiplier golden cannot exercise (test_bollinger_bands.py"
            "::TestBollingerBandsCorrectness::test_multiplier_scales_width)",
        ),
        SpecPin(
            label="constant_window_collapses_bands_after_large_value",
            inputs={"price": (1000000.0, 0.1, 0.1, 0.1, 0.1)},
            params_override={"window": 3},
            expected={
                "lower": (None, None, -609475.5473011592, 0.1, 0.1),
                "middle": (None, None, 333333.39999999997, 0.1, 0.1),
                "upper": (None, None, 1276142.3473011593, 0.1, 0.1),
            },
            reason="a constant window has exactly zero (pinned) deviation, so all three bands collapse onto the middle "
            "even after a much larger value has left the window, where the rolling kernel would otherwise leave a "
            "residue (test_bollinger_bands.py::TestBollingerBandsEdge::test_constant_window_collapses_bands)",
        ),
    ),
)
