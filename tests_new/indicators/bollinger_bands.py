"""
Declaration for ``pomata.indicators.bollinger_bands`` — the SMA-and-deviation band struct, window-nulling, degree-1.
"""

import math

from pomata.indicators import bollinger_bands
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_bollinger_bands
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape
from tests_new.support.tolerances import TOLERANCE_ABSOLUTE_ROLLING_ORACLE, TOLERANCE_RELATIVE_ROLLING_ORACLE

BOLLINGER_BANDS = suite_indicators(
    factory=bollinger_bands,
    inputs=("price",),
    params={"window": 20, "multiplier": 2.0},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.STRUCT,
    fields=("lower", "middle", "upper"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"lower": 19, "middle": 19, "upper": 19},
    oracle=reference_bollinger_bands,
    scaling=(ScaleAxis(roles=("price",), degree={"lower": 1, "middle": 1, "upper": 1}),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"multiplier": 0.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -1.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.nan}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.inf}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -math.inf}, r"multiplier must be a finite number > 0"),
    ),
    oracle_rel_tol=TOLERANCE_RELATIVE_ROLLING_ORACLE,
    oracle_abs_tol=TOLERANCE_ABSOLUTE_ROLLING_ORACLE,
    golden=Golden(
        inputs={"price": (2.0, 4.0, 4.0, 8.0)},
        output={
            "lower": (None, 1.0, 4.0, 2.0),
            "middle": (None, 3.0, 4.0, 6.0),
            "upper": (None, 5.0, 4.0, 10.0),
        },
        params={"window": 2},
    ),
    pins=(
        Pin(
            label="multiplier_one_halves_the_band_width",
            inputs={"price": (2.0, 4.0, 4.0, 8.0)},
            params_override={"window": 2, "multiplier": 1.0},
            expected={
                "lower": (None, 2.0, 4.0, 4.0),
                "middle": (None, 3.0, 4.0, 6.0),
                "upper": (None, 4.0, 4.0, 8.0),
            },
            reason="the band half-width is linear in the multiplier: at multiplier=1 it is half of the default, the "
            "arithmetic a single default-multiplier golden cannot exercise",
        ),
        Pin(
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
            "residue",
        ),
    ),
)
