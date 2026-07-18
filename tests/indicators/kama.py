"""
Declaration for ``pomata.indicators.kama`` — Kaufman's adaptive recursive mean, gap-bridging, NaN-latching, degree-1.
"""

from pomata.indicators import kama
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_kama
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

KAMA = suite_indicators(
    factory=kama,
    inputs=("price",),
    params={"window": 2, "window_fast": 2, "window_slow": 30},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_kama,
    scaling=(ScaleAxis(roles=("price",), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"window_fast": 0}, r"window_fast must be >= 1"),
        ({"window_slow": 0}, r"window_slow must be >= 1"),
        ({"window_fast": 30, "window_slow": 2}, r"windows must be ordered window_fast <= window_slow"),
    ),
    golden=Golden(
        inputs={"price": (10.0, 11.0, 12.0, 11.0, 13.0, 12.5)}, output=(None, 11.0, 11.4444, 11.4426, 11.5522, 11.724)
    ),
    pins=(
        Pin(
            label="flat_window_efficiency_ratio_zero",
            inputs={"price": (5.0, 5.0, 5.0, 5.0)},
            expected=(None, 5.0, 5.0, 5.0),
            reason="a flat series gives efficiency ratio 0 (the volatility==0 guard avoids 0/0), so KAMA stays pinned "
            "on the constant",
        ),
        Pin(
            label="interior_null_bridged",
            inputs={"price": (2.0, 4.0, None, 8.0, 10.0, 12.0)},
            expected=(None, 4.0, None, None, None, 7.555555555555554),
            reason="an interior null nulls its own row and the windows touching it; the recursion resumes from the "
            "seed carried across the gap",
        ),
    ),
)
