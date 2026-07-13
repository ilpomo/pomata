"""Spec for ``pomata.indicators.kama`` — Kaufman's adaptive recursive mean, gap-bridging, NaN-latching, degree-1."""

from tests.indicators.oracles import kama_reference
from tests.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import kama

KAMA = Spec(
    factory=kama,
    inputs=("price",),
    params={"window": 2, "window_fast": 2, "window_slow": 30},
    shape=Shape.SERIES,
    warmup=1,
    raises=(
        ({"window": 0}, r"window must be >= 1"),
        ({"window_fast": 0}, r"window_fast must be >= 1"),
        ({"window_slow": 0}, r"window_slow must be >= 1"),
        ({"window_fast": 30, "window_slow": 2}, r"windows must be ordered window_fast <= window_slow"),
    ),
    oracle=kama_reference,
    # A degree-1 homogeneous adaptive mean: kama(k*x) == k*kama(x).
    scale=(ScaleAxis(roles=("price",), degree=1),),
    golden_input={"price": (10.0, 11.0, 12.0, 11.0, 13.0, 12.5)},
    golden_output=(None, 11.0, 11.4444, 11.4426, 11.5522, 11.724),
    pins=(
        SpecPin(
            label="flat_window_efficiency_ratio_zero",
            inputs={"price": (5.0, 5.0, 5.0, 5.0)},
            expected=(None, 5.0, 5.0, 5.0),
            reason="a flat series gives efficiency ratio 0 (the volatility==0 guard avoids 0/0), so KAMA stays pinned "
            "on the constant",
        ),
        SpecPin(
            label="interior_null_bridged",
            inputs={"price": (2.0, 4.0, None, 8.0, 10.0, 12.0)},
            expected=(None, 4.0, None, None, None, 7.555555555555554),
            reason="an interior null nulls its own row and the windows touching it; the recursion resumes from the "
            "seed carried across the gap",
        ),
    ),
)
