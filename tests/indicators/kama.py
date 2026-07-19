"""
Declaration for ``pomata.indicators.kama`` — Kaufman's adaptive recursive mean, gap-bridging, NaN-latching, degree-1.
"""

from pomata.indicators import kama
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_kama
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

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
    reference="Kaufman, P. J. (1995). *Smarter Trading: Improving Performance in Changing Markets*. McGraw-Hill.",
    reference_url="https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/kaufmans-adaptive-moving-average-kama",
    see_also=(
        ("ema", "The fixed-smoothing exponential average KAMA adapts between."),
        ("rma", "Wilder's fixed-smoothing average."),
        ("mama", "The MESA adaptive average, steered by cycle phase rather than efficiency."),
    ),
    opener_override="The efficiency ratio and adaptive smoothing constant are checked against an independent "
    "reference, but the seeded recurrence they drive is one-shape with the implementation, so "
    "the oracle confirms its internal consistency, not its independence; the independent "
    "witnesses are the TA-Lib differential and frozen hand-derived golden masters. Agreement "
    "holds to ten significant figures (a ``1e-10`` band) on any finite input within a sane "
    "dynamic range; the documentation's *Correctness* page gives the method and the "
    "float-conditioning limit beyond it."
    "\n\n"
    "It is homogeneous of degree ``1`` (the efficiency ratio is scale-invariant — a ratio of "
    "absolute moves — and the recurrence is linear in the input, so ``kama(k * x) == k * "
    "kama(x)``).",
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap — "
            "whether the ``null`` reaches the recurrence directly through ``close`` or via the "
            "efficiency-ratio window touching one.",
        ),
        (
            "NaN",
            "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent non-null position.",
        ),
        (
            "Degenerate denominator",
            "when there is no bar-to-bar travel the efficiency ratio is taken as ``0`` (avoiding the "
            "``0 / 0`` degenerate), so the smoothing constant sits at the slow bound and KAMA barely "
            "moves.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The KAMA for each row, the same length as ``expr``. The first ``window - 1`` values are "
    "``null`` (warm-up); the value at row ``window - 1`` is ``close`` itself (the seed), and "
    "the adaptive recurrence runs from there.",
    raises_prose="ValueError: If ``window < 1``, ``window_fast < 1``, ``window_slow < 1``, or "
    "``window_fast > window_slow``.",
    args_prose={
        "window": "Number of observations in the efficiency-ratio look-back. Must be ``>= 1``.",
        "window_fast": "Period of the fast smoothing-constant bound (canonically ``2``), ``2 / (window_fast + "
        "1)``. Must be ``>= 1`` (the fast bound is the more responsive end of the adaptive "
        "range).",
        "window_slow": "Period of the slow smoothing-constant bound (canonically ``30``), ``2 / (window_slow + "
        "1)``. Must be ``>= 1`` and ``>= window_fast``.",
    },
    example_columns={"price": "close"},
    examples=(
        Example(
            inputs={"price": (10.0, 11.0, 12.0, 11.0, 13.0, 12.5)},
            params={"window": 2, "window_fast": 2, "window_slow": 30},
            round_to=4,
        ),
        Example(
            inputs={"price": (10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0, 22.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("A", "A", "A", "A", "A", "B", "B", "B", "B", "B"),
            params={"window": 2, "window_fast": 2, "window_slow": 30},
            round_to=4,
        ),
        Example(
            inputs={"price": (10.0, 11.0, 12.0, None, 13.0, float("nan"), 15.0, 16.0)},
            intro="A ``null`` (bridged) and a ``NaN`` (latched) make the handling visible:",
            params={"window": 2, "window_fast": 2, "window_slow": 30},
            round_to=4,
        ),
        Example(
            inputs={"price": (5.0, 5.0, 5.0, 5.0)},
            intro="**Degenerate denominator** — a flat series has zero bar-to-bar travel, so the efficiency "
            "ratio is taken as ``0`` (avoiding the ``0 / 0`` degenerate) and KAMA holds at the "
            "constant:",
            params={"window": 2, "window_fast": 2, "window_slow": 30},
        ),
    ),
)
