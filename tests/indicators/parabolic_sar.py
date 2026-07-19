"""
Declaration for ``pomata.indicators.parabolic_sar`` — Wilder's stop-and-reverse, propagating, degree-1 homogeneous.
"""

from pomata.indicators import parabolic_sar
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_parabolic_sar
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

PARABOLIC_SAR = suite_indicators(
    factory=parabolic_sar,
    inputs=("high", "low"),
    params={"acceleration": 0.02, "maximum": 0.2},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=1,
    oracle=reference_parabolic_sar,
    scaling=(ScaleAxis(roles=("high", "low"), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"acceleration": 0.0}, r"acceleration must be in the half-open interval \(0, 1\]"),
        ({"maximum": 0.0}, r"maximum must be in the half-open interval \(0, 1\]"),
        ({"acceleration": 1.5, "maximum": 2.0}, r"acceleration must be in the half-open interval \(0, 1\]"),
        ({"maximum": 1.5}, r"maximum must be in the half-open interval \(0, 1\]"),
        ({"acceleration": 0.5, "maximum": 0.3}, r"acceleration must be <= maximum"),
    ),
    flow_horizon=130,
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0, 10.0, 11.0),
            "low": (9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0, 9.0, 10.0),
        },
        output=(None, 9.0, 9.0, 9.12, 9.3528, 9.7246, 10.0666, 14.0, 13.92, 13.7232),
    ),
    pins=(
        Pin(
            label="short_seed_golden",
            inputs={"high": (10.0, 9.0, 8.0, 7.0, 8.5), "low": (9.0, 8.0, 7.0, 6.0, 7.5)},
            expected=(None, 10.0, 10.0, 9.88, 9.647200000000002),
            reason="a hand-derived golden whose first bar pair falls, so the trend seeds short — the branch the "
            "long-seeding top-level golden never reaches",
        ),
        Pin(
            label="null_bridging_golden",
            inputs={"high": (10.0, 11.0, 12.0, None, 13.0, 14.0), "low": (9.0, 10.0, 11.0, 11.5, 12.0, 13.0)},
            expected=(None, 9.0, 9.0, None, 9.12, 9.352799999999998),
            reason="the documented null-bridging behavior: the null row emits null while the running trend state is "
            "untouched and resumes on the next complete bar",
        ),
    ),
    reference="Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.",
    wikipedia="https://en.wikipedia.org/wiki/Parabolic_SAR",
    see_also=(
        ("supertrend", "The other trailing-stop trend tool, ATR-scaled rather than accelerating."),
        ("adx", "Wilder's directional-movement trend-strength index."),
        ("atr", "Wilder's volatility average."),
    ),
    notes=(
        (
            "Seeding",
            "Wilder's original leaves the initial trend unspecified; here it is taken long when the "
            "first bar-to-bar up-move is at least the down-move, else short, and the first stop is "
            "the prior low (long) or high (short).",
        ),
    ),
    opener_override="The parabolic SAR is a path-dependent stop-and-reverse recurrence, so its reference "
    "oracle necessarily mirrors the implementation's state machine and confirms internal "
    "consistency, not independence; the independent witness is the set of golden masters "
    "hand-computed from Wilder's published rules. Agreement holds to ten significant figures "
    "(a ``1e-10`` band) on any finite input within a sane dynamic range; the documentation's "
    "*Correctness* page gives the method and the float-conditioning limit beyond it."
    "\n\n"
    "It is homogeneous of degree ``1`` under a positive common rescaling of ``high`` and "
    "``low`` (the stop is a price level and the recurrence and crossings are linear in "
    "price).",
    bullets=(
        (
            "Null",
            "a ``null`` price makes that row ``null`` (``null`` takes precedence over ``NaN``) — the "
            "running trend state bridges the gap and resumes on the next complete bar, later rows "
            "reconverging as the stop recurrence contracts.",
        ),
        (
            "NaN",
            "a ``NaN`` inside the window propagates, yielding ``NaN`` there — the raw high/low feed "
            "the kernel directly with no recurrence to latch onto, so the running trend state bridges "
            "the gap and resumes on the next complete bar.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The Parabolic SAR for each row, the same length as the inputs. Row ``0`` is ``null`` "
    "(the trend is seeded from the first two bars); the value at row ``1`` is the seed stop, "
    "and the recurrence runs from there.",
    raises_prose="ValueError: If ``acceleration`` or ``maximum`` is not in the half-open interval ``(0, "
    "1]``, or if ``acceleration > maximum``.",
    args_prose={
        "acceleration": "Starting acceleration factor, and its per-extreme increment (canonical default ``0.02``, "
        "Wilder's step). Must be in the half-open interval ``(0, 1]``, and never above "
        "``maximum`` (so the factor is capped from the seed onward, not only on the increment "
        "path).",
        "maximum": "Cap on the acceleration factor. Must be in the half-open interval ``(0, 1]``, and at "
        "least ``acceleration``.",
    },
    examples=(
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0, 10.0, 11.0),
                "low": (9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0, 9.0, 10.0),
            },
            round_to=4,
        ),
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 22.0, 21.0, 20.0),
                "low": (9.0, 10.0, 11.0, 12.0, 13.0, 19.0, 20.0, 21.0, 20.0, 19.0),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker seeds independently:",
            partition=("AAPL",) * 5 + ("NVDA",) * 5,
            round_to=4,
        ),
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, None, 14.0, float("nan"), 12.0, 11.0),
                "low": (9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0),
            },
            intro="A ``null`` then a ``NaN`` in ``high`` each yield ``null`` / ``NaN`` at that row and are "
            "skipped, the running trend state bridging the gap:",
            round_to=4,
        ),
    ),
)
