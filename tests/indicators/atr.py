"""Declaration for ``pomata.indicators.atr`` — Wilder's Average True Range, gap-bridging, NaN-latching, degree-1."""

from pomata.indicators import atr
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_atr
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

ATR = suite_indicators(
    factory=atr,
    inputs=("high", "low", "close"),
    params={"window": 14},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_atr,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0),
            "low": (8.0, 9.0, 9.5, 10.0, 12.0, 11.0, 13.0, 15.0),
            "close": (9.0, 11.0, 10.0, 12.0, 14.0, 13.0, 15.0, 17.0),
        },
        output=(None, None, 2.1667, 2.4444, 2.6296, 2.7531, 2.8354, 2.8903),
        params={"window": 3},
    ),
    pins=(
        Pin(
            label="window_one_is_true_range",
            inputs={"high": (10.0, 12.0, 11.0, 13.0), "low": (8.0, 9.0, 9.5, 10.0), "close": (9.0, 11.0, 10.0, 12.0)},
            params_override={"window": 1},
            expected=(2.0, 3.0, 1.5, 3.0),
            reason="window=1 makes the Wilder smoothing the identity, so the ATR reproduces the true range",
        ),
    ),
    reference="Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.",
    wikipedia="https://en.wikipedia.org/wiki/Average_true_range",
    see_also=(
        ("true_range", "The per-bar range this Wilder-smooths."),
        ("rma", "The Wilder moving average used for the smoothing."),
        ("atr_normalized", "The same ATR expressed as a percent of the current close."),
    ),
    notes=(
        (
            "Scaling",
            "Scaling is homogeneous of degree ``1`` only for a positive factor: multiplying every "
            "price by ``k > 0`` scales the ATR by ``k``. A negative factor makes the bar incoherent "
            "(``high`` falls below ``low``), so it is not a clean rescale.",
        ),
        (
            "Seeding",
            "The Wilder smoothing (:func:`rma`) is seeded with the simple average of the first "
            "``window`` true ranges -- Wilder's canonical initialization. The first true range is the "
            "bar's high-low range (no prior close extends it), so the seed and warm-up include it.",
        ),
    ),
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap — "
            "the true range is ``null`` only when every :func:`true_range` candidate term is "
            "``null``.",
        ),
        (
            "NaN",
            "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent "
            "non-null position — except at ``window == 1``, where the smoothing is the identity and "
            "no recursion exists to latch: the ``NaN`` clears once it leaves the true range's one-bar "
            "reach.",
        ),
        (
            "window == 1",
            "the smoothing factor is ``1`` and the warm-up vanishes, so the ATR reproduces the true "
            "range exactly: the ``max_horizontal``-reduced true range (not a textbook three-term true "
            "range whenever a candidate term is dropped by a ``null``).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The ATR for each row, the same length as the inputs. The first ``window - 1`` values are "
    "``null`` (warm-up), inherited from the :func:`rma` over the true-range series: the "
    "running average emits only once ``window`` non-null true ranges have been counted, "
    "independent of where any interior ``null`` falls."
    "\n\n"
    "The true range itself is defined from row ``0`` (the first bar has no previous close, so "
    "it degenerates to ``high - low`` with the two gap terms dropped), so the ATR warm-up is "
    "exactly the ``rma`` warm-up of ``window - 1``.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "close": 'Close-price series (e.g. ``pl.col("close")``); the previous close supplies the two gap terms.',
        "window": "Number of observations in the Wilder moving window. Must be ``>= 1``.",
    },
    examples=(
        Example(
            inputs={
                "high": (10.0, 12.0, 13.0, 12.0, 14.0),
                "low": (9.0, 10.0, 11.0, 10.0, 12.0),
                "close": (9.5, 11.0, 12.0, 11.0, 13.0),
            },
            params={"window": 3},
            round_to=4,
        ),
        Example(
            inputs={
                "high": (12.0, 13.0, 12.5, 14.0, 22.0, 24.0, 23.0, 25.0),
                "low": (10.0, 11.0, 11.0, 12.0, 20.0, 21.0, 21.0, 23.0),
                "close": (11.0, 12.5, 11.5, 13.5, 21.5, 21.5, 22.5, 24.0),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("AAPL",) * 4 + ("NVDA",) * 4,
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={
                "high": (12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0),
                "low": (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0),
                "close": (11.5, 12.5, None, 14.5, 15.5, float("nan"), 17.5, 18.0),
            },
            intro="A ``null`` ``close`` (absorbed, so the next bar falls back to ``high - low``) then a "
            "``NaN`` ``close`` (which the Wilder recursion latches from the next bar on) make the "
            "exact handling visible at a glance:",
            params={"window": 2},
            round_to=4,
        ),
        Example(
            inputs={"high": (10.0, 12.0, 11.0, 13.0), "low": (8.0, 9.0, 9.5, 10.0), "close": (9.0, 11.0, 10.0, 12.0)},
            intro="**window == 1** — window=1 makes the Wilder smoothing the identity, so the ATR "
            "reproduces the true range:",
            params={"window": 1},
        ),
    ),
)
