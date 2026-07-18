"""
Declaration for ``pomata.indicators.atr_normalized`` — the ATR as a percentage of close, gap-bridging, scale-
invariant.
"""

from pomata.indicators import atr_normalized
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_atr_normalized
from tests.support.declaration import Golden, ScaleAxis, Shape

ATR_NORMALIZED = suite_indicators(
    factory=atr_normalized,
    inputs=("high", "low", "close"),
    params={"window": 14},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_atr_normalized,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    golden=Golden(
        inputs={
            "high": (10.2, 10.5, 10.7, 10.3, 10.8),
            "low": (9.8, 10.0, 10.2, 9.9, 10.3),
            "close": (10.0, 10.3, 10.5, 10.1, 10.6),
        },
        output=(None, 4.3689, 4.5238, 5.3218, 5.8373),
        params={"window": 2},
    ),
    reference="Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.",
    see_also=(
        ("atr", "The raw (price-unit) average true range this normalizes."),
        ("true_range", "The per-bar range underlying the ATR."),
        ("bollinger_bands", "Another volatility view, standard-deviation bands around a moving average."),
    ),
    note_extension="\n\n"
    "It is scale-invariant under a positive common rescaling of ``high``, ``low``, and "
    "``close`` (the ATR and the close scale together).",
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap — "
            "inherited from :func:`atr`, with a ``null`` ``close`` also nulling the ratio at that "
            "row.",
        ),
        (
            "NaN",
            "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent "
            "non-null position — inherited from :func:`atr`, with a ``NaN`` ``close`` also yielding "
            "``NaN`` for the ratio at that row.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The NATR (in percent) for each row, the same length as the inputs. The first ``window - "
    "1`` values are ``null`` (warm-up), inherited from the :func:`atr`.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the Wilder moving window. Must be ``>= 1``.",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
    intro_missing="A ``null`` ``close`` (voiding the ratio at that row) then a ``NaN`` ``close`` (which "
    "propagates through the ratio and the latched ATR) make the missing-data handling visible "
    "at a glance:",
)
