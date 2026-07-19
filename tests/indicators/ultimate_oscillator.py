"""
Declaration for ``pomata.indicators.ultimate_oscillator`` — Williams' three-window oscillator, window-nulling,
invariant.
"""

import math

from pomata.indicators import ultimate_oscillator
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_ultimate_oscillator
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

_HLC_HIGH = (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5)

_HLC_LOW = (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5)

_HLC_CLOSE = (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.5, 14.0)

ULTIMATE_OSCILLATOR = suite_indicators(
    factory=ultimate_oscillator,
    inputs=("high", "low", "close"),
    params={"window_short": 7, "window_medium": 14, "window_long": 28},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.EXPR,
    warmup_value=27,
    oracle=reference_ultimate_oscillator,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=0),),
    talib=RelationTalib.MATCHES,
    raises=(
        ({"window_short": 0}, r"window_short must be >= 1"),
        ({"window_medium": 0}, r"window_medium must be >= 1"),
        ({"window_long": 0}, r"window_long must be >= 1"),
        (
            {"window_short": 14, "window_medium": 7},
            r"windows must be ordered window_short <= window_medium <= window_long",
        ),
        (
            {"window_medium": 28, "window_long": 14},
            r"windows must be ordered window_short <= window_medium <= window_long",
        ),
    ),
    golden=Golden(
        inputs={"high": _HLC_HIGH, "low": _HLC_LOW, "close": _HLC_CLOSE},
        output=(None, None, None, 60.7143, 66.6667, 65.0433, 67.619, 65.4762, 67.619, 65.4762),
        params={"window_short": 2, "window_medium": 3, "window_long": 4},
    ),
    pins=(
        Pin(
            label="window_all_one_equal_windows_accepted",
            inputs={"high": _HLC_HIGH, "low": _HLC_LOW, "close": _HLC_CLOSE},
            params_override={"window_short": 1, "window_medium": 1, "window_long": 1},
            expected=(
                50.0,
                66.66666666666667,
                66.66666666666667,
                50.0,
                75.0,
                50.0,
                75.0,
                50.0,
                75.0,
                50.0,
            ),
            reason="equal windows are accepted (not raised) and the minimum window=1 is fully defined from row 0",
        ),
        Pin(
            label="flat_window_is_nan",
            inputs={"high": (10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0), "close": (10.0, 10.0, 10.0)},
            params_override={"window_short": 1, "window_medium": 1, "window_long": 2},
            expected=(None, math.nan, math.nan),
            reason="the 0/0 degenerate on a flat well-formed series, detected via residual-free rolling maxima",
        ),
        Pin(
            label="flat_window_is_nan_at_large_magnitude",
            inputs={"high": (1e9, 1e9, 1e9), "low": (1e9, 1e9, 1e9), "close": (1e9, 1e9, 1e9)},
            params_override={"window_short": 1, "window_medium": 1, "window_long": 2},
            expected=(None, math.nan, math.nan),
            reason="the exact-flat guard is residual-free at scale, yielding NaN rather than a falsely-saturated value",
        ),
        Pin(
            label="flat_range_missing_low_is_inf",
            inputs={"high": (10.0, 8.0), "low": (10.0, None), "close": (10.0, 12.0)},
            params_override={"window_short": 1, "window_medium": 1, "window_long": 2},
            expected=(None, math.inf),
            reason="a missing low sends the true range to zero through the prior-close fallback while the buying "
            "pressure stays positive, so the quotient is +/-inf — the infinity beside the 0/0 NaN pin",
        ),
    ),
    reference='Williams, L. (1985). "The Ultimate Oscillator." *Technical Analysis of Stocks & Commodities*.',
    wikipedia="https://en.wikipedia.org/wiki/Ultimate_oscillator",
    see_also=(
        ("rsi", "The single-period momentum oscillator this generalizes across three."),
        ("williams_r", "Another high-low-range momentum oscillator."),
        ("true_range", "The per-bar true range the buying pressure is normalized by."),
    ),
    notes=(
        (
            "Seeding",
            "Row ``0`` has no previous close, so the true low and true high fall back to that bar's own low and high.",
        ),
    ),
    note_extension="\n\n"
    "It is scale-invariant under a positive common rescaling of ``high``, ``low``, and "
    "``close`` (each averaged term is a ratio of price ranges).",
    bullets=(
        (
            "Null",
            "a ``null`` in a single ``high`` / ``low`` / ``close`` drops only the terms that "
            "reference it (the true low / high follow ``pl.min_horizontal`` / ``pl.max_horizontal``, "
            "which skip nulls); a ``null`` reaching a period sum yields ``null`` for the rows whose "
            "window touches it.",
        ),
        (
            "NaN",
            "the per-field behavior is asymmetric. A ``NaN`` in ``high`` or ``close`` propagates "
            "(``pl.max_horizontal`` treats it as the largest value, and a corrupt close poisons the "
            "next bar's true range), yielding ``NaN``. A ``NaN`` in ``low`` on a bar with a finite "
            "previous close is instead treated as absent: ``pl.min_horizontal`` skips it and the true "
            "low falls back to the previous close, so the bar reports a finite value computed from "
            "the substituted close (only at row ``0``, where there is no previous close, does a "
            "``NaN`` ``low`` propagate).",
        ),
        (
            "Degenerate denominator",
            "an exactly-flat true range with zero buying pressure — the genuine degenerate, detected "
            "via the residual-free rolling maxima of the true range and the buying pressure — is "
            "indeterminate, so the result is a ``0 / 0``, i.e. ``NaN``; a finite buying pressure over "
            "an exactly-zero true range (the missing-``low`` fallback) is left to IEEE-754 as "
            "``+/-inf``, and a near-flat range is reported, not clipped (the ``[0, 100]`` bound is "
            "conditional on well-formed bars, so past a sane dynamic range its precision degrades — "
            "see the precision note above).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The Ultimate Oscillator for each row, the same length as the inputs, in ``[0, 100]`` for "
    "well-formed bars. The first ``max(window_short, window_medium, window_long) - 1`` values "
    "are ``null`` (warm-up). The bound is not guaranteed for an incoherent bar: a missing or "
    "``NaN`` ``low`` on a down bar (the documented fallback below) substitutes the previous "
    "``close`` into the true low, which can make the buying pressure negative and push the "
    "value outside ``[0, 100]``.",
    raises_prose="ValueError: If ``window_short < 1``, ``window_medium < 1``, ``window_long < 1``, or the "
    "periods are not ordered ``window_short <= window_medium <= window_long`` (the three "
    "windows must run shortest to longest).",
    args_prose={
        "window_short": "Number of observations in the short averaging window (weight ``4``, canonically ``7``). "
        "Must be ``>= 1``.",
        "window_medium": "Number of observations in the medium averaging window (weight ``2``, canonically "
        "``14``). Must be ``>= 1``.",
        "window_long": "Number of observations in the long averaging window (weight ``1``, canonically ``28``). "
        "Must be ``>= 1``.",
    },
    intro_basic="Basic usage on high-low-close bars:",
    examples=(
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5),
                "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0),
            },
            params={"window_short": 2, "window_medium": 3, "window_long": 4},
            round_to=4,
        ),
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 20.0, 21.0, 22.0, 21.5, 23.0, 22.5),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 19.0, 20.0, 21.0, 20.5, 22.0, 21.5),
                "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, 19.5, 20.5, 21.5, 21.0, 22.5, 22.0),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"),
            params={"window_short": 2, "window_medium": 3, "window_long": 4},
            round_to=4,
        ),
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5, 16.0, 15.5, 17.0),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5, 16.0),
                "close": (9.5, 10.5, 11.5, 11.0, 12.5, 12.0, None, 13.0, 12.5, 13.5, float("nan"), 14.0, 15.0),
            },
            intro="A ``null`` (which nulls the windows that cover it) and a ``NaN`` (which propagates, also "
            "poisoning the next bar's true range) in ``close`` make the handling visible:",
            params={"window_short": 2, "window_medium": 3, "window_long": 4},
            round_to=4,
        ),
        Example(
            inputs={"high": (10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0), "close": (10.0, 10.0, 10.0)},
            intro="**Degenerate denominator** — the ``0/0`` degenerate on a flat well-formed series, "
            "detected via residual-free rolling maxima:",
            params={"window_short": 1, "window_medium": 1, "window_long": 2},
        ),
        Example(
            inputs={"high": (10.0, 8.0), "low": (10.0, None), "close": (10.0, 12.0)},
            intro="**Degenerate denominator** — a missing low sends the true range to zero through the "
            "prior-close fallback while the buying pressure stays positive, so the quotient is "
            "``+/-inf``:",
            params={"window_short": 1, "window_medium": 1, "window_long": 2},
        ),
    ),
)
