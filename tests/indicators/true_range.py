"""
Declaration for ``pomata.indicators.true_range`` — Wilder's single-bar True Range, windowless, absorbing, degree-1.
"""

import math

from pomata.indicators import true_range
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_true_range
from tests.support.declaration import Golden, Pin, ScaleAxis, Shape

TRUE_RANGE = suite_indicators(
    factory=true_range,
    inputs=("high", "low", "close"),
    params={},
    null=BehaviorNull.ABSORBED,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.NONE,
    oracle=reference_true_range,
    scaling=(ScaleAxis(roles=("high", "low", "close"), degree=1),),
    talib=RelationTalib.MATCHES,
    golden=Golden(
        inputs={
            "high": (10.0, 12.0, 11.5, 13.0, 12.5),
            "low": (9.0, 10.5, 10.0, 11.0, 11.5),
            "close": (9.5, 11.0, 10.5, 12.5, 12.0),
        },
        output=(1.0, 2.5, 1.5, 2.5, 1.0),
    ),
    pins=(
        Pin(
            label="null_high_drops_its_candidates",
            inputs={"high": (10.0, None, 11.0), "low": (9.0, 10.5, 10.0), "close": (9.5, 11.0, 10.5)},
            expected=(1.0, 1.0, 1.0),
            reason="a null high drops the two candidate terms it appears in, so the row resolves from the survivor "
            "|low - prev_close|",
        ),
        Pin(
            label="null_previous_close_falls_back_to_range",
            inputs={"high": (10.0, 12.0, 11.0, 13.0), "low": (9.0, 10.5, 10.0, 11.0), "close": (9.5, None, 10.5, 12.0)},
            expected=(1.0, 2.5, 1.0, 2.5),
            reason="a null previous close drops the two gap terms and the row falls back to high - low",
        ),
        Pin(
            label="nan_close_poisons_next_row_only",
            inputs={
                "high": (10.0, 12.0, 11.0, 13.0),
                "low": (9.0, 10.5, 10.0, 11.0),
                "close": (9.5, math.nan, 10.5, 12.0),
            },
            expected=(1.0, 2.5, math.nan, 2.5),
            reason="a NaN close is finite at its own row (high - low) but poisons the next row's gap terms to NaN, "
            "then recovers",
        ),
        Pin(
            label="high_equals_low_gap_terms_drive_range",
            inputs={"high": (10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0), "close": (10.0, 8.0, 12.0)},
            expected=(0.0, 0.0, 2.0),
            reason="a zero bar spread (high == low) leaves the gap-to-previous-close terms to drive the range: the "
            "row-2 gap |10 - 8| = 2 surfaces",
        ),
    ),
    reference="Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.",
    wikipedia="https://en.wikipedia.org/wiki/Average_true_range",
    see_also=(
        ("atr", "The Wilder-smoothed average of this per-bar range."),
        ("atr_normalized", "That average expressed as a percent of the current close."),
        ("vortex", "A directional indicator that normalizes its movement by this range."),
    ),
    notes=(
        (
            "Inputs",
            "``high``, ``low``, and ``close`` are taken as the canonical OHLC roles in that "
            "positional order and must share a length and alignment (the same row index is one bar).",
        ),
    ),
    bullets=(
        (
            "Null",
            "``null`` handling follows ``pl.max_horizontal``, which **skips** ``null`` candidates "
            "rather than propagating them: a ``null`` in ``high`` or ``low`` (or a ``null`` previous "
            "``close``) simply drops that candidate, so the row still resolves from whichever "
            "distances remain. The result is ``null`` only when all three candidates are ``null``: "
            "with a defined previous ``close`` that means ``high`` and ``low`` are both ``null`` at "
            "the row, but where the previous ``close`` is itself ``null`` (row ``0``, or any bar "
            "after a ``null`` close) the two gap distances are already ``null``, so a single ``null`` "
            "in ``high`` or ``low`` voids the row on its own.",
        ),
        (
            "NaN",
            "a ``NaN`` price yields ``NaN`` for that row — it is not skipped like a ``null`` (it "
            "dominates the maximum), so a ``NaN`` ``close`` also contaminates the two gap terms of "
            "the next row.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The True Range for each row, the same length as the inputs. There is no window and no "
    "warm-up -- every row is defined from row ``0``, which falls back to ``high - low`` "
    "because no previous close exists. On well-formed OHLC data (``high >= low``) every value "
    "is non-negative.",
    args_prose={
        "close": 'Close-price series (e.g. ``pl.col("close")``); the previous close supplies the two gap terms.',
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
    intro_missing="A ``null`` ``close`` (skipped, so the next bar falls back to ``high - low``) then a "
    "``NaN`` ``close`` (which contaminates only the following bar's gap terms) make the exact "
    "handling visible at a glance:",
)
