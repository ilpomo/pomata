"""
Declaration for ``pomata.indicators.true_range`` — Wilder's single-bar True Range, windowless, absorbing, degree-1.
"""

import math

from pomata.indicators import true_range
from tests_new.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests_new.indicators.harness import suite_indicators
from tests_new.indicators.oracles import reference_true_range
from tests_new.support.declaration import Golden, Pin, ScaleAxis, Shape

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
)
