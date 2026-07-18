"""
Declaration for ``pomata.indicators.dm_plus`` — Wilder's smoothed positive directional movement, gap-bridging,
degree-1.
"""

import math

from pomata.indicators import dm_plus
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_dm_plus
from tests.support.declaration import Deviant, Golden, Pin, ScaleAxis, Shape

DM_PLUS = suite_indicators(
    factory=dm_plus,
    inputs=("high", "low"),
    params={"window": 2},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW_MINUS_ONE,
    oracle=reference_dm_plus,
    scaling=(ScaleAxis(roles=("high", "low"), degree=1),),
    talib=RelationTalib.MATCHES,
    raises=(({"window": 0}, r"window must be >= 1"),),
    flow_deviation="the up-move guard turns a fully-missing bar into 0 movement, so a full-bar null / NaN is absorbed "
    "and the rma recurrence continues at 0 (never a null trace or a latch), while a single-column NaN on the high leg "
    "still latches — one shared policy cannot hold both; pinned below and covered by the missing-data property tier",
    golden=Golden(
        inputs={
            "high": (10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0),
            "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0),
        },
        output=(None, 0.5, 0.75, 0.375, 0.9375, 0.4688, 0.9844),
    ),
    deviant=Deviant(
        expected=(None, 0.0, 0.0, 0.0),
        reason="a null high or low leaves the raw +DM at 0 (the up-move guard is unsatisfied), which the Wilder rma "
        "smooths to 0 past the one-row warm-up",
    ),
    pins=(
        Pin(
            label="interior_null_bridged",
            inputs={
                "high": (10.0, 11.0, 12.0, None, 13.0, 13.5, 14.0, 13.5),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5),
            },
            expected=(None, 0.5, 0.75, 0.375, 0.1875, 0.09375, 0.296875, 0.1484375),
            reason="an interior null makes the raw +DM 0 there and the rma carries its state across the gap (no null "
            "trace)",
        ),
        Pin(
            label="nan_on_high_leg_latches",
            inputs={
                "high": (10.0, 11.0, 12.0, 12.5, 13.0, math.nan, 14.0, 13.5),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5),
            },
            expected=(None, 0.5, 0.75, 0.375, 0.4375, math.nan, math.nan, math.nan),
            reason="a NaN on the high leg (the up-move driver) poisons the raw +DM and latches the rma forever",
        ),
    ),
    reference="Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.",
    wikipedia="https://en.wikipedia.org/wiki/Average_directional_movement_index",
    see_also=(
        ("dm_minus", "The minus counterpart."),
        ("di_plus", "The plus directional indicator built from this and the :func:`atr`."),
        ("rma", "The Wilder moving average that smooths the raw movement."),
    ),
    notes=(
        (
            "Seeding",
            "Row ``0`` has no previous bar, so its raw movement is ``0`` and seeds the smoothing. The "
            "raw directional movement is then smoothed by Wilder's :func:`rma`, the mean-scale "
            "recursion ``m_t = m_{t-1} - m_{t-1} / window + raw_t / window`` (smoothing factor ``1 / "
            "window``). Wilder's original presentation instead smooths on the sum scale (``S_t = "
            "S_{t-1} - S_{t-1} / window + raw_t``, seeded from a simple sum of the first ``window`` "
            "raw movements), which equals ``window`` times the mean-scale value in steady state. That "
            "factor of ``window`` is structural and persists for every row — it is not a warm-up seed "
            "difference that washes out — so this series reads roughly ``window`` times smaller than "
            "the sum-scale convention throughout. The factor cancels in :func:`di_plus`, :func:`dx`, "
            "and :func:`adx`, which are therefore unaffected.",
        ),
    ),
    note_extension="\n\n"
    "It is homogeneous of degree ``1`` in a positive common rescaling of ``high`` and ``low`` "
    "(a range expansion in price units).",
    bullets=(
        (
            "Null",
            "a ``null`` in ``high`` or ``low`` makes the affected raw movement ``0`` for the rows "
            "whose difference it touches, so the raw movement carries no interior nulls and the only "
            "nulls emitted are the ``window - 1`` warm-up nulls from :func:`rma`.",
        ),
        (
            "NaN",
            "a ``NaN`` in ``high`` (the own-side input) poisons the recursion and yields ``NaN`` for "
            "every subsequent non-null row (except at ``window == 1``, where the smoothing is the "
            "identity and nothing latches: the ``NaN`` clears once it leaves the differencing's "
            "one-bar reach); a ``NaN`` in ``low`` (the opposing side) instead makes the directional "
            "comparison false, so the affected raw movement is sent to ``0`` and genuine upward "
            "movement is silently dropped there.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The smoothed plus directional movement for each row, the same length as the inputs. The "
    "first ``window - 1`` values are ``null`` (warm-up), inherited from the :func:`rma`.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the Wilder moving window. Must be ``>= 1``.",
    },
    intro_basic="On a small high/low frame with a short window:",
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
    intro_missing="A leading ``null`` ``high`` (which zeroes the raw movement it touches) and a later "
    "``NaN`` ``high`` (the own side, which poisons the recursion) make the handling visible:",
)
