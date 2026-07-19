"""
Declaration for ``pomata.metrics.max_drawdown_duration`` — reducing, the longest underwater run, scale-invariant.
"""

from pomata.metrics import max_drawdown_duration
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_max_drawdown_duration
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

MAX_DRAWDOWN_DURATION = suite_metrics(
    factory=max_drawdown_duration,
    inputs=("equity_curve",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    oracle=reference_max_drawdown_duration,
    scaling=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden=Golden(inputs={"equity_curve": (1.0, 0.9, 0.8, 0.85, 1.1, 1.05)}, output=(3.0,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a one-element series is never underwater, so the duration is 0 ",
        ),
        Pin(
            label="no_drawdown_is_zero",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(0.0,),
            reason="a monotonically rising curve is never underwater, so the duration is 0 ",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Drawdown_%28economics%29",
    see_also=(
        ("max_drawdown", "The depth dimension (worst decline)."),
        ("drawdown", "The running series whose underwater runs this counts."),
        ("ulcer_index", "Penalizes prolonged declines, blending depth and duration."),
    ),
    note_extension="\n\n"
    "The duration is a count of observations, not a calendar span; with irregular spacing "
    "scale it by the bar period externally.",
    bullets=(
        (
            "Null",
            "a ``null`` equity is skipped; an all-null (or empty) series yields ``null`` (the run is "
            "measured over the retained observations, so a gap neither breaks nor extends the "
            "underwater stretch).",
        ),
        ("NaN", "a ``NaN`` equity propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "a single observation is never underwater, so the result is exactly ``0``, not ``null``.",
        ),
        (
            "Degenerate denominator",
            "a monotonically non-decreasing curve is never underwater, so the result is ``0`` (not a ``0 / 0``).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the longest underwater run length in bars (one value in "
    "``select``, one per group under ``.over``). ``0`` when the curve never goes below a "
    "prior peak; ``null`` when there are no observations.",
    args_prose={
        "equity_curve": "Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.",
    },
    example_columns={"equity_curve": "equity"},
    examples=(
        Example(inputs={"equity_curve": (1.0, 0.9, 0.8, 0.85, 1.1, 1.05)}),
        Example(
            inputs={"equity_curve": (1.0, 0.9, 0.8, 0.85, 1.1, 1.05, 1.2, 1.0, 1.05, 0.95, 0.9, 1.1, 1.0, 1.2)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("AAPL",) * 7 + ("NVDA",) * 7,
            round_to=4,
        ),
        Example(
            inputs={"equity_curve": (1.0, 0.9, None, 0.85, float("nan"), 1.05)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            round_to=4,
        ),
        Example(
            inputs={"equity_curve": (1.0,)},
            intro="**Insufficient sample** — a one-element series is never underwater, so the duration is ``0``:",
        ),
        Example(
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            intro="**Degenerate denominator** — a monotonically rising curve is never underwater, so the "
            "duration is ``0``:",
        ),
    ),
)
