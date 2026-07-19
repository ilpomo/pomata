"""Declaration for ``pomata.metrics.pain_index`` — reducing, the mean absolute drawdown, scale-invariant."""

from pomata.metrics import pain_index
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_pain_index
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

PAIN_INDEX = suite_metrics(
    factory=pain_index,
    inputs=("equity_curve",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    oracle=reference_pain_index,
    scaling=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden=Golden(inputs={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)}, output=(0.0179,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a one-element series is at its own peak, so the pain index is exactly 0 ",
        ),
        Pin(
            label="no_drawdown_is_zero",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(0.0,),
            reason="a monotonically rising curve is never below its running peak, so the mean drawdown is 0 ",
        ),
    ),
    reference='Becker, T. (2006). "The Pain Index and Pain Ratio." *Zephyr Associates*.',
    see_also=(
        ("ulcer_index", "The root-mean-square counterpart."),
        ("pain_ratio", "The return-to-pain ratio built on this."),
        ("max_drawdown", "The single worst drawdown, against this average depth."),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` equity is skipped; an all-null (or empty) series yields ``null`` (the running "
            "peak carries across it).",
        ),
        ("NaN", "a ``NaN`` equity propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "a single observation is trivially at its own peak, so the result is exactly ``0``, not ``null``.",
        ),
        (
            "Degenerate denominator",
            "a monotonically non-decreasing curve is never below its peak, so the result is ``0`` (not a ``0 / 0``).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the pain index (one value in ``select``, one per group under "
    "``.over``). It is ``>= 0``; ``null`` when there are no observations.",
    args_prose={
        "equity_curve": "Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.",
    },
    example_columns={"equity_curve": "equity"},
    examples=(
        Example(inputs={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)}, round_to=4),
        Example(
            inputs={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4, 1.0, 0.9, 0.95, 1.1, 1.0, 1.2, 1.15)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("A", "A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B", "B"),
            round_to=4,
        ),
        Example(
            inputs={"equity_curve": (1.1, 1.05, None, 1.2, float("nan"), 1.15, 1.3)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            round_to=4,
        ),
        Example(
            inputs={"equity_curve": (1.0,)},
            intro="**Insufficient sample** — a one-element series is at its own peak, so the pain index is "
            "exactly ``0``:",
        ),
        Example(
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            intro="**Degenerate denominator** — a monotonically rising curve is never below its running "
            "peak, so the mean drawdown is ``0``:",
        ),
    ),
)
