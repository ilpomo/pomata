"""Declaration for ``pomata.metrics.total_return`` — reducing, the final growth factor minus one, scale-exempt."""

from pomata.metrics import total_return
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_total_return
from tests.support.declaration import Example, Golden, Pin, ScaleExempt

TOTAL_RETURN = suite_metrics(
    factory=total_return,
    inputs=("equity_curve",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    oracle=reference_total_return,
    scaling=ScaleExempt(
        reason="a growth-factor series normalized to a unit start (the result is the final value minus "
        "one), neither scale-homogeneous nor scale-invariant"
    ),
    golden=Golden(inputs={"equity_curve": (1.1, 1.045, 1.254, 1.3794)}, output=(0.3794,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"equity_curve": (1.21,)},
            expected=(0.21,),
            reason="a one-element series resolves to the final growth minus one ",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Total_return",
    see_also=(
        ("cagr", "The annualized (per-year) form of this total growth."),
        ("total_return_rolling", "The windowed twin, over each trailing window."),
        ("equity_curve", "The pnl builder that produces the input curve."),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` equity is skipped; an all-null (or empty) series yields ``null`` — the result "
            "uses the last defined equity.",
        ),
        ("NaN", "a ``NaN`` equity propagates, yielding ``NaN``."),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the total compounded return (one value in ``select``, one "
    "per group under ``.over``). ``null`` when there are no observations.",
    args_prose={
        "equity_curve": "Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive; "
        "its ``N`` values are ``N`` period growth factors, and its final value is the total "
        "growth multiple.",
    },
    example_columns={"equity_curve": "equity"},
    examples=(
        Example(inputs={"equity_curve": (1.1, 1.045, 1.254, 1.3794)}, round_to=4),
        Example(
            inputs={"equity_curve": (1.1, 1.045, 1.254, 1.3794, 1.02, 1.05, 0.98, 1.12)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("AAPL",) * 4 + ("NVDA",) * 4,
            round_to=4,
        ),
        Example(
            inputs={"equity_curve": (1.1, 1.045, None, 1.254, float("nan"), 1.3794)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            round_to=4,
        ),
    ),
)
