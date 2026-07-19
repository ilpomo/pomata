"""Declaration for ``pomata.metrics.ulcer_index`` — reducing, the RMS drawdown, scale-invariant."""

from pomata.metrics import ulcer_index
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_ulcer_index
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

ULCER_INDEX = suite_metrics(
    factory=ulcer_index,
    inputs=("equity_curve",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    oracle=reference_ulcer_index,
    scaling=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden=Golden(inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2, 0.9, 1.0)}, output=(0.1241,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a one-element series has no drawdown, so the Ulcer Index is 0",
        ),
        Pin(
            label="monotonic_rise_is_zero",
            inputs={"equity_curve": (1.0, 1.1, 1.2, 1.3)},
            expected=(0.0,),
            reason="a never-declining curve has all-zero drawdowns, so the Ulcer Index is exactly 0",
        ),
    ),
    reference="Martin, P. G. & McCann, B. B. (1989). *The Investor's Guide to Fidelity Funds*. Wiley.",
    wikipedia="https://en.wikipedia.org/wiki/Ulcer_index",
    see_also=(
        ("max_drawdown", "The single worst drawdown, which the Ulcer Index complements with a continuous measure."),
        ("ulcer_performance_ratio", "The return-over-Ulcer ratio built on this."),
        ("pain_index", "The arithmetic-mean counterpart of this root-mean-square."),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` equity is skipped; an all-null (or empty) series yields ``null`` (excluded from the mean).",
        ),
        ("NaN", "a ``NaN`` equity propagates, yielding ``NaN``."),
        ("Insufficient sample", "a single observation has no drawdown, so the result is exactly ``0``, not ``null``."),
        (
            "Degenerate denominator",
            "a never-declining curve has all-zero drawdowns, so the result is ``0`` (not a ``0 / 0``).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the Ulcer Index (one value in ``select``, one per group "
    "under ``.over``). It is ``>= 0``; ``null`` when there are no observations.",
    args_prose={
        "equity_curve": "Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.",
    },
    example_columns={"equity_curve": "equity"},
    examples=(
        Example(inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2, 0.9, 1.0)}, round_to=4),
        Example(
            inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2, 0.9, 1.0, 1.1, 1.0, 0.95, 1.05, 1.0, 1.15, 1.1, 1.2)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("AAPL",) * 7 + ("NVDA",) * 7,
            round_to=4,
        ),
        Example(
            inputs={"equity_curve": (1.0, 1.1, None, 1.2, float("nan"), 1.0)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            round_to=4,
        ),
        Example(
            inputs={"equity_curve": (1.0,)},
            intro="**Insufficient sample** — a one-element series has no drawdown, so the Ulcer Index is ``0``:",
        ),
        Example(
            inputs={"equity_curve": (1.0, 1.1, 1.2, 1.3)},
            intro="**Degenerate denominator** — a never-declining curve has all-zero drawdowns, so the "
            "Ulcer Index is exactly ``0``:",
        ),
    ),
)
