"""Declaration for ``pomata.metrics.max_drawdown`` — reducing, the deepest peak-to-trough decline, scale-invariant."""

from pomata.metrics import max_drawdown
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_max_drawdown
from tests.support.declaration import Golden, Pin, ScaleAxis

MAX_DRAWDOWN = suite_metrics(
    factory=max_drawdown,
    inputs=("equity_curve",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    oracle=reference_max_drawdown,
    scaling=(ScaleAxis(roles=("equity_curve",), degree=0),),
    golden=Golden(inputs={"equity_curve": (1.0, 1.1, 1.05, 1.2, 0.9, 1.0)}, output=(-0.25,)),
    pins=(
        Pin(
            label="single_row_is_zero",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a one-element series is at its own peak, so the maximum drawdown is 0",
        ),
        Pin(
            label="monotonic_rise_is_zero",
            inputs={"equity_curve": (1.0, 1.1, 1.2, 1.3)},
            expected=(0.0,),
            reason="a never-declining curve has zero drawdown",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Drawdown_%28economics%29",
    see_also=(
        ("drawdown", "The running series this reduces."),
        ("calmar_ratio", "The return-over-drawdown ratio built on this."),
        ("max_drawdown_duration", "The duration dimension (longest underwater stretch)."),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` equity is skipped; an all-null (or empty) series yields ``null`` (a missing "
            "bar does not start a drawdown).",
        ),
        (
            "NaN",
            "a ``NaN`` equity propagates, yielding ``NaN`` (an undefined equity makes the "
            "worst-drawdown summary undefined).",
        ),
        (
            "Insufficient sample",
            "a single observation is trivially at its own peak, so the result is exactly ``0``, not ``null``.",
        ),
        (
            "Degenerate denominator",
            "a never-declining curve has zero drawdown throughout, so the result is ``0`` (not a ``0 / 0``).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the maximum drawdown (one value in ``select``, one per group "
    "under ``.over``). It is ``<= 0`` (``0`` for a never-declining curve); ``null`` when "
    "there are no observations.",
    args_prose={
        "equity_curve": "Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.",
    },
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
    intro_missing="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
    "handling visible:",
)
