"""Declaration for ``pomata.metrics.conditional_drawdown_at_risk`` — reducing, the mean of the worst drawdowns."""

from pomata.metrics import conditional_drawdown_at_risk
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_conditional_drawdown_at_risk
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

CONDITIONAL_DRAWDOWN_AT_RISK = suite_metrics(
    factory=conditional_drawdown_at_risk,
    inputs=("equity_curve",),
    params={"confidence": 0.95},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    oracle=reference_conditional_drawdown_at_risk,
    scaling=(ScaleAxis(roles=("equity_curve",), degree=0),),
    raises=(
        ({"confidence": 0.0}, r"confidence must be in the open interval"),
        ({"confidence": 1.0}, r"confidence must be in the open interval"),
        ({"confidence": -0.1}, r"confidence must be in the open interval"),
        ({"confidence": 1.5}, r"confidence must be in the open interval"),
    ),
    golden=Golden(inputs={"equity_curve": (1.1, 1.05, 1.2, 1.15, 1.3, 1.25, 1.4)}, output=(-0.0455,)),
    pins=(
        Pin(
            label="single_row",
            inputs={"equity_curve": (1.0,)},
            expected=(0.0,),
            reason="a one-element series is at its own peak, so CDaR is exactly 0",
        ),
        Pin(
            label="no_drawdown_is_zero",
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            expected=(0.0,),
            reason="a monotonically rising curve has an all-zero drawdown series, so CDaR is 0",
        ),
        Pin(
            label="fractional_weight_golden",
            inputs={"equity_curve": (1.0, 0.8, 1.0, 0.9, 0.7, 1.0)},
            expected=(-0.26666666666666666,),
            reason="the Rockafellar-Uryasev fractional boundary-weight case at confidence=0.75 (worst "
            "averaged in full, second-worst at weight 0.5)",
            params_override={"confidence": 0.75},
        ),
    ),
    reference='Chekhlov, A., Uryasev, S. & Zabarankin, M. (2005). "Drawdown Measure in Portfolio '
    'Optimization." *International Journal of Theoretical and Applied Finance*, 8(1), 13-58.',
    doi="https://doi.org/10.1142/S0219024905002767",
    wikipedia="https://en.wikipedia.org/wiki/Drawdown_%28economics%29",
    see_also=(
        ("max_drawdown", "The single worst drawdown."),
        ("conditional_value_at_risk", "The return-space analog (expected shortfall)."),
        ("pain_index", "The full-sample mean drawdown, against this worst-tail mean."),
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
            "a monotonically non-decreasing curve has an all-zero drawdown series, so the result is "
            "``0`` (not a ``0 / 0``).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the conditional drawdown at risk (one value in ``select``, "
    "one per group under ``.over``). ``null`` when there are no observations.",
    raises_prose="ValueError: If ``confidence`` is not in the open interval ``(0, 1)``.",
    args_prose={
        "equity_curve": "Compounded growth-factor series (e.g. from :func:`~pomata.pnl.equity_curve`), positive.",
        "confidence": "The tail confidence level (canonically ``0.95``); the mean is taken over the worst ``1 - "
        "confidence`` of drawdowns. Must be in the open interval ``(0, 1)``.",
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
            intro="**Insufficient sample** — a one-element series is at its own peak, so CDaR is exactly ``0``:",
        ),
        Example(
            inputs={"equity_curve": (1.0, 1.1, 1.21)},
            intro="**Degenerate denominator** — a monotonically rising curve has an all-zero drawdown "
            "series, so CDaR is ``0``:",
        ),
    ),
)
