"""Declaration for ``pomata.metrics.kelly_criterion`` — reducing, the optimal betting fraction, scale-invariant."""

import polars as pl

from pomata.metrics import kelly_criterion, payoff_ratio, win_rate
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_kelly_criterion
from tests.support.declaration import Golden, Pin, ScaleAxis


def _kelly_component() -> pl.Expr:
    """Kelly recomposed from the public ``win_rate`` and ``payoff_ratio`` factories: p - (1 - p) / W."""
    probability = win_rate(pl.col("returns"))
    return probability - (1.0 - probability) / payoff_ratio(pl.col("returns"))


KELLY_CRITERION = suite_metrics(
    factory=kelly_criterion,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_kelly_criterion,
    recomposition=_kelly_component,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    golden=Golden(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, output=(0.1758,)),
    pins=(
        Pin(
            label="single_row_one_sided",
            inputs={"returns": (0.02,)},
            expected=(None,),
            reason="a one-element series is one-sided, so the payoff ratio is undefined and the fraction is null",
        ),
        Pin(
            label="no_losses_is_null",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(None,),
            reason="an all-positive series has an undefined payoff ratio, so the fraction is null ",
        ),
        Pin(
            label="no_wins_is_null",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(None,),
            reason="an all-negative series has an undefined payoff ratio, so the fraction is null ",
        ),
    ),
    reference='Kelly, J. L. (1956). "A New Interpretation of Information Rate." *Bell System Technical '
    "Journal*, 35(4), 917-926.",
    doi="https://doi.org/10.1002/j.1538-7305.1956.tb03809.x",
    wikipedia="https://en.wikipedia.org/wiki/Kelly_criterion",
    see_also=(
        ("win_rate", "The win probability ``p``."),
        ("payoff_ratio", "The average-win to average-loss ratio ``W``."),
        ("risk_of_ruin", "The ruin probability from the same win-rate model."),
    ),
    note_extension="\n\n"
    "This is the **discrete win/loss** form (from the win rate and payoff ratio). A common "
    "alternative for continuous returns is the ratio of the mean return to its variance; the "
    "two coincide only under specific assumptions.",
    bullets=(
        ("Null", "a ``null`` return is skipped; an all-null (or empty) series yields ``null``."),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "a one-element series is one-sided (its payoff ratio is undefined), so the result is ``null``.",
        ),
        (
            "Degenerate denominator",
            "with no winning or no losing returns the payoff ratio is undefined, and with no non-zero "
            "returns the win rate is undefined, so the result is ``null``.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value: the Kelly fraction (one value in ``select``, one per group "
    "under ``.over``). ``null`` when the win rate or the payoff ratio is undefined (no "
    "decisive returns, or one-sided returns).",
    intro_over="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
    intro_missing="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
    "handling visible:",
)
