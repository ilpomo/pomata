"""
Declaration for ``pomata.metrics.win_rate`` — reducing, the fraction of decisive returns that are gains, scale-
invariant.
"""

from pomata.metrics import win_rate
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_win_rate
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

WIN_RATE = suite_metrics(
    factory=win_rate,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.COLLAPSES,
    oracle=reference_win_rate,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    golden=Golden(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, output=(0.5714,)),
    pins=(
        Pin(
            label="single_row_positive_win",
            inputs={"returns": (0.05,)},
            expected=(1.0,),
            reason="one positive decisive return gives a rate of 1",
        ),
        Pin(
            label="all_positive_is_one",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(1.0,),
            reason="an all-positive series wins every decisive period",
        ),
        Pin(
            label="all_negative_is_zero",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(0.0,),
            reason="an all-negative series wins no decisive period",
        ),
        Pin(
            label="all_zero_is_null",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(None,),
            reason="an all-zero series has no decisive returns, so the rate is null ",
        ),
        Pin(
            label="zero_excluded_from_denominator",
            inputs={"returns": (0.01, 0.0, 0.02)},
            expected=(1.0,),
            reason="an exact-zero return is excluded from the denominator ",
        ),
    ),
    reference="Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies* (2nd ed.). Wiley.",
    see_also=(
        ("payoff_ratio", "The average size of a win versus a loss."),
        ("profit_factor", "The aggregate gain-to-loss ratio."),
        ("kelly_criterion", "The growth-optimal bet fraction built on this rate."),
    ),
    notes=(
        ("Zero return", "A return of exactly ``0`` is neither a win nor a loss and is excluded from the denominator."),
    ),
    note_extension="\n\n"
    "This is a **bar-level** statistic: each return observation is treated as one win or "
    "loss. It is not a per-trade statistic -- true per-trade win rate needs trade-level fill "
    "data, which is outside this toolkit's scope.",
    bullets=(
        ("Null", "a ``null`` return is skipped; an all-null (or empty) series yields ``null``."),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Degenerate denominator",
            "a series with no decisive (non-zero) returns — an all-``0`` series, say — has an empty "
            "denominator, so the result is ``null``.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value in ``[0, 1]``: the win rate (one value in ``select``, one per "
    "group under ``.over``). ``null`` when there are no decisive (non-zero) returns.",
    examples=(
        Example(inputs={"returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02)}, round_to=4),
        Example(
            inputs={
                "returns": (0.03, -0.01, 0.02, -0.015, 0.01, 0.005, -0.02, 0.04, -0.02, 0.03, 0.01, 0.02, 0.01, -0.03)
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced independently:",
            partition=("AAPL",) * 7 + ("NVDA",) * 7,
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.03, None, -0.01, 0.02, float("nan"), -0.015, 0.01)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            round_to=4,
        ),
        Example(
            inputs={"returns": (-0.01, -0.02, -0.03)},
            intro="**Degenerate denominator** — an all-negative series wins no decisive period, so the rate is ``0``:",
        ),
        Example(
            inputs={"returns": (0.0, 0.0, 0.0)},
            intro="**Degenerate denominator** — an all-zero series has no decisive returns, so the rate is ``null``:",
        ),
    ),
)
