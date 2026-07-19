"""Declaration for ``pomata.metrics.risk_of_ruin`` — reducing, the gambler's-ruin probability from the win rate."""

from pomata.metrics import risk_of_ruin
from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.metrics.harness import suite_metrics
from tests.metrics.oracles import reference_risk_of_ruin
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

RISK_OF_RUIN = suite_metrics(
    factory=risk_of_ruin,
    inputs=("returns",),
    params={},
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
    degenerate=Degenerate.RATIO_SIGNED_INF_OR_NAN,
    oracle=reference_risk_of_ruin,
    scaling=(ScaleAxis(roles=("returns",), degree=0),),
    golden=Golden(inputs={"returns": (0.02, -0.01, 0.03, -0.02)}, output=(1.0,)),
    pins=(
        Pin(label="single_row", inputs={"returns": (0.05,)}, expected=(0.0,), reason="a single win (p=1) gives ruin 0"),
        Pin(
            label="all_wins_is_zero",
            inputs={"returns": (0.01, 0.02, 0.03)},
            expected=(0.0,),
            reason="an all-winning series (p=1) has no ruin risk",
        ),
        Pin(
            label="all_losses_is_one",
            inputs={"returns": (-0.01, -0.02, -0.03)},
            expected=(1.0,),
            reason="an all-losing series (p=0) is certain ruin",
        ),
        Pin(
            label="all_zero_is_null",
            inputs={"returns": (0.0, 0.0, 0.0)},
            expected=(None,),
            reason="a series of exact-zero returns has no decisive bars, so the win rate and ruin are null ",
        ),
    ),
    reference="Vince, R. (1990). *Portfolio Management Formulas*. Wiley.",
    wikipedia="https://en.wikipedia.org/wiki/Gambler%27s_ruin",
    see_also=(
        ("win_rate", "The win probability the model is built on."),
        ("kelly_criterion", "The growth-optimal bet fraction from the same inputs."),
        ("payoff_ratio", "The average win/loss size this symmetric model ignores."),
    ),
    note_extension="\n\n"
    "This is the **symmetric** form: it depends only on the win rate and the bet count, "
    "assuming equal-sized wins and losses and ruin at the loss of all capital. It "
    "deliberately ignores win/loss size and capital units. Because the bet count ``n`` "
    "doubles as the capital cushion, the result is sensitive to the series length: more bars "
    "drive it toward ``0`` with an edge and toward ``1`` without one. Compare series of the "
    "same length. The win rate ``p`` counts only decisive (non-zero) bars, while ``n`` counts "
    "every non-null bar, so padding a series with flat ``0`` bars raises ``n`` without moving "
    "``p``.",
    bullets=(
        ("Null", "a ``null`` return is skipped; an all-null (or empty) series yields ``null``."),
        ("NaN", "a ``NaN`` return propagates, yielding ``NaN``."),
        (
            "Insufficient sample",
            "a series with no decisive (non-zero) returns has an undefined win rate, so the result is ``null``.",
        ),
        (
            "Degenerate denominator",
            "the ruin odds ratio is ``(1 - p) / p``: with no edge (``p <= 0.5``) it is ``>= 1`` and "
            "the probability saturates at ``1`` (an all-losing ``p = 0`` divides by zero and clips to "
            "``1``), while an all-winning ``p = 1`` gives ``0``.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A single ``Float64`` value in ``[0, 1]``: the risk of ruin (one value in ``select``, one "
    "per group under ``.over``). ``null`` when there are no decisive (non-zero) returns (the "
    "win rate is undefined).",
    examples=(
        Example(inputs={"returns": (0.02, -0.01, 0.03, -0.02)}, round_to=4),
        Example(
            inputs={"returns": (0.02, -0.01, 0.03, -0.02, 0.02, 0.01, 0.03, -0.02)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker is reduced "
            "independently (here ``A`` has no edge, so its ruin is certain, while the winning ``B`` "
            "is small):",
            partition=("A", "A", "A", "A", "B", "B", "B", "B"),
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.02, None, -0.01, 0.03, float("nan"), -0.02)},
            intro="A ``null`` (skipped) and a ``NaN`` (which poisons the result) make the missing-data "
            "handling visible:",
            round_to=4,
        ),
        Example(
            inputs={"returns": (0.05,)}, intro="**Insufficient sample** — a single win (``p = 1``) gives ruin ``0``:"
        ),
        Example(
            inputs={"returns": (0.01, 0.02, 0.03)},
            intro="**Degenerate denominator** — an all-winning series (``p = 1``) has no ruin risk, so the "
            "probability is ``0``:",
        ),
        Example(
            inputs={"returns": (0.0, 0.0, 0.0)},
            intro="**Degenerate denominator** — a series of exact-zero returns has no decisive bars, so the "
            "win rate and ruin are ``null``:",
        ),
    ),
)
