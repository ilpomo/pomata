"""Declaration for ``pomata.pnl.returns_gross`` — the weight-times-asset-return product, propagating, degree-1."""

import math

from pomata.pnl import returns_gross
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_returns_gross
from tests.support.declaration import Golden, Pin, ScaleAxis

RETURNS_GROSS = suite_pnl(
    factory=returns_gross,
    inputs=("weight", "asset_returns"),
    params={},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.RETURNS,
    sign=ConventionSign.LONG_SHORT,
    oracle=reference_returns_gross,
    # Degree-1 homogeneous in the weight; only the weight axis is exercised.
    scaling=(ScaleAxis(roles=("weight",), degree=1),),
    golden=Golden(
        inputs={
            "weight": (1.0, 0.5, -1.0, -1.0, 0.5),
            "asset_returns": (0.02, -0.01, 0.03, -0.02, 0.04),
        },
        output=(0.02, -0.005, -0.03, 0.02, 0.02),
    ),
    pins=(
        Pin(
            label="single_row",
            inputs={"weight": (0.5,), "asset_returns": (0.04,)},
            expected=(0.02,),
            reason="a one-row series resolves to the single product 0.5 * 0.04 = 0.02",
        ),
        Pin(
            label="null_takes_precedence_over_nan",
            inputs={"weight": (None, 0.5), "asset_returns": (math.nan, 0.04)},
            expected=(None, 0.02),
            reason="a null in one input against a NaN in the other at the same row yields null (null wins over NaN)",
        ),
        Pin(
            label="infinite_legs_sign_the_return",
            inputs={"weight": (math.inf, -1.0, -math.inf), "asset_returns": (0.1, math.inf, -0.2)},
            expected=(math.inf, -math.inf, math.inf),
            reason="the gross return keeps the sign of weight * asset_returns even at infinite magnitude; the "
            "property tiers set allow_infinity=False",
        ),
    ),
    reference='Meucci, A. (2010). "Quant Nugget 2: Linear vs. Compounded Returns." *GARP Risk '
    "Professional*, April 2010, 49-51.",
    wikipedia="https://en.wikipedia.org/wiki/Rate_of_return",
    see_also=(
        ("returns_simple", "The usual source of ``asset_returns``."),
        ("turnover", "The traded fraction of the same ``weight``, the basis for transaction costs."),
        ("equity_curve", "Compounds these per-bar returns into a capital curve."),
    ),
    notes=(
        (
            "No lookahead (alignment is the caller's)",
            "The product assumes ``weight`` at row ``t`` is the weight held over ``asset_returns`` at "
            "row ``t``. To stay lookahead-free, that weight must depend only on information available "
            "before that return; if it is decided on the same bar's close, lag it by one bar "
            "(``returns_gross(weight.shift(1), asset_returns)``). Nothing is shifted for you, so a "
            "weight you have already aligned is never double-shifted.",
        ),
    ),
    bullets=(
        ("Null", "a ``null`` weight makes that row ``null`` (``null`` takes precedence over ``NaN``)."),
        ("NaN", "a ``NaN`` weight yields ``NaN`` for that row."),
        (
            "Non-finite input",
            "an ``inf`` weight follows IEEE-754 through the arithmetic, so the return signs with "
            "``weight * asset_returns`` (the sign included).",
        ),
        (
            "Partitioning",
            "already correct on a multi-series panel: ``.over(...)`` partitions identically and is "
            "therefore optional here.",
        ),
    ),
    returns_body="The gross strategy return for each row, the same length as the inputs. There is no "
    "window and no warm-up of its own: every row is the product of its own ``weight`` and "
    "``asset_returns`` (so a warm-up ``null`` is inherited only from the inputs, e.g. the "
    "first row of :func:`returns_simple`).",
    args_prose={
        "asset_returns": "Per-bar asset returns, typically from :func:`returns_simple` (e.g. "
        '``returns_simple(pl.col("close"))``).',
    },
    intro_basic="Basic usage on a weight and an asset-return series:",
    intro_over="The product is elementwise, so ``.over`` partitions identically and is shown only for consistency:",
    intro_missing="A ``null`` then a ``NaN`` in ``asset_returns`` (both propagate through the product) make "
    "the missing-data handling visible:",
)
