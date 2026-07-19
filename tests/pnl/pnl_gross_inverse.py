"""Declaration for ``pomata.pnl.pnl_gross_inverse`` — coin-settled inverse-contract PnL, one-bar lag, homogeneous."""

import math

from pomata.pnl import pnl_gross_inverse
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost, Warmup
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_pnl_gross_inverse
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

PNL_GROSS_INVERSE = suite_pnl(
    factory=pnl_gross_inverse,
    inputs=("quantity", "price"),
    params={"multiplier": 1.0},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.CASH,
    sign=ConventionSign.LONG_SHORT,
    warmup=Warmup.ONE_ROW,
    oracle=reference_pnl_gross_inverse,
    # Degree-1 homogeneous in quantity, degree-(-1) homogeneous in price (the reciprocal payoff).
    scaling=(
        ScaleAxis(roles=("quantity",), degree=1),
        ScaleAxis(roles=("price",), degree=-1),
    ),
    raises=(
        ({"multiplier": 0.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -5.0}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.nan}, r"multiplier must be a finite number > 0"),
        ({"multiplier": math.inf}, r"multiplier must be a finite number > 0"),
        ({"multiplier": -math.inf}, r"multiplier must be a finite number > 0"),
    ),
    golden=Golden(
        inputs={
            "quantity": (1.0, 1.0, -2.0, -2.0, 3.0),
            "price": (100.0, 110.0, 105.0, 120.0, 115.0),
        },
        output=(None, 0.000909, 0.000866, -0.002381, -0.001087),
        round_to=6,
    ),
    pins=(
        Pin(
            label="null_quantity_nan_price",
            inputs={"quantity": (1.0, None, 2.0), "price": (100.0, math.nan, 110.0)},
            expected=(None, None, math.nan),
            reason="a null quantity against a NaN price yields null (null wins); the NaN previous price then "
            "propagates to the next bar",
        ),
        Pin(
            label="nan_quantity_null_price",
            inputs={"quantity": (1.0, math.nan, 2.0), "price": (100.0, None, 110.0)},
            expected=(None, None, None),
            reason="the reverse direction: a NaN quantity against a null price yields null, and the null previous "
            "price also nulls the next bar",
        ),
        Pin(
            label="short_flat_price_signed_zero",
            inputs={"quantity": (-5.0, -5.0), "price": (100.0, 100.0)},
            expected=(None, -0.0),
            reason="a short over a flat price yields IEEE -0.0 (the reciprocal change is an exact +0.0, and a negative "
            "quantity carries the sign bit)",
            signed=True,
        ),
        Pin(
            label="domain_boundaries",
            inputs={"quantity": (1.0, 1.0, 1.0, 1.0), "price": (100.0, 0.0, 50.0, -50.0)},
            expected=(None, -math.inf, math.inf, 0.04),
            reason="the IEEE reciprocal boundaries pinned as data: a zero current price makes the bar -inf, a zero "
            "previous price makes the next bar +inf, a negative price stays finite",
        ),
        Pin(
            label="notional_multiplier_100",
            inputs={
                "quantity": (1.0, 1.0, -2.0, -2.0, 3.0),
                "price": (100.0, 110.0, 105.0, 120.0, 115.0),
            },
            expected=(
                None,
                0.09090909090909097,
                0.08658008658008684,
                -0.2380952380952383,
                -0.10869565217391311,
            ),
            reason="a 100x inverse-contract notional golden, at full precision (a pin has no "
            "rounding step), also subsuming the multiplier-scaling property",
            params_override={"multiplier": 100.0},
        ),
        Pin(
            label="infinite_legs_propagate",
            inputs={"quantity": (1.0, -math.inf, 2.0), "price": (math.inf, 20.0, 5.0)},
            expected=(None, math.inf, -0.30000000000000004),
            reason="an infinite entry price contributes 1 / inf = 0 to the inverse difference and an infinite short "
            "quantity drives the payoff to +inf, while the finite tail row is unaffected; the property tiers set "
            "allow_infinity=False",
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Perpetual_futures",
    see_also=(
        (
            "pnl_gross",
            "The linear (quote-margined) counterpart; use it when the contract settles in the quote "
            "currency rather than the base coin.",
        ),
        ("pnl_net", "Subtracts the composed cost from this gross PnL."),
        (
            "cost_funding",
            "The perpetual-swap funding leg — beware the units: it computes the **quote-margined** "
            "(linear) funding ``q * P * f``, while this gross PnL is in the base coin, so the "
            "coin-margined funding of an inverse contract must be built directly (e.g. ``quantity * "
            "multiplier / price * funding_rate``) before the two are composed.",
        ),
    ),
    notes=(
        (
            "No lookahead (alignment is the caller's)",
            "The PnL assumes ``quantity`` at row ``t`` is the position held over the price change "
            "into row ``t``. To stay lookahead-free, that quantity must depend only on information "
            "available before that price; if it is decided on the same bar's close, lag it by one bar "
            "(``pnl_gross_inverse(quantity.shift(1), price)``). Nothing is shifted for you, so a "
            "quantity you have already aligned is never double-shifted.",
        ),
    ),
    bullets=(
        (
            "Null",
            "a ``null`` quantity makes that row ``null`` (``null`` takes precedence over ``NaN``) — a "
            "``null`` ``price`` also nulls the next bar, as the previous ``price`` there.",
        ),
        (
            "NaN",
            "a ``NaN`` quantity yields ``NaN`` for that row — a ``NaN`` ``price`` also poisons the "
            "next bar, as the previous ``price`` there.",
        ),
        (
            "Domain",
            "the reciprocal payoff is defined on strictly positive prices; a negative price is out of "
            "that domain yet is not rejected: the reciprocal flips sign, so the bar is a finite but "
            "economically meaningless value. A zero current price makes ``1 / P_t`` infinite, so the "
            "bar is ``-inf`` (a long) or ``+inf`` (a short), and a zero previous price makes the next "
            "bar ``+/-inf`` — reported, not clipped.",
        ),
        (
            "Non-finite input",
            "an ``inf`` quantity follows IEEE-754 through the arithmetic, where an infinite ``price`` "
            "contributes ``1 / inf = 0`` to the reciprocal change (the sign, and any ``inf - inf = "
            "NaN``, included).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The gross PnL for each row, the same length as the inputs, in the base coin. The first "
    "value is ``null`` (warm-up): the previous price ``price.shift(1)`` is undefined for the "
    "first row, so no price change can be measured there.",
    raises_prose="ValueError: If ``multiplier`` is not a finite number ``> 0`` (i.e. ``<= 0``, ``NaN``, or ``±inf``).",
    args_prose={
        "quantity": "Signed position size in units / shares / contracts held over the bar (e.g. ``100``, ``-2``).",
        "price": "Instrument price series, the quote per base unit (e.g. USD per BTC, "
        '``pl.col("close")``); must be strictly positive (see the **Domain** note) and share a '
        "length and alignment with ``quantity``.",
        "multiplier": "Contract notional in the quote currency — the quote value of one contract (e.g. ``1`` "
        "USD for an inverse BTC/USD perpetual, ``100`` on some venues); ``1.0`` for a one-unit "
        "contract. Must be a finite number ``> 0``.",
    },
    intro_basic="Basic usage on an inverse (coin-margined) contract:",
    examples=(
        Example(
            inputs={
                "quantity": (1.0, 1.0, -2.0, -2.0, 3.0, 3.0, -1.0, -1.0),
                "price": (100.0, 110.0, 105.0, 120.0, 115.0, 118.0, 112.0, 120.0),
            },
            round_to=6,
        ),
        Example(
            inputs={
                "quantity": (1.0, 1.0, -2.0, -2.0, 2.0, 2.0, 2.0, 2.0),
                "price": (100.0, 110.0, 105.0, 120.0, 50.0, 55.0, 52.0, 58.0),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("AAPL",) * 4 + ("NVDA",) * 4,
            round_to=6,
        ),
        Example(
            inputs={"quantity": (1.0, None, -2.0, float("nan"), 3.0), "price": (100.0, 110.0, 105.0, 120.0, 115.0)},
            intro="A leading warm-up ``null`` (row 0, no prior price), then a ``null`` and a ``NaN`` in "
            "``quantity`` that void only their own rows:",
            round_to=6,
        ),
        Example(
            inputs={"quantity": (1.0, 1.0, 1.0, 1.0), "price": (100.0, 0.0, 50.0, -50.0)},
            intro="**Domain** — a zero current price sends the reciprocal to ``-inf``, a zero previous "
            "price sends the next bar to ``+inf``, and a negative price stays finite:",
        ),
        Example(
            inputs={"quantity": (1.0, float("-inf"), 2.0), "price": (float("inf"), 20.0, 5.0)},
            intro="**Non-finite input** — an infinite price contributes ``1 / inf = 0`` to the reciprocal "
            "change while an infinite short quantity drives that bar's payoff to ``+inf``, leaving "
            "the finite tail row unaffected:",
            round_to=4,
        ),
    ),
)
