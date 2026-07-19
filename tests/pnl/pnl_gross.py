"""Declaration for ``pomata.pnl.pnl_gross`` — the linear mark-to-market PnL, one-bar lag, propagating, degree-1."""

import math

from pomata.pnl import pnl_gross
from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, SpaceCost, Warmup
from tests.pnl.harness import suite_pnl
from tests.pnl.oracles import reference_pnl_gross
from tests.support.declaration import Example, Golden, Pin, ScaleAxis

PNL_GROSS = suite_pnl(
    factory=pnl_gross,
    inputs=("quantity", "price"),
    params={"multiplier": 1.0},
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    space=SpaceCost.CASH,
    sign=ConventionSign.LONG_SHORT,
    warmup=Warmup.ONE_ROW,
    oracle=reference_pnl_gross,
    # Degree-1 homogeneous in the position and in the price (each scales the currency P&L linearly).
    scaling=(
        ScaleAxis(roles=("quantity",), degree=1),
        ScaleAxis(roles=("price",), degree=1),
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
            "quantity": (10.0, 10.0, -5.0, -5.0, 20.0),
            "price": (100.0, 102.0, 101.0, 104.0, 103.0),
        },
        output=(None, 20.0, 5.0, -15.0, -20.0),
    ),
    pins=(
        Pin(
            label="null_precedence_null_quantity_nan_price",
            inputs={"quantity": (1.0, None, 2.0), "price": (100.0, math.nan, 110.0)},
            expected=(None, None, math.nan),
            reason="a null quantity against a NaN price yields null (null wins); the next bar reads the NaN previous "
            "price and propagates to NaN",
        ),
        Pin(
            label="null_precedence_nan_quantity_null_price",
            inputs={"quantity": (1.0, math.nan, 2.0), "price": (100.0, None, 110.0)},
            expected=(None, None, None),
            reason="the reverse precedence direction: a NaN quantity against a null price yields null, and the null "
            "previous price also nulls the next bar",
        ),
        Pin(
            label="short_on_flat_price_is_signed_zero",
            inputs={"quantity": (-5.0, -5.0), "price": (100.0, 100.0)},
            expected=(None, -0.0),
            reason="a short over a flat price yields IEEE -0.0 (a negative quantity times an exact +0.0 delta carries "
            "the sign bit); assert_matches reads -0.0 == 0.0",
            signed=True,
        ),
        Pin(
            label="consecutive_infinities_make_nan",
            inputs={"quantity": (10.0, 10.0, 10.0, 10.0), "price": (math.inf, math.inf, 1.0, -math.inf)},
            expected=(None, math.nan, -math.inf, -math.inf),
            reason="two consecutive equal-sign infinite prices make the second bar's price change inf - inf = NaN; the "
            "property tiers set allow_infinity=False",
        ),
        Pin(
            label="multiplier_50x_golden",
            inputs={
                "quantity": (10.0, 10.0, -5.0, -5.0, 20.0),
                "price": (100.0, 102.0, 101.0, 104.0, 103.0),
            },
            expected=(None, 1000.0, 250.0, -750.0, -1000.0),
            reason="a 50x futures-multiplier golden, exercising the exact multiplier "
            "arithmetic a single canonical golden cannot",
            params_override={"multiplier": 50.0},
        ),
    ),
    wikipedia="https://en.wikipedia.org/wiki/Mark-to-market_accounting",
    see_also=(
        ("returns_gross", "The return-flow counterpart (weight times asset return)."),
        ("pnl_net", "Subtracts the composed cost from this gross PnL."),
        ("pnl_gross_inverse", "The coin-margined (inverse-contract) version, nonlinear in price."),
    ),
    notes=(
        (
            "No lookahead (alignment is the caller's)",
            "The PnL assumes ``quantity`` at row ``t`` is the position held over the price change "
            "into row ``t``. To stay lookahead-free, that quantity must depend only on information "
            "available before that price; if it is decided on the same bar's close, lag it by one bar "
            "(``pnl_gross(quantity.shift(1), price)``). Nothing is shifted for you, so a quantity you "
            "have already aligned is never double-shifted.",
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
            "Non-finite input",
            "an ``inf`` quantity follows IEEE-754 through the arithmetic, and an infinite ``price`` "
            "propagates through the one-bar change (the sign, and any ``inf - inf = NaN``, included).",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The gross PnL for each row, the same length as the inputs. The first value is ``null`` "
    "(warm-up) -- the previous price ``price.shift(1)`` is undefined for the first row, so no "
    "price change can be measured there.",
    raises_prose="ValueError: If ``multiplier`` is not a finite number ``> 0`` (i.e. ``<= 0``, ``NaN``, or ``±inf``).",
    args_prose={
        "quantity": "Signed position size in units / shares / contracts held over the bar (e.g. ``100``, ``-2``).",
        "price": 'Instrument price series (e.g. ``pl.col("close")``); must share a length and alignment '
        "with ``quantity``.",
        "multiplier": "Contract multiplier / point value (e.g. ``50`` for an E-mini S&P future); ``1.0`` for "
        "cash equity and spot. Must be a finite number ``> 0``.",
    },
    intro_basic="Basic usage on a held quantity and a price series:",
    examples=(
        Example(
            inputs={
                "quantity": (10.0, 10.0, -5.0, -5.0, 20.0, 20.0, -10.0, -10.0),
                "price": (100.0, 102.0, 101.0, 104.0, 103.0, 105.0, 104.0, 106.0),
            },
            round_to=4,
        ),
        Example(
            inputs={
                "quantity": (10.0, 10.0, -5.0, -5.0, 2.0, 2.0, 2.0, 2.0),
                "price": (100.0, 102.0, 101.0, 104.0, 50.0, 51.0, 49.0, 52.0),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("AAPL",) * 4 + ("NVDA",) * 4,
            round_to=4,
        ),
        Example(
            inputs={"quantity": (10.0, None, -5.0, float("nan"), 20.0), "price": (100.0, 102.0, 101.0, 104.0, 103.0)},
            intro="A leading warm-up ``null`` (row 0, no prior price), then a ``null`` and a ``NaN`` in "
            "``quantity`` that void only their own rows:",
            round_to=4,
        ),
        Example(
            inputs={"quantity": (10.0, 10.0, 10.0, 10.0), "price": (float("inf"), float("inf"), 1.0, float("-inf"))},
            intro="**Non-finite input** — two consecutive equal-sign infinite prices make that bar's price "
            "change ``inf - inf``, so the PnL is ``NaN``:",
        ),
    ),
)
