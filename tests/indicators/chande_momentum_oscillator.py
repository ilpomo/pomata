"""Declaration for ``pomata.indicators.chande_momentum_oscillator`` — the gain/loss momentum ratio, window-nulling."""

import math

import polars as pl

from pomata.indicators import chande_momentum_oscillator
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_chande_momentum_oscillator
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape
from tests.support.strategies import windows_well_spread


def _windows_well_spread(frame: pl.DataFrame) -> bool:
    """
    Reject a near-flat window: where the changes over a window are tiny relative to the accumulated series magnitude,
    the implementation's one-pass rolling gain/loss sums lose the window's contribution to a sub-ULP residual (and its
    residual-free flat guard may fire ``NaN`` outright), while the naive oracle recomputes the window fresh — the two
    round to opposite sides of the 0/0 boundary there. Measured: zero impl-vs-oracle failures on every admitted
    draw, with real disagreement starting right at the predicate's own boundary, so the cut sits where the hazard
    actually is. The exact-flat corners are fixed by the ``flat_window`` / ``flat_tail`` pins; the fuzz stays on
    genuinely-moving windows.
    """
    return windows_well_spread(frame.to_series(0).to_list(), 3)


CHANDE_MOMENTUM_OSCILLATOR = suite_indicators(
    factory=chande_momentum_oscillator,
    inputs=("price",),
    params={"window": 3},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=Warmup.WINDOW,
    oracle=reference_chande_momentum_oscillator,
    scaling=(ScaleAxis(roles=("price",), degree=0),),
    talib=RelationTalib.DOCUMENTED_DIVERGENCE,
    talib_reason="pomata uses Chande's original fixed-window sums; TA-Lib uses Wilder smoothing (CMO == 2*RSI - 100).",
    raises=(({"window": 0}, r"window must be >= 1"),),
    conditioning=_windows_well_spread,
    golden=Golden(
        inputs={"price": (10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0)},
        output=(None, None, None, 33.3333, 50.0, 50.0, 50.0, 50.0),
    ),
    pins=(
        Pin(
            label="flat_window_is_nan",
            inputs={"price": (10.0, 10.0, 10.0, 10.0, 10.0)},
            expected=(None, None, None, math.nan, math.nan),
            reason="an all-flat window (every change exactly zero) is the 0/0 degenerate, returned as NaN — the "
            "exact core of the near-flat regime the conditioning filter excludes from the property tiers",
            covers_conditioning=True,
        ),
        Pin(
            label="flat_tail_after_movement_is_nan",
            inputs={
                "price": (
                    30426.583515139646,
                    30426.583514906622,
                    30426.583514906622,
                    30426.58351574153,
                    126995.79017007923,
                    126995.79017011753,
                    112548.45267126478,
                    112548.45267126478,
                    112548.4526722116,
                    -512653.3416246533,
                    -512653.3416243748,
                    -1000000.0,
                    -1000000.0,
                    -1000000.0,
                    -1000000.0,
                    -1000000.0,
                    -1000000.0,
                    -1000000.0,
                    -1000000.0,
                    -1000000.0,
                    -1000000.0,
                    -1000000.0,
                    -1000000.0,
                    -1000000.0,
                )
            },
            params_override={"window": 10},
            expected=(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                -73.7661501723971,
                -84.21510799887183,
                -84.21510799887183,
                -84.21510799899752,
                -99.99999999977577,
                -99.99999999978253,
                -99.9999999997797,
                -99.9999999997797,
                -99.99999999994994,
                -99.99999999988569,
                -100.0,
                math.nan,
                math.nan,
                math.nan,
            ),
            reason="the residual-free flat guard must not over-fire: windows straddling the flat transition still "
            "resolve to the streaming quotient until the window is entirely inside the flat tail, where it is the 0/0 "
            "degenerate",
        ),
        Pin(
            label="saturates_up",
            inputs={"price": (10.0, 11.0, 12.0, 13.0, 14.0)},
            expected=(None, None, None, 100.0, 100.0),
            reason="an all-up window (zero loss) saturates at exactly +100",
        ),
        Pin(
            label="saturates_down",
            inputs={"price": (14.0, 13.0, 12.0, 11.0, 10.0)},
            expected=(None, None, None, -100.0, -100.0),
            reason="an all-down window (zero gain) saturates at exactly -100",
        ),
        Pin(
            label="window_one_is_move_direction",
            inputs={"price": (1.0, 3.0, 2.0, 5.0)},
            params_override={"window": 1},
            expected=(None, 100.0, -100.0, 100.0),
            reason="window=1 collapses the rolling gain/loss sums to the raw move direction: +100 up, -100 down",
        ),
    ),
    reference="Chande, T. S. & Kroll, S. (1994). *The New Technical Trader*. Wiley.",
    see_also=(
        ("rsi", "The Wilder-smoothed sibling, bounded in ``[0, 100]``."),
        ("roc", "A simpler single-horizon momentum measure."),
        ("mom", "The absolute-difference momentum sibling."),
    ),
    notes=(
        (
            "Documented TA-Lib divergence",
            "TA-Lib Wilder-smooths the gains and losses, so its CMO equals ``2 * RSI - 100``; pomata "
            "sums them over a fixed window, Chande's original 1994 construction, so the two never "
            "agree even at steady state and the differential tier holds the CMO out as a documented "
            "divergence.",
        ),
    ),
    note_extension="For this oscillator the limit is concrete: the windowed gain / loss sums ride Polars' "
    "incremental sliding kernel, so a window whose scale sits tens of orders of magnitude "
    "below a value that has already slid out can inherit a stale residue; the clamp keeps the "
    "output inside ``[-100, 100]``, and no real market series builds that spread.",
    bullets=(
        ("Null", "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null values)."),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there."),
        (
            "Degenerate denominator",
            "an exactly-flat window has every change zero, detected via the residual-free rolling "
            "maximum of ``|change|``, so the result is a ``0 / 0``, i.e. ``NaN``.",
        ),
        (
            "Stability",
            "a near-flat window (tiny changes after a much larger one has slid out of the streaming "
            "sums) is not silenced: its quotient is clipped to ``[-100, +100]``, so it stays in range "
            "but, past a sane dynamic range, degrades in precision (see the precision note above).",
        ),
        (
            "window == 1",
            "the rolling sums collapse to the single change, so each row reports ``+100`` on an up "
            "move, ``-100`` on a down move, and ``NaN`` on no move.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The oscillator (in percent) for each row, the same length as the input. The first "
    "``window`` rows are ``null`` (warm-up): row ``0`` has no change, and the rolling sums "
    "need ``window`` non-null changes before emitting.",
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of one-step changes summed in the window. Must be ``>= 1``.",
    },
    intro_basic="Basic usage on a single price series:",
    example_columns={"price": "close"},
    examples=(
        Example(inputs={"price": (10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0)}, params={"window": 3}, round_to=4),
        Example(
            inputs={"price": (10.0, 11.0, 12.0, 11.0, 13.0, 20.0, 19.0, 21.0, 22.0, 20.0)},
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker warms up independently:",
            partition=("A", "A", "A", "A", "A", "B", "B", "B", "B", "B"),
            params={"window": 3},
            round_to=4,
        ),
        Example(
            inputs={"price": (10.0, 11.0, 12.0, None, 14.0, float("nan"), 16.0, 17.0)},
            intro="A ``null`` (any window it touches yields ``null``) and a ``NaN`` (which propagates) make "
            "the handling visible:",
            params={"window": 3},
            round_to=4,
        ),
        Example(
            inputs={"price": (10.0, 10.0, 10.0, 10.0, 10.0)},
            intro="**Degenerate denominator** — an all-flat window has every change exactly zero, the "
            "``0/0`` degenerate, so the result is ``NaN``:",
            params={"window": 3},
        ),
        Example(
            inputs={"price": (1.0, 3.0, 2.0, 5.0)},
            intro="**window == 1** — a one-bar window collapses the rolling gain/loss sums to the raw move "
            "direction, so each row reports ``+100`` on an up move and ``-100`` on a down move:",
            params={"window": 1},
        ),
    ),
)
