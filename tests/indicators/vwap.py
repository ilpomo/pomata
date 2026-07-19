"""Declaration for ``pomata.indicators.vwap`` — the cumulative volume-weighted average price, gap-bridging, degree-1."""

import math

from pomata.indicators import vwap
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_vwap
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

VWAP = suite_indicators(
    factory=vwap,
    inputs=("high", "low", "close", "volume"),
    params={},
    null=BehaviorNull.BRIDGED,
    nan=BehaviorNan.LATCHES,
    shape=Shape.SERIES,
    warmup=Warmup.NONE,
    oracle=reference_vwap,
    scaling=(
        ScaleAxis(roles=("high", "low", "close"), degree=1),
        ScaleAxis(roles=("volume",), degree=0),
    ),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no VWAP.",
    golden=Golden(
        inputs={
            "high": (2.0, 4.0, 6.0),
            "low": (0.0, 2.0, 4.0),
            "close": (1.0, 3.0, 5.0),
            "volume": (10.0, 20.0, 30.0),
        },
        output=(1.0, 2.3333, 3.6667),
    ),
    pins=(
        Pin(
            label="zero_leading_volume_is_nan_then_recovers",
            inputs={
                "high": (10.0, 11.0, 12.0),
                "low": (8.0, 9.0, 10.0),
                "close": (9.0, 10.0, 11.0),
                "volume": (0.0, 100.0, 100.0),
            },
            expected=(math.nan, 10.0, 10.5),
            reason="a zero cumulative volume at the first bar is the 0/0 degenerate (NaN); once volume accrues the "
            "running average recovers",
        ),
    ),
    reference='Berkowitz, S. A., Logue, D. E., & Noser, E. A. (1988). "The Total Cost of Transactions '
    'on the NYSE." *The Journal of Finance*, 43(1), 97-112.',
    doi="https://doi.org/10.1111/j.1540-6261.1988.tb02591.x",
    wikipedia="https://en.wikipedia.org/wiki/Volume-weighted_average_price",
    see_also=(
        ("vwma", "The windowed volume-weighted moving average, for a rolling rather than anchored weight."),
        ("price_typical", "The per-bar price this weights."),
        ("sma", "The equal-weighted moving average, the volume-blind analog."),
    ),
    notes=(
        (
            "Anchoring",
            "VWAP accumulates from the start of the partition, so wrap the call in "
            "``.over(session_key)`` to reset it per session (e.g. one trading day): "
            '``vwap(...).over("session")``. Without an anchor it accumulates across the whole series, '
            "the classic VWAP misuse.",
        ),
        (
            "Inputs",
            "``high`` / ``low`` / ``close`` / ``volume`` must share a length and alignment; "
            "``volume`` is expected non-negative (a negative volume is summed as-is, with no guard -- "
            "garbage in, garbage out).",
        ),
        (
            "Seeding",
            "At the head a zero cumulative volume gives ``0 / 0 == NaN`` until volume accrues; an "
            "interior zero-volume bar adds nothing (the prefix sums carry forward, with no "
            "subtract-on-exit residual).",
        ),
    ),
    opener_override="Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` "
    "band) on any finite input within a sane dynamic range; the documentation's *Correctness* "
    "page gives the method and the conditioning limit of the long cumulative sums beyond it.",
    bullets=(
        (
            "Null",
            "a leading ``null`` run stays ``null`` until the first non-null seed; an interior "
            "``null`` yields ``null`` at that position while the recursion continues across the gap — "
            "both cumulative sums skip the bar together (a ``null`` price input drops its volume from "
            "the denominator too), so the bar is a clean missing observation, not a denominator-only "
            "contribution.",
        ),
        (
            "NaN",
            "a ``NaN`` contaminates the recursive state and yields ``NaN`` for every subsequent "
            "non-null position — unless a ``null`` sits on the same row, which masks the whole row "
            "out of both sums first, so nothing is poisoned.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="The running VWAP for each row, the same length as the inputs. There is no warm-up -- row "
    "``0`` is defined as soon as its cumulative volume is positive (a leading zero-volume run "
    "reads ``NaN`` until volume accrues).",
    examples=(
        Example(
            inputs={
                "high": (2.0, 4.0, 6.0),
                "low": (0.0, 2.0, 4.0),
                "close": (1.0, 3.0, 5.0),
                "volume": (10.0, 20.0, 30.0),
            },
            round_to=4,
        ),
        Example(
            inputs={
                "high": (2.0, 4.0, 12.0, 14.0),
                "low": (0.0, 2.0, 10.0, 12.0),
                "close": (1.0, 3.0, 11.0, 13.0),
                "volume": (10.0, 20.0, 10.0, 20.0),
            },
            intro="Anchor per session with ``.over`` so each day's VWAP restarts:",
            partition=("a", "a", "b", "b"),
            partition_col="session",
            round_to=4,
        ),
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 13.0, 14.0, 15.0),
                "low": (8.0, 9.0, 10.0, 11.0, 12.0, 13.0),
                "close": (9.0, 10.0, None, 12.0, float("nan"), 14.0),
                "volume": (100.0, 200.0, 300.0, 400.0, 500.0, 600.0),
            },
            intro="A ``null`` (yields ``null`` at that row) and a ``NaN`` (which latches in the running "
            "totals) make it visible:",
            round_to=4,
        ),
    ),
)
