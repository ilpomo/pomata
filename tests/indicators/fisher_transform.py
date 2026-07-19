"""Declaration for ``pomata.indicators.fisher_transform`` — the Gaussianized channel struct (fisher, signal)."""

import math

from pomata.indicators import fisher_transform
from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Warmup
from tests.indicators.harness import suite_indicators
from tests.indicators.oracles import reference_fisher_transform
from tests.support.declaration import Example, Golden, Pin, ScaleAxis, Shape

FISHER_TRANSFORM = suite_indicators(
    factory=fisher_transform,
    inputs=("high", "low"),
    params={"window": 10},
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.STRUCT,
    fields=("fisher", "signal"),
    warmup=Warmup.PER_FIELD,
    warmup_value={"fisher": 9, "signal": 10},
    oracle=reference_fisher_transform,
    scaling=(ScaleAxis(roles=("high", "low"), degree={"fisher": 0, "signal": 0}),),
    talib=RelationTalib.NO_EQUIVALENT,
    talib_reason="TA-Lib has no Fisher Transform.",
    raises=(({"window": 0}, r"window must be >= 1"),),
    flow_horizon=60,
    golden=Golden(
        inputs={"high": (2.0, 4.0, 3.0), "low": (0.0, 2.0, 1.0)},
        output={
            "fisher": (None, 0.3428, 0.0621),
            "signal": (None, None, 0.3428),
        },
        params={"window": 2},
    ),
    pins=(
        Pin(
            label="window_one_single_row_is_flat_nan",
            inputs={"high": (11.0,), "low": (9.0,)},
            params_override={"window": 1},
            expected={"fisher": (math.nan,), "signal": (None,)},
            reason="window=1 is flat by construction (max == min), so fisher is NaN from the first row while signal "
            "is still warm-up null",
        ),
        Pin(
            label="flat_window_is_nan",
            inputs={"high": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0)},
            params_override={"window": 3},
            expected={
                "fisher": (None, None, math.nan, math.nan, math.nan, math.nan),
                "signal": (None, None, None, math.nan, math.nan, math.nan),
            },
            reason="a constant series has max == min over every window: the channel normalization is 0/0 NaN, which "
            "bridges through the recursion",
        ),
    ),
    reference='Ehlers, J. F. (2002). "Using the Fisher Transform." *Technical Analysis of Stocks & '
    "Commodities*, 20(11).",
    see_also=(
        ("williams_r", "The raw channel position the transform sharpens."),
        ("rsi_stochastic", "Another channel-normalized momentum oscillator, bounded rather than tail-stretched."),
        ("stochastic_fast", "The %K channel position, the same normalization before the transform."),
    ),
    notes=(
        (
            "Clamp convention",
            "The smoothed position is held to a symmetric ``[-0.999, 0.999]`` -- a monotone clamp at "
            "the threshold, keeping the argument strictly inside the log's domain. Ehlers' original "
            "snaps any value past ``\\pm 0.99`` straight to ``\\pm 0.999``; pomata uses the monotone "
            "form (the modern convention), which agrees everywhere except on the thin ``(0.99, "
            "0.999)`` band the original discontinuously lifts (at ``0.999`` both map to ``0.999``).",
        ),
        (
            "Seeding",
            "Both recursions start from ``0`` (the bar before the first defined row contributes ``0`` "
            "to each), matching Ehlers' zero-initialized series; the smoothing then washes the seed "
            "out geometrically.",
        ),
    ),
    note_extension="\n\n"
    "It is invariant under a positive affine rescaling of the inputs: the channel "
    "normalization ``(p - \\min)/(\\max - \\min)`` cancels any common scale, so the transform "
    "depends only on the price's *shape*, not its level or units.",
    bullets=(
        (
            "Null",
            "a window containing a ``null`` yields ``null`` (the window must hold ``window`` non-null "
            "values) — the recursion bridges those rows and resumes once the window clears.",
        ),
        ("NaN", "a ``NaN`` inside the window propagates, yielding ``NaN`` there, likewise bridged."),
        ("Insufficient sample", "a window longer than the series never completes, so the result is ``null``."),
        (
            "Degenerate denominator",
            "when ``max == min`` over the window the channel has no range to normalize by, so the "
            "result is a ``0 / 0``, i.e. ``NaN``.",
        ),
        (
            "window == 1",
            "the channel spans a single bar, so ``max == min`` makes it flat by construction and "
            "``fisher`` is ``NaN`` from the first row.",
        ),
        (
            "Partitioning",
            "wrap the call in ``.over(...)`` for a multi-series panel so each series is computed on its own history.",
        ),
    ),
    returns_body="A struct ``pl.Expr`` with two ``Float64`` fields, the same length as the inputs:"
    "\n\n"
    "- ``fisher`` — the Fisher Transform. - ``signal`` — ``fisher`` lagged one bar (``null`` "
    "for one further row)."
    "\n\n"
    "The first ``window - 1`` rows are ``null`` (the channel's warm-up). Read a field with "
    '``.struct.field("fisher")`` or split both with ``.struct.unnest()``.',
    raises_prose="ValueError: If ``window < 1``.",
    args_prose={
        "window": "Number of observations in the moving window (canonically ``10``). Must be ``>= 1``.",
    },
    intro_basic="Basic usage on high-low bars:",
    examples=(
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 13.0, 14.0, 15.0),
                "low": (9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 12.0, 13.0, 14.0),
            },
            params={"window": 3},
            round_to=4,
            fields=("fisher", "signal"),
        ),
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, 11.5, 13.0, 20.0, 21.0, 22.0, 21.5, 23.0),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 19.0, 20.0, 21.0, 20.5, 22.0),
            },
            intro="On a multi-ticker panel, wrap the call in ``.over`` so each ticker's channel warms up "
            "independently:",
            partition=("A", "A", "A", "A", "A", "B", "B", "B", "B", "B"),
            params={"window": 3},
            round_to=4,
            fields=("fisher",),
        ),
        Example(
            inputs={
                "high": (10.0, 11.0, 12.0, None, 13.0, float("nan"), 15.0),
                "low": (9.0, 10.0, 11.0, 10.5, 12.0, 11.5, 14.0),
            },
            intro="A ``null`` (a window touching it yields ``null``) and a ``NaN`` (which propagates) make it visible:",
            params={"window": 3},
            round_to=4,
            fields=("fisher",),
        ),
        Example(
            inputs={"high": (11.0,), "low": (9.0,)},
            intro="**Insufficient sample** — a one-bar window is flat by construction (``max == min``), so "
            "``fisher`` is ``NaN`` from the first row:",
            params={"window": 1},
            fields=("fisher",),
        ),
        Example(
            inputs={"high": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0), "low": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0)},
            intro="**Degenerate denominator** — a constant series has ``max == min`` over every window, so "
            "the channel normalization is the ``0/0`` boundary, which bridges through the recursion "
            "as ``NaN``:",
            params={"window": 3},
            fields=("fisher",),
        ),
    ),
)
