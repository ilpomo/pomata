"""
Self-tests of the indicators-dialect engine: the STRUCT per-field structural-flow contract and the recomposition
identity rung.

The STRUCT contract runs each declared field's lane through the SAME series-flow shape handlers a plain SERIES lane
uses — the behavior axes are per-declaration, the check is per-field, and every field is read (not only the first).
These are exercised against a real ``pomata.indicators`` struct (``donchian_channels``, a faithful oracle inlined)
plus a minimal heterogeneous struct that isolates the "every field, not the first" reading: its second field violates
a declaration its first field satisfies, so a green ``pytest.raises`` naming that second field proves the loop reached
it. The recomposition rung compares the factory against a zero-argument recomposition expression — never the oracle —
so its declarations carry a name-only oracle stub. Every contract is shown both green (the true declaration passes)
and red (a wrong declaration raises), the born-red demonstrations reverted by construction.
"""

import dataclasses
import enum
import math
from collections.abc import Sequence

import polars as pl
import pytest

from pomata.indicators import donchian_channels, sma
from pomata.metrics import max_drawdown, recovery_ratio, total_return
from tests.support import rungs
from tests.support.declaration import (
    Declaration,
    OracleFn,
    ScaleAxis,
    ScaleExempt,
    Shape,
)

# ======================================================================================================================
# Family-neutral dialects — the indicators family enums land with that family; the engine reads a member's NAME to pick
# a shape handler, so a local mirror is enough to drive the contracts here.
# ======================================================================================================================


class _BehaviorNull(enum.Enum):
    BRIDGED = "bridged"
    IN_WINDOW_IS_NULL = "in_window_is_null"
    PROPAGATES = "propagates"
    LATCHES = "latches"
    ABSORBED = "absorbed"


class _BehaviorNan(enum.Enum):
    PROPAGATES = "propagates"
    LATCHES = "latches"


class _Unhandled(enum.Enum):
    """A reducing-only member the SERIES / STRUCT shape map has no handler for — drives the fail-closed path."""

    SKIPPED = "skipped"


def _stub(name: str) -> OracleFn:
    """A name-only oracle for a rung that reads the factory, not the oracle; loud if it is ever actually called."""

    def oracle(*_args: object, **_kwargs: object) -> object:
        message = f"{name}: this oracle is not exercised by the rung under test"
        raise AssertionError(message)

    oracle.__name__ = name
    return oracle


# ======================================================================================================================
# A faithful inlined oracle for the real struct under test — the rolling high/low channel (Polars rolling max/min)
# ======================================================================================================================


def _window_extreme(window_values: Sequence[float | None], *, take_max: bool) -> float | None:
    """The window's rolling extreme: null if any value is null, else NaN if any is NaN, else the max or min."""
    if any(value is None for value in window_values):
        return None
    finite = [value for value in window_values if value is not None]
    if any(math.isnan(value) for value in finite):
        return math.nan
    return max(finite) if take_max else min(finite)


def _midline(top: float | None, bottom: float | None) -> float | None:
    """The mean of the two bands: null if either is null, else NaN if either is NaN, else their midpoint."""
    if top is None or bottom is None:
        return None
    if math.isnan(top) or math.isnan(bottom):
        return math.nan
    return (top + bottom) / 2


def reference_donchian_channels(
    high: Sequence[float | None], low: Sequence[float | None], window: int
) -> dict[str, list[float | None]]:
    """Naive Donchian channels: the window's highest high (upper), lowest low (lower), and their mean (middle)."""
    lower: list[float | None] = []
    middle: list[float | None] = []
    upper: list[float | None] = []
    for index in range(len(high)):
        if index + 1 < window:
            lower.append(None)
            middle.append(None)
            upper.append(None)
            continue
        top = _window_extreme(high[index + 1 - window : index + 1], take_max=True)
        bottom = _window_extreme(low[index + 1 - window : index + 1], take_max=False)
        upper.append(top)
        lower.append(bottom)
        middle.append(_midline(top, bottom))
    return {"lower": lower, "middle": middle, "upper": upper}


DONCHIAN_CHANNELS = Declaration(
    family="indicators",
    factory=donchian_channels,
    inputs=("high", "low"),
    params={"window": 3},
    shape=Shape.STRUCT,
    fields=("lower", "middle", "upper"),
    behavior_null=_BehaviorNull.IN_WINDOW_IS_NULL,
    behavior_nan=_BehaviorNan.PROPAGATES,
    oracle=reference_donchian_channels,
    scaling=(ScaleAxis(roles=("high", "low"), degree={"lower": 1, "middle": 1, "upper": 1}),),
    warmup={"lower": 2, "middle": 2, "upper": 2},
    window="window",
    raises=(({"window": 0}, r"window must be >= 1"),),
)


# ======================================================================================================================
# The STRUCT per-field structural-flow contract on a real 3-field struct
# ======================================================================================================================


class TestStructFlow:
    """A struct runs the SERIES shape once per field: every field is held to the declared null / NaN behavior."""

    def test_in_window_null_green(self) -> None:
        """Donchian's three bands each null every window they overlap, then recover — the IN_WINDOW_IS_NULL shape."""
        rungs.check_behavior_null(DONCHIAN_CHANNELS)

    def test_nan_propagates_green(self) -> None:
        """Donchian's three bands each carry an interior NaN for the windows it spans, then clear it — PROPAGATES."""
        rungs.check_behavior_nan(DONCHIAN_CHANNELS)

    def test_nan_declared_latches_red(self) -> None:
        """A PROPAGATES struct wrongly declared LATCHES goes red — a band clears its NaN past the window."""
        wrong = dataclasses.replace(DONCHIAN_CHANNELS, behavior_nan=_BehaviorNan.LATCHES)
        with pytest.raises(AssertionError, match=r"declared LATCHES"):
            rungs.check_behavior_nan(wrong)

    def test_unknown_null_member_is_fail_closed(self) -> None:
        """A struct null declared with a member the SERIES/STRUCT engine has no shape for is fail-closed."""
        wrong = dataclasses.replace(DONCHIAN_CHANNELS, behavior_null=_Unhandled.SKIPPED)
        with pytest.raises(AssertionError, match=r"no structural contract for behavior 'SKIPPED'"):
            rungs.check_behavior_null(wrong)


# ======================================================================================================================
# Per-field warm-up and per-field scaling on the same real struct
# ======================================================================================================================


class TestStructPerField:
    """The per-field forms a struct declares — warm-up and scale degree — are read and checked per field, not once."""

    def test_warmup_per_field_green(self) -> None:
        """Each band carries exactly its declared leading-null count."""
        rungs.check_warmup(DONCHIAN_CHANNELS)

    def test_warmup_field_off_by_one_red(self) -> None:
        """One band's declared warm-up off by one goes red naming THAT field — proof the check reads every field."""
        wrong = dataclasses.replace(DONCHIAN_CHANNELS, warmup={"lower": 2, "middle": 2, "upper": 3})
        with pytest.raises(AssertionError, match=r"donchian_channels\.upper: 2 leading nulls, declared 3"):
            rungs.check_warmup(wrong)

    def test_scaling_per_field_green(self) -> None:
        """Scaling the price legs scales each band by its declared degree — every field's degree, read from the map."""
        rungs.check_scaling(DONCHIAN_CHANNELS)


# ======================================================================================================================
# The flow contract reads EVERY field, not the first — a minimal heterogeneous struct isolates the reading
# ======================================================================================================================


def recover_latch_struct(returns: pl.Expr) -> pl.Expr:
    """A struct whose first field clears an interior NaN (identity) and whose second latches it (a cumulative sum)."""
    return pl.struct(recovers=returns, latches=returns.cum_sum())


def reference_recover_latch_struct(returns: Sequence[float | None]) -> dict[str, list[float | None]]:
    """Identity for ``recovers``; a running sum for ``latches`` (a NaN latches the total) on a null-free probe."""
    latches: list[float | None] = []
    total = 0.0
    for value in returns:
        assert value is not None  # the interior-NaN flow probe injects a NaN, never a null
        total = total + value
        latches.append(total)
    return {"recovers": list(returns), "latches": latches}


RECOVER_LATCH = Declaration(
    family="indicators",
    factory=recover_latch_struct,
    inputs=("returns",),
    params={},
    shape=Shape.STRUCT,
    fields=("recovers", "latches"),
    behavior_null=_BehaviorNull.PROPAGATES,
    behavior_nan=_BehaviorNan.PROPAGATES,
    oracle=reference_recover_latch_struct,
    scaling=(ScaleAxis(roles=("returns",), degree={"recovers": 1, "latches": 1}),),
)


def test_flow_reads_the_second_field() -> None:
    """PROPAGATES holds for the first field but not the second (which latches), so the contract must reach the second.

    The declaration is true for ``recovers`` and false for ``latches``; if the flow loop stopped at the first field it
    would find no violation and pass. The raise, naming lane ``'latches'``, proves every field is read.
    """
    with pytest.raises(AssertionError, match=r"lane 'latches'"):
        rungs.check_behavior_nan(RECOVER_LATCH)


# ======================================================================================================================
# The recomposition identity rung — factory vs a recomposition expression (never the oracle)
# ======================================================================================================================


RECOVERY_RATIO = Declaration(
    family="metrics",
    factory=recovery_ratio,
    inputs=("equity_curve",),
    params={},
    shape=Shape.REDUCING,
    behavior_null=_BehaviorNull.PROPAGATES,
    behavior_nan=_BehaviorNan.PROPAGATES,
    oracle=_stub("reference_recovery_ratio"),
    scaling=ScaleExempt(reason="the engine tests do not exercise the scale rung"),
    recomposition=lambda: total_return(pl.col("equity_curve")) / max_drawdown(pl.col("equity_curve")).abs(),
)


def struct_two_sma(close: pl.Expr) -> pl.Expr:
    """A two-field struct of two real smoothers at different windows — a recomposition rebuilds it field by field."""
    return pl.struct(fast=sma(close, 3), slow=sma(close, 5))


STRUCT_TWO_SMA = Declaration(
    family="indicators",
    factory=struct_two_sma,
    inputs=("close",),
    params={},
    shape=Shape.STRUCT,
    fields=("fast", "slow"),
    behavior_null=_BehaviorNull.PROPAGATES,
    behavior_nan=_BehaviorNan.PROPAGATES,
    oracle=_stub("reference_struct_two_sma"),
    scaling=(ScaleAxis(roles=("close",), degree={"fast": 1, "slow": 1}),),
    recomposition=lambda: pl.struct(fast=sma(pl.col("close"), 3), slow=sma(pl.col("close"), 5)),
)


class TestRecomposition:
    """The factory equals its recomposition from public functions, lane by lane; a detached recomposition goes red."""

    def test_reducing_recomposition_green(self) -> None:
        """recovery_ratio equals its own total-return-over-max-drawdown recomposition on the probe frame."""
        rungs.check_recomposition(RECOVERY_RATIO)

    def test_struct_recomposition_green(self) -> None:
        """A struct of two smoothers equals a recomposition that rebuilds each field — every lane compared."""
        rungs.check_recomposition(STRUCT_TWO_SMA)

    def test_no_recomposition_skips(self) -> None:
        """A declaration carrying no recomposition skips the rung cleanly."""
        without = dataclasses.replace(RECOVERY_RATIO, recomposition=None)
        with pytest.raises(pytest.skip.Exception, match=r"no recomposition declared"):
            rungs.check_recomposition(without)

    def test_detached_reducing_recomposition_red(self) -> None:
        """recovery_ratio pointed at a wrong recomposition (numerator only, no drawdown divisor) goes red."""
        detached = dataclasses.replace(RECOVERY_RATIO, recomposition=lambda: total_return(pl.col("equity_curve")))
        with pytest.raises(AssertionError, match=r"check_recomposition disagreed"):
            rungs.check_recomposition(detached)

    def test_detached_struct_recomposition_red(self) -> None:
        """A struct recomposition whose second field has the wrong window goes red, naming that field's lane."""
        detached = dataclasses.replace(
            STRUCT_TWO_SMA, recomposition=lambda: pl.struct(fast=sma(pl.col("close"), 3), slow=sma(pl.col("close"), 3))
        )
        with pytest.raises(AssertionError, match=r"lane 'slow'"):
            rungs.check_recomposition(detached)
