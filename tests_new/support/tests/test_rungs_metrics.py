"""
Self-tests of the metrics-dialect engine: the REDUCING structural-flow contracts, the SERIES ``IN_WINDOW_IS_NULL``
shape, the twin-coherence rung, and the closed-form annualization rung.

These exercise the new rungs against real ``pomata.metrics`` functions with hand-built declarations (never registered,
so they do not touch the family registry the collectible modules parametrize over). The reference oracles the value
layer needs are inlined here so the file stays self-contained; the rungs that never call the oracle (twin coherence,
annualization) carry a name-only stub, since the declaration constructor validates the oracle NAME but the rung reads
the factory. Every new contract is shown both green (the true declaration passes) and red (a wrong declaration raises
the triaged failure) — the born-red demonstrations, permanent and reverted-by-construction.
"""

import dataclasses
import enum
import math
import re
from collections.abc import Sequence

import polars as pl
import pytest

from pomata.metrics import (
    cagr,
    downside_deviation,
    drawdown,
    drawdown_rolling,
    max_drawdown,
    total_return,
    treynor_ratio,
    volatility,
    volatility_rolling,
)
from tests_new.metrics.enums import Annualization, BehaviorNan, BehaviorNull
from tests_new.support import rungs
from tests_new.support.declaration import (
    Declaration,
    FactoryExpr,
    OracleFn,
    Pin,
    ScalarParam,
    ScaleAxis,
    ScaleExempt,
    Shape,
)
from tests_new.support.frames import probe_frame


class _Space(enum.Enum):
    RETURNS = "returns"


class _Sign(enum.Enum):
    LONG_SHORT = "long_short"


class _NonFinite(enum.Enum):
    IEEE_FLOW = "ieee_flow"


def _always(_frame: pl.DataFrame) -> bool:
    """A conditioning predicate that admits everything — enough to exercise the pairing guard."""
    return True


# ======================================================================================================================
# Inlined reference oracles — the value layer needs these; the coherence / annualization rungs read the factory instead
# ======================================================================================================================


def reference_total_return(equity_curve: Sequence[float | None]) -> float | None:
    """The last non-null equity minus one; nulls skipped, a NaN poisons — the reducing SKIPPED / POISONS dialect."""
    defined = [value for value in equity_curve if value is not None]
    if any(math.isnan(value) for value in defined):
        return math.nan
    if not defined:
        return None
    return defined[-1] - 1


def reference_volatility(returns: Sequence[float | None], periods_per_year: int) -> float | None:
    """The annualized two-pass sample standard deviation; nulls skipped, a NaN poisons, a constant series is zero."""
    observations = [value for value in returns if value is not None]
    if len(observations) < 2:
        return None
    if any(math.isnan(value) for value in observations):
        return math.nan
    if max(observations) == min(observations):
        return 0.0
    mean = sum(observations) / len(observations)
    variance = sum((value - mean) ** 2 for value in observations) / (len(observations) - 1)
    return math.sqrt(variance) * math.sqrt(periods_per_year)


def reference_volatility_rolling(
    values: Sequence[float | None], window: int, periods_per_year: int
) -> list[float | None]:
    """The reducing volatility over each trailing window; warm-up and any-null windows are ``None``."""
    output: list[float | None] = []
    for index in range(len(values)):
        if index < window - 1:
            output.append(None)
            continue
        window_slice = list(values[index - window + 1 : index + 1])
        if any(value is None for value in window_slice):
            output.append(None)
            continue
        output.append(reference_volatility(window_slice, periods_per_year))
    return output


def _stub(name: str) -> OracleFn:
    """A name-only oracle for a rung that reads the factory, not the oracle; loud if it is ever actually called."""

    def oracle(*_args: object, **_kwargs: object) -> object:
        message = f"{name}: this oracle is not exercised by the rung under test"
        raise AssertionError(message)

    oracle.__name__ = name
    return oracle


# ======================================================================================================================
# Declaration templates over real pomata.metrics functions
# ======================================================================================================================


def _metric(  # noqa: PLR0913
    *,
    factory: FactoryExpr,
    inputs: tuple[str, ...],
    params: dict[str, ScalarParam],
    oracle: OracleFn,
    null: BehaviorNull,
    nan: BehaviorNan,
    shape: Shape = Shape.REDUCING,
    annualization: enum.Enum | None = None,
    rolling_of: Declaration | None = None,
    window: str | None = None,
    warmup: int | None = None,
) -> Declaration:
    """Build a metrics declaration with the family-neutral placeholders filled in, keeping each test call terse."""
    raises = (({next(iter(params)): -999}, r"."),) if params else ()
    return Declaration(
        family="metrics",
        factory=factory,
        inputs=inputs,
        params=params,
        shape=shape,
        behavior_null=null,
        behavior_nan=nan,
        space=_Space.RETURNS,
        sign=_Sign.LONG_SHORT,
        nonfinite=_NonFinite.IEEE_FLOW,
        oracle=oracle,
        scaling=ScaleExempt(reason="the engine tests do not exercise the scale rung"),
        raises=raises,
        annualization=annualization,
        rolling_of=rolling_of,
        window=window,
        warmup=warmup,
    )


TOTAL_RETURN = _metric(
    factory=total_return,
    inputs=("equity_curve",),
    params={},
    oracle=reference_total_return,
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
)

VOLATILITY = _metric(
    factory=volatility,
    inputs=("returns",),
    params={"periods_per_year": 252},
    oracle=reference_volatility,
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.SQRT_TIME,
)

TREYNOR_RATIO = _metric(
    factory=treynor_ratio,
    inputs=("returns", "benchmark"),
    params={"periods_per_year": 252, "risk_free_rate": 0.0},
    oracle=_stub("reference_treynor_ratio"),
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.LINEAR,
)

CAGR = _metric(
    factory=cagr,
    inputs=("equity_curve",),
    params={"periods_per_year": 252},
    oracle=_stub("reference_cagr"),
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.GEOMETRIC,
)

MAX_DRAWDOWN = _metric(
    factory=max_drawdown,
    inputs=("equity_curve",),
    params={},
    oracle=_stub("reference_max_drawdown"),
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
    annualization=Annualization.NONE,
)

DRAWDOWN = _metric(
    factory=drawdown,
    inputs=("equity_curve",),
    params={},
    oracle=_stub("reference_drawdown"),
    null=BehaviorNull.PROPAGATES,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
)

VOLATILITY_ROLLING = _metric(
    factory=volatility_rolling,
    inputs=("returns",),
    params={"window": 3, "periods_per_year": 252},
    oracle=reference_volatility_rolling,
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=2,
    rolling_of=VOLATILITY,
    window="window",
)

DRAWDOWN_ROLLING = _metric(
    factory=drawdown_rolling,
    inputs=("equity_curve",),
    params={"window": 3},
    oracle=_stub("reference_drawdown_rolling"),
    null=BehaviorNull.IN_WINDOW_IS_NULL,
    nan=BehaviorNan.PROPAGATES,
    shape=Shape.SERIES,
    warmup=2,
    rolling_of=DRAWDOWN,
    window="window",
)

# A reducing metric with the same ``returns`` input and ``periods_per_year`` knob as volatility_rolling but a different
# statistic (a downside semi-deviation) — a WRONG twin whose per-window value does not match, so the coherence rung
# must go red. Its oracle is never called by the coherence rung, so a name-only stub suffices.
WRONG_RETURNS_TWIN = _metric(
    factory=downside_deviation,
    inputs=("returns",),
    params={"periods_per_year": 252},
    oracle=_stub("reference_downside_deviation"),
    null=BehaviorNull.SKIPPED,
    nan=BehaviorNan.POISONS,
)


# ======================================================================================================================
# Reducing structural-flow contracts
# ======================================================================================================================


class TestReducingFlow:
    """The REDUCING dialect: SKIPPED equals the row-removed recompute, POISONS goes NaN, PROPAGATES goes null."""

    def test_skipped_null_green(self) -> None:
        """A reducing SKIPPED metric passes: the poisoned scalar equals the factory with the null row removed."""
        rungs.check_behavior_null(TOTAL_RETURN)
        rungs.check_behavior_null(VOLATILITY)

    def test_poisons_nan_green(self) -> None:
        """A reducing POISONS metric passes: a NaN in the input poisons the scalar to NaN."""
        rungs.check_behavior_nan(TOTAL_RETURN)
        rungs.check_behavior_nan(VOLATILITY)

    def test_skipped_declared_propagates_red(self) -> None:
        """A SKIPPED reduction wrongly declared PROPAGATES goes red — its scalar is defined, not null."""
        wrong = dataclasses.replace(VOLATILITY, behavior_null=BehaviorNull.PROPAGATES)
        with pytest.raises(AssertionError, match=r"declared PROPAGATES: a null in the input must null the reduction"):
            rungs.check_behavior_null(wrong)

    def test_skipped_declared_poisons_is_fail_closed(self) -> None:
        """A reducing null declared with a NaN-only member (POISONS) has no null contract — fail-closed message."""
        wrong = dataclasses.replace(VOLATILITY, behavior_null=BehaviorNan.POISONS)
        with pytest.raises(AssertionError, match=r"no structural contract for behavior 'POISONS'"):
            rungs.check_behavior_null(wrong)

    def test_poisons_declared_propagates_is_fail_closed(self) -> None:
        """A reducing NaN declared PROPAGATES has no reducing-NaN contract — fail-closed shape message."""
        wrong = dataclasses.replace(VOLATILITY, behavior_nan=BehaviorNan.PROPAGATES)
        with pytest.raises(AssertionError, match=r"no structural contract for behavior 'PROPAGATES'"):
            rungs.check_behavior_nan(wrong)


# ======================================================================================================================
# The SERIES IN_WINDOW_IS_NULL shape
# ======================================================================================================================


class TestInWindowShape:
    """A rolling null nulls every window it overlaps, then the lane recovers to the clean run within tolerance."""

    def test_in_window_null_green(self) -> None:
        """A rolling IN_WINDOW_IS_NULL metric passes on the interior-null probe (tolerance tail, one-pass residue)."""
        rungs.check_behavior_null(VOLATILITY_ROLLING)

    def test_in_window_nan_propagates_green(self) -> None:
        """A rolling PROPAGATES-NaN metric passes on the interior-NaN flow probe."""
        rungs.check_behavior_nan(VOLATILITY_ROLLING)

    def test_in_window_declared_propagates_red(self) -> None:
        """An IN_WINDOW metric wrongly declared PROPAGATES goes red — a whole window it overlaps stays null."""
        wrong = dataclasses.replace(VOLATILITY_ROLLING, behavior_null=BehaviorNull.PROPAGATES)
        with pytest.raises(AssertionError, match=r"beyond the horizon .* the lane must equal the clean run"):
            rungs.check_behavior_null(wrong)

    def test_struct_flow_is_fail_closed(self) -> None:
        """A STRUCT flow is still fail-closed — the structural contract lands with the indicators family."""

        def two_field_struct(equity_curve: pl.Expr) -> pl.Expr:
            return pl.struct(a=equity_curve, b=equity_curve * 2.0)

        def reference_two_field_struct(equity_curve: Sequence[float | None]) -> dict[str, list[float | None]]:
            return {
                "a": list(equity_curve),
                "b": [None if v is None else v * 2.0 for v in equity_curve],
            }

        struct_decl = Declaration(
            family="metrics",
            factory=two_field_struct,
            inputs=("equity_curve",),
            params={},
            shape=Shape.STRUCT,
            fields=("a", "b"),
            behavior_null=BehaviorNull.PROPAGATES,
            behavior_nan=BehaviorNan.PROPAGATES,
            space=_Space.RETURNS,
            sign=_Sign.LONG_SHORT,
            nonfinite=_NonFinite.IEEE_FLOW,
            oracle=reference_two_field_struct,
            scaling=(ScaleAxis(roles=("equity_curve",), degree={"a": 1, "b": 1}),),
        )
        with pytest.raises(NotImplementedError, match=r"structural flow contracts for Shape.STRUCT"):
            rungs.check_behavior_null(struct_decl)


# ======================================================================================================================
# Twin coherence
# ======================================================================================================================


class TestTwinCoherence:
    """A rolling row equals its twin reduced over the trailing window — the reducing twin's value, the series' last."""

    def test_reducing_twin_green(self) -> None:
        """volatility_rolling row i equals volatility over the trailing window (a reducing twin's single value)."""
        rungs.check_twin_coherence(VOLATILITY_ROLLING)

    def test_series_twin_green(self) -> None:
        """drawdown_rolling row i equals drawdown over the trailing window (a series twin's last value)."""
        rungs.check_twin_coherence(DRAWDOWN_ROLLING)

    def test_non_rolling_skips(self) -> None:
        """A non-rolling declaration skips the coherence rung cleanly."""
        with pytest.raises(pytest.skip.Exception, match=r"no rolling twin declared"):
            rungs.check_twin_coherence(VOLATILITY)

    def test_wrong_twin_red(self) -> None:
        """volatility_rolling declared to roll the wrong (returns-input) twin goes red — the rows do not agree."""
        detached = dataclasses.replace(VOLATILITY_ROLLING, rolling_of=WRONG_RETURNS_TWIN)
        with pytest.raises(AssertionError, match=r"check_twin_coherence disagreed"):
            rungs.check_twin_coherence(detached)


# ======================================================================================================================
# Annualization
# ======================================================================================================================


class TestAnnualization:
    """Where the annualization is closed-form the period-count ratio is exact; otherwise the rung skips or fails."""

    def test_sqrt_time_green(self) -> None:
        """A SQRT_TIME metric passes: factory(P=252)/factory(P=63) == sqrt(252/63) == 2."""
        rungs.check_annualization(VOLATILITY)

    def test_linear_green(self) -> None:
        """A LINEAR metric passes: factory(P=252)/factory(P=63) == 252/63 == 4."""
        rungs.check_annualization(TREYNOR_RATIO)

    def test_geometric_skips(self) -> None:
        """A GEOMETRIC metric skips — no closed-form ratio, the oracle carries the convention."""
        with pytest.raises(pytest.skip.Exception, match=r"GEOMETRIC has no closed-form annualization ratio"):
            rungs.check_annualization(CAGR)

    def test_none_skips(self) -> None:
        """A NONE metric skips — it is not annualized."""
        with pytest.raises(pytest.skip.Exception, match=r"NONE has no closed-form annualization ratio"):
            rungs.check_annualization(MAX_DRAWDOWN)

    def test_undeclared_skips(self) -> None:
        """A declaration carrying no annualization skips the rung cleanly."""
        with pytest.raises(pytest.skip.Exception, match=r"no annualization declared"):
            rungs.check_annualization(TOTAL_RETURN)

    def test_unknown_member_is_fail_closed(self) -> None:
        """An annualization member with no closed-form contract is fail-closed."""

        class _BadAnnual(enum.Enum):
            WEEKLY = "weekly"

        bogus = dataclasses.replace(VOLATILITY, annualization=_BadAnnual.WEEKLY)
        with pytest.raises(NotImplementedError, match=r"no closed-form annualization contract for 'WEEKLY'"):
            rungs.check_annualization(bogus)

    def test_sqrt_time_declared_linear_red(self) -> None:
        """A SQRT_TIME metric wrongly declared LINEAR goes red — the ratio is 2, not 4."""
        wrong = dataclasses.replace(VOLATILITY, annualization=Annualization.LINEAR)
        with pytest.raises(AssertionError, match=r"LINEAR annualization at row .* != 4.0"):
            rungs.check_annualization(wrong)


# ======================================================================================================================
# The rolling-twin constructor guards, and the conditioning pairing guard on a metrics declaration
# ======================================================================================================================


class TestRollingGuards:
    """The declaration constructor's rolling-twin and conditioning guards bite on a metrics-shaped counterexample."""

    def test_twin_without_window_rejected(self) -> None:
        """A rolling twin that names no window parameter dies at construction."""
        with pytest.raises(ValueError, match=r"a rolling twin .* must name its window parameter"):
            dataclasses.replace(VOLATILITY_ROLLING, window=None)

    def test_window_naming_a_missing_param_rejected(self) -> None:
        """A window field naming a parameter that is not in params dies at construction."""
        with pytest.raises(ValueError, match=r"window='lookback' must name a parameter in params"):
            dataclasses.replace(VOLATILITY_ROLLING, window="lookback")

    def test_twin_with_mismatched_inputs_rejected(self) -> None:
        """A rolling twin whose inputs differ from its twin's dies at construction — the window slice drives both."""
        with pytest.raises(ValueError, match=r"a rolling twin shares its twin's inputs"):
            dataclasses.replace(VOLATILITY_ROLLING, rolling_of=TREYNOR_RATIO)

    def test_conditioning_without_covering_pin_rejected(self) -> None:
        """A metrics declaration with a conditioning filter but no witnessing pin dies at construction."""
        with pytest.raises(ValueError, match=r"no exclusion without a fixed case"):
            dataclasses.replace(VOLATILITY_ROLLING, conditioning=_always)

    def test_conditioning_paired_with_covering_pin_constructs(self) -> None:
        """A conditioning filter paired with its covering pin is accepted — the pairing the guard requires."""
        paired = dataclasses.replace(
            VOLATILITY_ROLLING,
            conditioning=_always,
            pins=(
                Pin(
                    label="degenerate_window",
                    inputs={"returns": (0.5, 0.5, 0.5)},
                    expected=(None, None, 0.0),
                    reason="a constant trailing window has zero dispersion — the regime the filter excludes",
                    covers_conditioning=True,
                ),
            ),
        )
        assert paired.conditioning is not None


def test_reproduction_snippet_shape() -> None:
    """A sanity check that the metrics declarations build a probe frame the rungs can read."""
    frame = probe_frame(VOLATILITY_ROLLING.inputs, 14)
    assert frame.height == 14
    assert re.fullmatch(r"volatility_rolling", VOLATILITY_ROLLING.name)
