"""
Consumer-side typing conformance for the public metric factories.

Three complementary guards. ``test_every_public_factory_builds_an_expr`` is parametrised over
``pomata.metrics.__all__``: it calls every public factory with minimal valid arguments (synthesized from the signature,
which fills the required keyword-only ``periods_per_year``) and asserts the result is a ``pl.Expr`` at runtime.
``test_every_public_factory_is_annotated_as_expr`` is parametrised the same way and asserts each factory's *declared*
return type is exactly ``pl.Expr``, extending the return-type contract to the whole public surface — so a metric whose
annotation drifted off ``pl.Expr`` (a silent failure the runtime sweep would miss) is caught. Both CANNOT go stale,
since a newly added metric is swept in automatically. ``test_factories_are_typed_as_expr`` then pins the *inferred*
static contract with ``typing.assert_type`` (a compile-time assertion, runtime no-op) across the shape variants, so the
type checkers (mypy ``--strict``, pyright, ty) confirm the inferred return type is ``pl.Expr``. The ``assert_type``
calls must be literal (they pin what the checker *infers*), so the family is spelled out, not parametrized.
"""

from typing import assert_type, get_type_hints

import polars as pl
import pytest
from tests_new.support import synthesize_call

from pomata import metrics
from pomata.metrics import (
    alpha_rolling,
    beta_rolling,
    cagr,
    cagr_rolling,
    calmar_ratio,
    conditional_value_at_risk,
    downside_deviation,
    downside_deviation_rolling,
    drawdown,
    drawdown_rolling,
    information_ratio_rolling,
    kurtosis,
    kurtosis_rolling,
    max_drawdown,
    omega_ratio,
    omega_ratio_rolling,
    sharpe_ratio,
    sharpe_ratio_rolling,
    skewness,
    skewness_rolling,
    sortino_ratio,
    sortino_ratio_rolling,
    tail_ratio,
    tail_ratio_rolling,
    total_return,
    total_return_rolling,
    treynor_ratio_rolling,
    ulcer_index,
    ulcer_performance_ratio,
    value_at_risk,
    value_at_risk_rolling,
    volatility,
    volatility_rolling,
)


@pytest.mark.parametrize("name", metrics.__all__)
def test_every_public_factory_builds_an_expr(name: str) -> None:
    """
    Verifies that every public factory in ``__all__``, called with minimal valid arguments, builds a ``pl.Expr``.

    The call is synthesized from the signature (see :func:`tests_new.support.synthesize_call`), so parametrizing over
    ``__all__`` keeps coverage in lock-step with the public API: a newly added metric is swept in automatically.
    """
    factory = getattr(metrics, name)
    positional, keywords = synthesize_call(factory)
    assert isinstance(factory(*positional, **keywords), pl.Expr)


@pytest.mark.parametrize("name", metrics.__all__)
def test_every_public_factory_is_annotated_as_expr(name: str) -> None:
    """
    Verifies every public factory declares ``pl.Expr`` as its return type, exhaustively across ``__all__``.

    Parametrizing over ``__all__`` keeps the return-type contract in lock-step with the public API: a newly added
    metric is swept in automatically. A factory whose return annotation drifted off ``pl.Expr`` — the silent failure
    the runtime ``isinstance`` sweep would miss — fails here.
    """
    assert get_type_hints(getattr(metrics, name)).get("return") is pl.Expr


def test_factories_are_typed_as_expr() -> None:
    """
    Verifies the public metric factories are statically inferred as returning ``pl.Expr``.

    Each ``assert_type`` covers one factory (return-based metrics take a return series, equity-based metrics a
    growth-factor series; the windowed twins additionally take a positional ``int`` window), so the checkers confirm the
    return type across the whole family. The body is also exercised at runtime by pytest, which fails if any call no
    longer builds against the current Polars API.
    """
    returns = pl.col("returns")
    equity = pl.col("equity")
    benchmark = pl.col("benchmark")
    window = 20

    assert_type(volatility(returns, periods_per_year=252), pl.Expr)
    assert_type(downside_deviation(returns, periods_per_year=252), pl.Expr)
    assert_type(skewness(returns), pl.Expr)
    assert_type(kurtosis(returns), pl.Expr)
    assert_type(value_at_risk(returns, confidence=0.95), pl.Expr)
    assert_type(conditional_value_at_risk(returns, confidence=0.95), pl.Expr)
    assert_type(tail_ratio(returns), pl.Expr)
    assert_type(omega_ratio(returns), pl.Expr)
    assert_type(sharpe_ratio(returns, periods_per_year=252), pl.Expr)
    assert_type(sortino_ratio(returns, periods_per_year=252), pl.Expr)
    assert_type(total_return(equity), pl.Expr)
    assert_type(cagr(equity, periods_per_year=252), pl.Expr)
    assert_type(drawdown(equity), pl.Expr)
    assert_type(max_drawdown(equity), pl.Expr)
    assert_type(ulcer_index(equity), pl.Expr)
    assert_type(calmar_ratio(equity, periods_per_year=252), pl.Expr)
    assert_type(ulcer_performance_ratio(equity, periods_per_year=252), pl.Expr)

    assert_type(volatility_rolling(returns, window, periods_per_year=252), pl.Expr)
    assert_type(downside_deviation_rolling(returns, window, periods_per_year=252), pl.Expr)
    assert_type(skewness_rolling(returns, window), pl.Expr)
    assert_type(kurtosis_rolling(returns, window), pl.Expr)
    assert_type(value_at_risk_rolling(returns, window, confidence=0.95), pl.Expr)
    assert_type(tail_ratio_rolling(returns, window), pl.Expr)
    assert_type(omega_ratio_rolling(returns, window), pl.Expr)
    assert_type(sharpe_ratio_rolling(returns, window, periods_per_year=252), pl.Expr)
    assert_type(sortino_ratio_rolling(returns, window, periods_per_year=252), pl.Expr)
    assert_type(total_return_rolling(equity, window), pl.Expr)
    assert_type(cagr_rolling(equity, window, periods_per_year=252), pl.Expr)
    assert_type(drawdown_rolling(equity, window), pl.Expr)
    assert_type(beta_rolling(returns, benchmark, window), pl.Expr)
    assert_type(alpha_rolling(returns, benchmark, window, periods_per_year=252), pl.Expr)
    assert_type(information_ratio_rolling(returns, benchmark, window, periods_per_year=252), pl.Expr)
    assert_type(treynor_ratio_rolling(returns, benchmark, window, periods_per_year=252), pl.Expr)
