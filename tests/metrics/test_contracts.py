"""
Universal structural contract for the public metrics factories -- the rungs identical for every function.

Every metric owes the same six structural guarantees; rather than copy them into all 60 test modules, this contract
parametrizes over ``metrics.__all__`` and applies the shared, shape-aware
assertions in :mod:`tests.support.contracts`. Metrics share one rung the other families keep per-file: ``.over`` is
universal here (every metric -- reducing or rolling -- partitions per group), and ``all_null`` covers every metric,
rolling ones included (``sharpe_ratio_rolling``, ``beta_rolling``, ...); a new metric is swept in automatically.

The function-specific rungs stay in each function's own file: the validation raises, the null / NaN policy, correctness
(golden masters), and the property tiers. Dtype and name-preservation stay in ``test_dtype.py`` / ``test_naming.py``.
"""

import pytest
from tests.support import contracts

from pomata import metrics

_METRICS = sorted(metrics.__all__)


@pytest.mark.parametrize("name", _METRICS)
def test_returns_expr(name: str) -> None:
    """Verifies the factory returns a ``pl.Expr`` without touching a frame."""
    contracts.assert_returns_expr(getattr(metrics, name))


@pytest.mark.parametrize("name", _METRICS)
def test_shape(name: str) -> None:
    """Verifies the output has a coherent shape (scalar / series), at ``Float64``."""
    contracts.assert_shape(getattr(metrics, name))


@pytest.mark.parametrize("name", _METRICS)
def test_lazy_eager_parity(name: str) -> None:
    """Verifies eager and lazy application produce identical materialized output."""
    contracts.assert_lazy_eager_parity(getattr(metrics, name))


@pytest.mark.parametrize("name", _METRICS)
def test_over_partitions_independently(name: str) -> None:
    """Verifies that under ``.over`` the metric is computed per group and never spans a boundary."""
    contracts.assert_over_partitions(getattr(metrics, name))


@pytest.mark.parametrize("name", _METRICS)
def test_empty(name: str) -> None:
    """Verifies an empty series yields an empty result (a null scalar for a reducing metric)."""
    contracts.assert_empty(getattr(metrics, name))


@pytest.mark.parametrize("name", _METRICS)
def test_all_null(name: str) -> None:
    """Verifies an all-null series stays all-null (per the declared shape)."""
    contracts.assert_all_null(getattr(metrics, name))
