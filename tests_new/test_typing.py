"""
Return-annotation contract for every public factory, derived from the registry.

One guard remains of the old per-family typing modules: every public factory must declare its return type as
exactly ``pl.Expr``. Everything else those modules checked is guaranteed elsewhere by construction. The runtime
"builds a ``pl.Expr``" sweep is the ladder's ``test_returns_expr`` (over ``ALL_SPECS``, held to the public surface
by the registry bijection). A factory whose annotation drifts BROADER (e.g. ``pl.Expr | None``) is rejected
statically by the spec language itself: ``Spec.factory`` is typed ``Callable[..., pl.Expr]``, so the type checkers
flag the drifting factory's own spec file (verified on a leaf function whose ``src`` module alone stays clean).
What neither path pins is exact identity — the covariant ``Callable`` accepts a subclass silently, while the
public contract is one predictable expression type — so this module holds that last claim, registry-derived so a
newly added function is swept in the moment its spec lands. No ``assert_type`` blocks live here: the package
declares no overloads, so the inferred return type IS the declared annotation, and one sweep over the declaration
covers what dozens of literal call sites used to restate.
"""

from typing import get_type_hints

import polars as pl
import pytest
from tests_new.all_specs import ALL_SPECS
from tests_new.support.spec import Spec, spec_id


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_return_annotation_is_expr(spec: Spec) -> None:
    """Verifies every public factory declares exactly ``pl.Expr`` as its return type."""
    assert get_type_hints(spec.factory).get("return") is pl.Expr, f"{spec.name}: return annotation is not pl.Expr"
