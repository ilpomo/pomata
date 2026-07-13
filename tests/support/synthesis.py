"""
Signature-driven call synthesis for the public factories.

Builds a minimal valid call for any public factory from its signature alone (``pl.Expr`` -> the column
:data:`tests.support.columns.COLUMN_X`, ``int`` -> a small window, ``float`` -> a small factor), so a test
parametrized over a package's ``__all__`` can invoke every factory without a hand-maintained argument table: a newly
added factory is swept in automatically and cannot drift out of coverage.
"""

import inspect
from collections.abc import Callable

import polars as pl
from tests.support.columns import COLUMN_X


def sample_argument(annotation: object) -> object:
    """
    A minimal valid value for a required parameter of the given annotation.

    ``pl.Expr`` -> a reference to :data:`tests.support.columns.COLUMN_X`; ``int`` -> a small window; ``float`` -> a
    small factor. Any other annotation raises, so an unsupported new parameter type fails loudly rather than silently
    skewing coverage.

    Raises:
        AssertionError: If ``annotation`` is not ``pl.Expr``, ``int``, or ``float``.
    """
    if annotation is pl.Expr:
        return pl.col(COLUMN_X)
    if annotation is int:
        return 3
    if annotation is float:
        return 0.1
    raise AssertionError(f"cannot synthesize a value for a required parameter annotated {annotation!r}")


def synthesize_call(factory: Callable[..., pl.Expr]) -> tuple[list[object], dict[str, object]]:
    """
    Build a minimal ``(positional, keywords)`` call for ``factory`` from its signature.

    Required positional / positional-or-keyword parameters are synthesized by position and required keyword-only
    parameters by name (a small equal window for each, which satisfies the ``window_fast <= window_slow`` style
    ordering contracts); parameters that already carry a default are left to it.
    """
    parameters = inspect.signature(factory, eval_str=True).parameters.values()
    positional = [
        sample_argument(parameter.annotation)
        for parameter in parameters
        if parameter.default is inspect.Parameter.empty
        and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    keywords = {
        parameter.name: sample_argument(parameter.annotation)
        for parameter in parameters
        if parameter.default is inspect.Parameter.empty and parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    return positional, keywords
