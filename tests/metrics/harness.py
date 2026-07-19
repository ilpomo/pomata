"""
The metrics family suite function: the one call a metrics declaration file makes.

``suite_metrics`` is the contributor's whole interface to the family. It takes the family enums plus a naive oracle,
fixes the fact the family shares by construction (the family is ``metrics``), derives the output shape (a reduction
unless the function rolls a twin or names a window), assembles the generic
:class:`~tests.support.declaration.Declaration`, registers it as a side effect, and returns it — so
``SHARPE_RATIO = suite_metrics(...)`` both binds the module constant and enrolls it in the metrics registry the ladder
parametrizes over.

The metrics family does not answer the pnl dialect's cost-space, sign-convention, or non-finite axes — a metric is a
statistic of a return or equity series, not a signed cash flow, and its reduction over an infinite input is an
implementation-defined artifact the naive oracle does not model — so it leaves those three declaration fields unset.

A rolling twin (``rolling_of=...``) inherits its twin's annualization and degenerate regime unless it states its own:
those two axes are the same statistic per trailing window, so the window form never redeclares them. Its
missing-value behavior is stated on the twin every time (a reduction skips an interior null; a rolling window it
overlaps is null), so ``null`` / ``nan`` are always given explicitly.
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import cast

import polars as pl

from tests.metrics.enums import Annualization, BehaviorNan, BehaviorNull, Degenerate
from tests.support.declaration import (
    Declaration,
    Deviant,
    Example,
    FactoryExpr,
    Golden,
    OracleFn,
    Pin,
    ScalarParam,
    ScaleAxis,
    ScaleExempt,
    Shape,
)
from tests.support.registry import register


def _derive_shape(*, rolling_of: Declaration | None, window: str | None) -> Shape:
    """A metric reduces to one scalar unless it rolls a twin or names a window — then it is a same-length series."""
    return Shape.SERIES if (rolling_of is not None or window is not None) else Shape.REDUCING


# A family suite states every declaration axis by name, so its signature is deliberately wide; a bag-of-kwargs would
# throw away the type-checked, autocompleted contract that is the whole point of the closed family vocabulary.
def suite_metrics(  # noqa: PLR0913
    *,
    factory: FactoryExpr,
    inputs: tuple[str, ...],
    params: Mapping[str, ScalarParam],
    null: BehaviorNull,
    nan: BehaviorNan,
    oracle: OracleFn,
    scaling: tuple[ScaleAxis, ...] | ScaleExempt,
    shape: Shape | None = None,
    annualization: Annualization | None = None,
    degenerate: Degenerate | None = None,
    warmup: int | None = None,
    rolling_of: Declaration | None = None,
    window: str | None = None,
    raises: tuple[tuple[Mapping[str, ScalarParam], str], ...] = (),
    golden: Golden | None = None,
    pins: tuple[Pin, ...] = (),
    recomposition: Callable[[], pl.Expr] | None = None,
    deviant: Deviant | None = None,
    conditioning: Callable[[pl.DataFrame], bool] | None = None,
    oracle_rel_tol: float | None = None,
    oracle_abs_tol: float | None = None,
    reference: str = "",
    doi: str = "",
    wikipedia: str = "",
    reference_url: str = "",
    see_also: tuple[tuple[str, str], ...] = (),
    notes: tuple[tuple[str, str], ...] = (),
    opener_override: str = "",
    note_extension: str = "",
    bullets: tuple[tuple[str, str], ...] = (),
    note_postscript: str = "",
    returns_body: str = "",
    raises_prose: str = "",
    args_prose: Mapping[str, str] = MappingProxyType({}),
    example_alias: str = "",
    example_imports: tuple[str, ...] = (),
    intro_basic: str = "",
    example_columns: Mapping[str, str] = MappingProxyType({}),
    examples: tuple[Example, ...] = (),
) -> Declaration:
    """
    Build, register, and return the :class:`Declaration` for one public ``pomata.metrics`` function.

    The family-fixed fact is supplied here: ``family="metrics"``. The pnl dialect's space / sign / non-finite axes are
    left unset (see the module docstring). The output ``shape`` is derived — a reduction unless the function rolls a
    twin or names a window — unless the caller states it (the plain series ``drawdown`` is neither, so it passes
    ``shape=Shape.SERIES``). A rolling twin inherits its twin's ``annualization`` and ``degenerate`` when unstated.

    Args:
        factory: The ``pl.Expr`` factory under test; its ``__name__`` is the declaration's name and the key everything
            (the pytest id, the registry, the required oracle name) derives from.
        inputs: The ordered input column roles, drawn from the probe-frame vocabulary.
        params: The default keyword arguments the factory is exercised under (non-empty implies ``raises``).
        null: The interior-``null`` behavior (the metrics dialect: SKIPPED, IN_WINDOW_IS_NULL, or PROPAGATES).
        nan: The interior-``NaN`` behavior (the metrics dialect: POISONS or PROPAGATES).
        oracle: The naive reference, named ``reference_{name}``, mirroring the factory signature.
        scaling: A non-empty tuple of homogeneity axes, or a ``ScaleExempt`` for a scale-exempt function.
        shape: The output shape, or ``None`` to derive it from ``rolling_of`` / ``window``.
        annualization: The annualization convention, or ``None`` to inherit a rolling twin's (else no annualization).
        degenerate: The degenerate-denominator regime, or ``None`` to inherit a rolling twin's (else none declared).
        warmup: The exact leading-null count for a windowed series, or ``None`` for a reduction / unwindowed series.
        rolling_of: The reducing or series twin this function rolls per trailing window, or ``None``.
        window: The name of the window-length parameter (a key in ``params``); required with ``rolling_of``.
        raises: Validation counterexamples, each ``(kwargs overriding params, the ValueError match regex)``.
        golden: The frozen golden master (the recommended hand-computed anchor), or ``None``.
        pins: Crafted-input cases for exact values the synthesis and the oracle cannot derive.
        recomposition: A zero-argument callable returning the ``pl.Expr`` that rebuilds this metric from other public
            functions (a ratio as its numerator over its denominator), or ``None`` when there is no such identity.
        deviant: The documented answer to the all-null regime, or ``None`` for the ordinary all-null answer.
        conditioning: An optional Hypothesis filter excluding an ill-conditioned input regime (paired with a covering
            pin), or ``None``.
        oracle_rel_tol: The oracle-agreement relative band override (a one-pass rolling form vs its two-pass oracle).
        oracle_abs_tol: The oracle-agreement absolute band override.
        reference: The literature citation line for the definition (author, year, title), or empty.
        doi: The DOI URL for the reference, or empty.
        wikipedia: The encyclopedic (Wikipedia) reference URL, or empty.
        reference_url: A reference URL that is neither a DOI nor a Wikipedia page (a methodology page), or empty.
        see_also: The See Also entries, each a ``(public-function name, one-line clause)`` pair.
        notes: The pre-list Note subheaders, each a ``(label, body)`` pair.
        opener_override: A per-function replacement of the whole Note opener body, or empty for the family template.
        note_extension: The per-function extension of the Note opener body, beyond the family template, or empty.
        bullets: The Edge-case behavior bullets, each a ``(label, body)`` pair, in source order.
        note_postscript: A Note paragraph after the Edge-case list, for the one function whose Note trails it, or empty.
        returns_body: The Returns section body, verbatim.
        raises_prose: The Raises ValueError clause, verbatim (the TypeError line is the shared template).
        args_prose: Per-parameter Args descriptions overriding the mined majority template, keyed by parameter name.
        example_alias: The Examples import alias (``as ...``), or empty for the bare function name.
        example_imports: Extra Examples imports beyond ``import polars as pl``, each a full import statement.
        intro_basic: The optional prose line opening the whole Examples block, or empty.
        example_columns: The display column name each input role is shown under (a role absent uses its own name).
        examples: The Examples scenarios in source order, each rendered in the canonical idiom and executed.

    Returns:
        The registered declaration, so ``FOO = suite_metrics(...)`` both binds and enrolls it.
    """
    if rolling_of is not None:
        if annualization is None:
            annualization = cast("Annualization | None", rolling_of.annualization)
        if degenerate is None:
            degenerate = cast("Degenerate | None", rolling_of.degenerate)
    declaration = Declaration(
        family="metrics",
        factory=factory,
        inputs=inputs,
        params=params,
        shape=_derive_shape(rolling_of=rolling_of, window=window) if shape is None else shape,
        behavior_null=null,
        behavior_nan=nan,
        oracle=oracle,
        scaling=scaling,
        warmup=warmup,
        annualization=annualization,
        degenerate=degenerate,
        rolling_of=rolling_of,
        window=window,
        raises=raises,
        golden=golden,
        pins=pins,
        recomposition=recomposition,
        deviant=deviant,
        conditioning=conditioning,
        oracle_rel_tol=oracle_rel_tol,
        oracle_abs_tol=oracle_abs_tol,
        reference=reference,
        doi=doi,
        wikipedia=wikipedia,
        reference_url=reference_url,
        see_also=see_also,
        notes=notes,
        opener_override=opener_override,
        note_extension=note_extension,
        bullets=bullets,
        note_postscript=note_postscript,
        returns_body=returns_body,
        raises_prose=raises_prose,
        args_prose=args_prose,
        example_alias=example_alias,
        example_imports=example_imports,
        intro_basic=intro_basic,
        example_columns=example_columns,
        examples=examples,
    )
    return register(declaration)
