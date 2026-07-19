"""
The pnl family suite function: the one call a pnl declaration file makes.

``suite_pnl`` is the contributor's whole interface to the family. It takes the family enums plus a naive oracle, fixes
the facts the family shares by construction (the output is always a same-length ``Float64`` series, the family is
``pnl``), resolves the family's :class:`~tests.pnl.enums.Warmup` enum to the leading-null ``int`` the generic
:class:`~tests.support.declaration.Declaration` and its rungs speak, assembles that declaration, registers it as a
side effect, and returns it — so ``COST_BORROW = suite_pnl(...)`` both binds the module constant and enrolls it in the
pnl registry the ladder parametrizes over.
"""

from collections.abc import Mapping
from types import MappingProxyType

from tests.pnl.enums import BehaviorNan, BehaviorNull, ConventionSign, NonFinite, SpaceCost, Warmup
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

# The pnl family's warm-up dialect is a two-valued enum; the generic declaration speaks the resolved leading-null count
# (an ``int`` or ``None``), so the harness translates before it constructs — no rung ever sees a family enum.
_WARMUP_ROWS: dict[Warmup, int | None] = {Warmup.NONE: None, Warmup.ONE_ROW: 1}


# A family suite states every declaration axis by name, so its signature is deliberately wide; a bag-of-kwargs would
# throw away the type-checked, autocompleted contract that is the whole point of the closed family vocabulary.
def suite_pnl(  # noqa: PLR0913
    *,
    factory: FactoryExpr,
    inputs: tuple[str, ...],
    params: Mapping[str, ScalarParam],
    null: BehaviorNull,
    nan: BehaviorNan,
    space: SpaceCost,
    sign: ConventionSign,
    oracle: OracleFn,
    scaling: tuple[ScaleAxis, ...] | ScaleExempt,
    warmup: Warmup = Warmup.NONE,
    nonfinite: NonFinite = NonFinite.IEEE_FLOW,
    raises: tuple[tuple[Mapping[str, ScalarParam], str], ...] = (),
    golden: Golden | None = None,
    pins: tuple[Pin, ...] = (),
    deviant: Deviant | None = None,
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
    Build, register, and return the :class:`Declaration` for one public ``pomata.pnl`` function.

    The family-fixed facts are supplied here: ``family="pnl"`` and ``shape=Shape.SERIES`` (every pnl function returns a
    same-length ``Float64`` column, never a struct or a reduction). Everything else is stated by the caller in the
    family dialect; the ``Warmup`` enum is resolved to the leading-null count the generic declaration stores.

    Args:
        factory: The ``pl.Expr`` factory under test; its ``__name__`` is the declaration's name and the key everything
            (the pytest id, the registry, the required oracle name) derives from.
        inputs: The ordered input column roles, drawn from the probe-frame vocabulary.
        params: The default keyword arguments the factory is exercised under (non-empty implies ``raises``).
        null: The interior-``null`` behavior (the pnl dialect).
        nan: The interior-``NaN`` behavior (the pnl dialect).
        space: The units the output lives in (cash flow vs returns flow).
        sign: The sign convention the payoff follows.
        oracle: The naive reference, named ``reference_{name}``, mirroring the factory signature.
        scaling: A non-empty tuple of homogeneity axes, or a ``ScaleExempt`` for a scale-exempt function.
        warmup: The family warm-up enum, resolved to a leading-null ``int`` / ``None`` before the declaration is built.
        nonfinite: How the function carries ``±inf`` inputs; the family shares one IEEE-754 flow.
        raises: Validation counterexamples, each ``(kwargs overriding params, the ValueError match regex)``.
        golden: The frozen golden master (the recommended hand-computed anchor), or ``None``.
        pins: Crafted-input cases for exact values the synthesis and the oracle cannot derive.
        deviant: The documented answer to the all-null regime, or ``None`` for the ordinary all-null answer.
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
        The registered declaration, so ``FOO = suite_pnl(...)`` both binds and enrolls it.
    """
    declaration = Declaration(
        family="pnl",
        factory=factory,
        inputs=inputs,
        params=params,
        shape=Shape.SERIES,
        behavior_null=null,
        behavior_nan=nan,
        space=space,
        sign=sign,
        nonfinite=nonfinite,
        oracle=oracle,
        scaling=scaling,
        warmup=_WARMUP_ROWS[warmup],
        raises=raises,
        golden=golden,
        pins=pins,
        deviant=deviant,
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
