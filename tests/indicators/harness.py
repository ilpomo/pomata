"""
The indicators family suite function: the one call an indicator declaration file makes.

``suite_indicators`` is the contributor's whole interface to the family. It takes the family enums plus a naive
oracle, fixes the fact the family shares by construction (the family is ``indicators``), resolves the family's
:class:`~tests.indicators.enums.Warmup` enum to the leading-null ``int`` / mapping / ``None`` the generic
:class:`~tests.support.declaration.Declaration` and its rungs speak, assembles that declaration, registers it as
a side effect, and returns it — so ``SMA = suite_indicators(...)`` both binds the module constant and enrolls it in
the indicators registry the ladder parametrizes over.

One family fact is stored but not read by any rung — :class:`~tests.indicators.enums.Seeding` (docstring
metadata) — while :class:`~tests.indicators.enums.RelationTalib` and its reason drive the differential tier's
partition of the public surface. The escape hatches (``flow_deviation`` for the directional-movement pair,
``flow_horizon`` for a long contracting recursion) are passed through unchanged.
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType

import polars as pl

from tests.indicators.enums import BehaviorNan, BehaviorNull, RelationTalib, Seeding, Warmup
from tests.support.declaration import (
    Declaration,
    Deviant,
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


def _resolve_warmup(
    warmup: Warmup,
    warmup_value: int | Mapping[str, int] | None,
    params: Mapping[str, ScalarParam],
    window: str,
    name: str,
) -> int | Mapping[str, int] | None:
    """Resolve the family ``Warmup`` enum to the leading-null count the generic declaration stores.

    ``WINDOW`` / ``WINDOW_MINUS_ONE`` read the named window parameter; ``EXPR`` / ``PER_FIELD`` carry an explicit
    ``warmup_value`` (a composite count no single window derives); ``NONE`` is a windowless transform. The forms that
    take a value and the forms that compute one are mutually exclusive, checked here so a mistaken pairing is loud.
    """
    computed = warmup in (Warmup.WINDOW, Warmup.WINDOW_MINUS_ONE)
    explicit = warmup in (Warmup.EXPR, Warmup.PER_FIELD)
    if (computed or warmup is Warmup.NONE) and warmup_value is not None:
        msg = f"{name}: warmup={warmup.name} computes its count and takes no warmup_value"
        raise ValueError(msg)
    if explicit and warmup_value is None:
        msg = f"{name}: warmup={warmup.name} needs an explicit warmup_value"
        raise ValueError(msg)
    if warmup is Warmup.NONE:
        return None
    if computed and window not in params:
        msg = f"{name}: warmup={warmup.name} reads params[{window!r}], which is absent from {sorted(params)}"
        raise ValueError(msg)
    if warmup is Warmup.WINDOW_MINUS_ONE:
        return int(params[window]) - 1
    if warmup is Warmup.WINDOW:
        return int(params[window])
    return warmup_value  # EXPR (int) or PER_FIELD (mapping)


# A family suite states every declaration axis by name, so its signature is deliberately wide; a bag-of-kwargs would
# throw away the type-checked, autocompleted contract that is the whole point of the closed family vocabulary.
def suite_indicators(  # noqa: PLR0913
    *,
    factory: FactoryExpr,
    inputs: tuple[str, ...],
    params: Mapping[str, ScalarParam],
    null: BehaviorNull,
    nan: BehaviorNan,
    shape: Shape,
    oracle: OracleFn,
    scaling: tuple[ScaleAxis, ...] | ScaleExempt,
    talib: RelationTalib,
    warmup: Warmup = Warmup.NONE,
    warmup_value: int | Mapping[str, int] | None = None,
    window: str = "window",
    seeding: Seeding = Seeding.NONE,
    talib_reason: str = "",
    fields: tuple[str, ...] = (),
    raises: tuple[tuple[Mapping[str, ScalarParam], str], ...] = (),
    golden: Golden | None = None,
    pins: tuple[Pin, ...] = (),
    recomposition: Callable[[], pl.Expr] | None = None,
    deviant: Deviant | None = None,
    conditioning: Callable[[pl.DataFrame], bool] | None = None,
    flow_deviation: str = "",
    flow_horizon: int = -1,
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
    intro_over: str = "",
    intro_missing: str = "",
) -> Declaration:
    """
    Build, register, and return the :class:`Declaration` for one public ``pomata.indicators`` function.

    The family-fixed fact is supplied here: ``family="indicators"``. The output ``shape`` is stated by the caller
    (SERIES or STRUCT, a struct naming its ``fields``). The ``Warmup`` enum is resolved to the leading-null count the
    generic declaration stores; ``RelationTalib`` and its ``talib_reason`` are stored for the differential tier.

    Args:
        factory: The ``pl.Expr`` factory under test; its ``__name__`` is the declaration's name and the key everything
            (the pytest id, the registry, the required oracle name) derives from.
        inputs: The ordered input column roles, drawn from the probe-frame vocabulary.
        params: The default keyword arguments the factory is exercised under (non-empty implies ``raises``).
        null: The interior-``null`` behavior (the indicators dialect).
        nan: The interior-``NaN`` behavior (the indicators dialect).
        shape: The output shape (SERIES or STRUCT; a struct also names ``fields``).
        oracle: The naive reference, named ``reference_{name}``, mirroring the factory signature.
        scaling: A non-empty tuple of homogeneity axes, or a ``ScaleExempt`` for a scale-exempt function.
        talib: The TA-Lib relation (MATCHES / DOCUMENTED_DIVERGENCE / NO_EQUIVALENT), read by the differential tier.
        warmup: The family warm-up enum, resolved to a leading-null ``int`` / mapping / ``None``.
        warmup_value: The explicit warm-up for the EXPR (int) / PER_FIELD (mapping) forms; ``None`` for the others.
        window: The name of the window parameter the WINDOW / WINDOW_MINUS_ONE forms read (default ``"window"``).
        seeding: The recursion's seeding convention — docstring metadata, not read by any rung.
        talib_reason: The reason a DOCUMENTED_DIVERGENCE / NO_EQUIVALENT relation carries (empty for MATCHES).
        fields: The struct's ordered fields (required non-empty for a struct, empty otherwise).
        raises: Validation counterexamples, each ``(kwargs overriding params, the ValueError match regex)``.
        golden: The frozen golden master (the recommended hand-computed anchor), or ``None``.
        pins: Crafted-input cases for exact values the synthesis and the oracle cannot derive.
        recomposition: A zero-argument callable returning the ``pl.Expr`` that rebuilds this indicator from other
            public functions (an oscillator as a difference of two lines), or ``None`` when there is no such identity.
        deviant: The documented answer to the all-null regime, or ``None`` for the ordinary all-null answer.
        conditioning: An optional Hypothesis filter excluding an ill-conditioned input regime (paired with a covering
            pin), or ``None``.
        flow_deviation: A reason the interior-missing-bar flow is input-dependent, skipping the flow rungs; else empty.
        flow_horizon: The rows past a missing bar the flow must have played out in, or ``-1`` to derive it.
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
        intro_basic: The optional line opening the basic Examples block, or empty.
        intro_over: The ``.over`` panel scenario intro, or empty.
        intro_missing: The null / NaN scenario intro, or empty.

    Returns:
        The registered declaration, so ``FOO = suite_indicators(...)`` both binds and enrolls it.
    """
    declaration = Declaration(
        family="indicators",
        factory=factory,
        inputs=inputs,
        params=params,
        shape=shape,
        behavior_null=null,
        behavior_nan=nan,
        oracle=oracle,
        scaling=scaling,
        warmup=_resolve_warmup(warmup, warmup_value, params, window, factory.__name__),
        fields=fields,
        raises=raises,
        golden=golden,
        pins=pins,
        recomposition=recomposition,
        deviant=deviant,
        conditioning=conditioning,
        flow_deviation=flow_deviation,
        flow_horizon=flow_horizon,
        oracle_rel_tol=oracle_rel_tol,
        oracle_abs_tol=oracle_abs_tol,
        talib=talib,
        talib_reason=talib_reason,
        seeding=seeding,
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
        intro_over=intro_over,
        intro_missing=intro_missing,
    )
    return register(declaration)
