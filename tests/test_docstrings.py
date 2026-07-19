"""
Declaration-level truth-couplers for the generated docstring.

Since the round-trip guard (:mod:`tests.test_docstring_roundtrip`) locks every source docstring tail to
``tail_for(declaration)`` byte-for-byte, a docstring can no longer diverge from its declaration by hand — so these
couplers no longer parse the assembled docstring. They read the declaration's own data and hold it to the facts the
signature, the behavior axes, and the pins already state: every parameter is described, a latching ``NaN`` bullet states
its persistence, the Returns body and its warm-up formula match the shape and the leading-null count, the shared Args
and Raises prose stay uniform across the package, the edge-case bullets and the Examples scenarios are exactly the
classes the pins activate, and each asserted degenerate outcome is witnessed by a pin. The couplings conditional on a
single declaration's data (the two conditional Note headers, the Raises naming) sweep the registry too — they bind only
the registered declarations that generate a docstring, not the bare test doubles the rung tests build; the
mirror-oracle disclosure lives beside the oracle-docstring guards. None of this is reachable by ruff's pydocstyle shell
checks or the doctest gate, so it is proven from the declaration.
"""

import ast
import inspect
import operator
import re
from collections.abc import Callable

import polars as pl
import pytest

import tests.all_declarations as _registered
from tests.support.declaration import Declaration, Example
from tests.support.docstring import ARG_DESCRIPTIONS, execute_scenario
from tests.support.edge_classes import (
    EdgeClass,
    asserted_outcome_kinds,
    degenerate_witness_kinds,
    required_classes,
    scenario_witnesses,
)
from tests.support.registry import registry_all

# ``all_declarations`` is imported only to run its registration side effects; nothing is referenced from it directly.
del _registered

_DECLS = {declaration.name: declaration for declaration in registry_all()}
_NAMES = sorted(_DECLS)


def _flat(text: str) -> str:
    """Whitespace-normalized text, so a comparison survives any wrapping the prose was authored under."""
    return " ".join(text.split())


# --- the Args prose: every parameter described; the shared parameters uniform ---------------------------------


def _effective_arg(declaration: Declaration, param: str) -> str:
    """The Args description a parameter resolves to in the generator: the per-function ``args_prose`` override where
    declared, else the mined shared table.
    """
    return _flat(declaration.args_prose.get(param) or ARG_DESCRIPTIONS.get(param, ""))


@pytest.mark.parametrize("name", _NAMES)
def test_every_parameter_is_described(name: str) -> None:
    """Every factory parameter resolves to an Args description — an ``args_prose`` entry or the mined shared table — so
    the generated Args can never silently drop a parameter.
    """
    declaration = _DECLS[name]
    undescribed = [
        param
        for param in inspect.signature(declaration.factory).parameters
        if param not in declaration.args_prose and param not in ARG_DESCRIPTIONS
    ]
    assert not undescribed, f"{name}: parameters {undescribed} carry no Args description (declare args_prose)"


# The shared parameters whose Args prose is uniform across the package, with the sanctioned per-role deviants pinned by
# name: a new deviant (or a silently "healed" one) is a red build. A parameter absent from the mined shared table falls
# to whichever wording the majority of its sites declare.
_SHARED_PARAMS: dict[str, frozenset[str]] = {
    "expr": frozenset({"obv"}),
    "high": frozenset(),
    "low": frozenset(),
    "close": frozenset({"atr", "price_weighted_close", "true_range"}),
    "volume": frozenset(),
    "open": frozenset(),
    "benchmark": frozenset(),
    "threshold": frozenset(),
    "confidence": frozenset(
        {
            "conditional_drawdown_at_risk",
            "conditional_value_at_risk",
        }
    ),
    "periods_per_year": frozenset({"probabilistic_sharpe_ratio"}),
    "weight": frozenset(),
    "quantity": frozenset({"cost_borrow", "dividend"}),
    "price": frozenset({"pnl_gross_inverse"}),
    "returns": frozenset({"cumulative_pnl", "equity_curve", "stability"}),
    "equity_curve": frozenset({"cagr", "total_return"}),
    "risk_free_rate": frozenset(
        {
            "burke_ratio",
            "modigliani_risk_adjusted_performance",
            "pain_ratio",
            "sterling_ratio",
            "ulcer_performance_ratio",
        }
    ),
    "ddof": frozenset({"variance_rolling"}),
    "adjust": frozenset({"dema", "ema", "t3", "tema"}),
    "multiplier": frozenset({"keltner_channels", "pnl_gross", "pnl_gross_inverse", "supertrend"}),
}


@pytest.mark.parametrize("param", sorted(_SHARED_PARAMS))
def test_shared_param_prose_is_pinned(param: str) -> None:
    """Every shared parameter's Args description equals the modal wording, except the pinned per-role deviants."""
    texts = {
        name: _effective_arg(_DECLS[name], param)
        for name in _NAMES
        if param in inspect.signature(_DECLS[name].factory).parameters
    }
    counts: dict[str, int] = {}
    for text in texts.values():
        counts[text] = counts.get(text, 0) + 1
    modal = max(counts, key=lambda text: counts[text])
    deviants = frozenset(name for name, text in texts.items() if text != modal)
    pinned = _SHARED_PARAMS[param]
    assert deviants == pinned, f"{param}: deviants {sorted(deviants)} != pinned {sorted(pinned)}"


# --- the Note NaN vocabulary: a latching NaN states its persistence -------------------------------------------

# Persistence markers a latching NaN bullet may use; the bare recovery verb alone is the lie the coupler rejects.
_LATCH_MARKERS = ("latch", "contaminat", "poison", "every subsequent", "every later", "never recover")
# A bullet may instead delegate to its latching components — but only without the bare recovery verb.
_DELEGATION_MARKERS = ("documented for each", "inherited from", "inherits")


def _note_prose(declaration: Declaration) -> str:
    """The whitespace-flattened per-function Note prose — the opener override / extension, the pre-list note bodies, and
    every edge-case bullet body — i.e. everywhere a declaration states a behavior in words (the family opener template
    carries no persistence vocabulary, so omitting it changes nothing).
    """
    parts = [declaration.opener_override, declaration.note_extension]
    parts += [body for _, body in declaration.notes]
    parts += [body for _, body in declaration.bullets]
    return _flat(" ".join(parts))


@pytest.mark.parametrize("name", _NAMES)
def test_nan_vocabulary_matches_policy(name: str) -> None:
    """A latching ``NaN`` is described with a persistence marker in the Note prose, never the bare recovery verb."""
    declaration = _DECLS[name]
    if declaration.behavior_nan.name != "LATCHES":
        return
    note = _note_prose(declaration).lower()
    stated = any(marker in note for marker in _LATCH_MARKERS)
    delegated = any(marker in note for marker in _DELEGATION_MARKERS) and "propagates" not in note
    assert stated or delegated, f"{name}: LATCHES note states no persistence and no clean delegation"


# --- the Raises prose: shared validators phrased canonically --------------------------------------------------

# The canonical Raises clause per shared-validator parameter, pinned whitespace-flattened: every ``raises_prose`` that
# names the parameter must carry one of its allowed fragments verbatim. One shared validator, one phrasing — the
# ``risk_free_rate`` tuple admits the two sanctioned role variants (the ``< -1`` bound where the rate is de-annualized
# through ``per_period_rate``; the combined finiteness clauses where it is only ``validate_finite``).
_RAISES_FRAGMENTS: dict[str, tuple[str, ...]] = {
    "confidence": ("``confidence`` is not in the open interval ``(0, 1)``",),
    "fee": ("``fee`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``)",),
    "multiplier": ("``multiplier`` is not a finite number ``> 0`` (i.e. ``<= 0``, ``NaN``, or ``±inf``)",),
    "rate": ("``rate`` is not a finite number ``>= 0`` (i.e. ``< 0``, ``NaN``, or ``±inf``)",),
    "risk_free_rate": (
        "``risk_free_rate`` is not finite or is ``< -1``",
        "``risk_free_rate`` is not finite",
        "``risk_free_rate`` or ``excess`` is not finite",
    ),
    "threshold": ("``threshold`` is not finite",),
}


@pytest.mark.parametrize("param", sorted(_RAISES_FRAGMENTS))
def test_shared_raises_prose_is_pinned(param: str) -> None:
    """Every declared Raises clause naming a shared-validator parameter carries its canonical fragment verbatim."""
    fragments = _RAISES_FRAGMENTS[param]
    offenders = [
        declaration.name
        for declaration in _DECLS.values()
        if f"``{param}``" in _flat(declaration.raises_prose)
        and not any(fragment in _flat(declaration.raises_prose) for fragment in fragments)
    ]
    assert not offenders, f"{param}: non-canonical Raises clause in {offenders}"


# --- the Examples edge scenarios: every demanded witness demonstrated -----------------------------------------

# A demanded scenario whose executed block would duplicate one already shown for an earlier class of the same function
# (a single-row window-one witness is its own insufficient-sample witness): the block is not repeated under a second
# heading, so the class stays documented in the Note without its own Examples witness. Shrink-only: an entry whose class
# heading appears anyway fails the coupler loudly.
_MERGED_SCENARIOS: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("common_sense_ratio", "Degenerate denominator", "single_row_all_loss"),
        ("dema", "window == 1", "single_row_window_one"),
        ("fisher_transform", "window == 1", "window_one_single_row_is_flat_nan"),
        ("roc", "window == 1", "single_row_window_one"),
        ("sma", "window == 1", "single_row_window_one_identity"),
        ("trima", "window == 1", "single_row_window_one_identity"),
        ("vwma", "window == 1", "single_row_window_one_identity"),
        ("williams_r", "window == 1", "single_row_window_one"),
    }
)

# The bold edge-class label opening an edge scenario's declared intro (``**Class** — ...``).
_EDGE_INTRO = re.compile(
    r"^\*\*(Domain|Insufficient sample|Degenerate denominator|Non-finite input|window == 1)\*\* — "
)

# The token each asserted outcome kind must show in a scenario's executed output line. The ``zero`` pattern matches an
# exact ``0.0`` / ``-0.0`` lane without matching the digits of an ordinary value such as ``10.0`` or ``0.01``.
_KIND_TOKENS: dict[str, str] = {
    "null": r"None",
    "nan": r"nan",
    "inf": r"inf",
    "zero": r"(?<![\d.])-?0\.0(?![\d])",
}


def _edge_examples(declaration: Declaration) -> list[tuple[str, Example]]:
    """Each Examples scenario whose intro opens with a bold edge-class label, as ``(label, example)`` in order."""
    labeled: list[tuple[str, Example]] = []
    for example in declaration.examples:
        match = _EDGE_INTRO.match(example.intro)
        if match is not None:
            labeled.append((match.group(1), example))
    return labeled


@pytest.mark.parametrize("name", _NAMES)
def test_examples_demonstrate_every_asserted_outcome(name: str) -> None:
    """Each edge scenario the declaration's pins demand is an Examples record, labeled, in taxonomy order, and its
    executed output prints the asserted degenerate outcome.
    """
    declaration = _DECLS[name]
    demanded = [
        (edge_class, label, kinds)
        for edge_class, label, kinds in scenario_witnesses(declaration)
        if (name, edge_class.value, label) not in _MERGED_SCENARIOS
    ]
    merged_labels = {
        edge_value
        for spec_name, edge_value, _ in _MERGED_SCENARIOS
        if spec_name == name and not any(entry[0].value == edge_value for entry in demanded)
    }
    shown = _edge_examples(declaration)
    labels = [label for label, _ in shown]
    assert labels == [edge_class.value for edge_class, _, _ in demanded], (
        f"{name}: Examples edge scenarios {labels} != demanded {[entry[0].value for entry in demanded]}"
    )
    assert not (merged_labels & set(labels)), f"{name}: a merged scenario is shown anyway — shrink _MERGED_SCENARIOS"
    for (edge_class, label, kinds), (_, example) in zip(demanded, shown, strict=True):
        if not kinds:
            continue
        outputs = execute_scenario(declaration, example)
        for kind in kinds:
            assert any(re.search(_KIND_TOKENS[kind], output) for output in outputs), (
                f"{name}: the {edge_class.value} scenario ({label}) prints {outputs!r}, not the asserted {kind}"
            )


# --- the edge-case list: labels demanded by the declaration; asserted outcomes witnessed by pins --------------


@pytest.mark.parametrize("name", _NAMES)
def test_edge_bullets_match_the_required_classes(name: str) -> None:
    """The declared edge-case bullet labels are exactly the classes the declaration demands, in canonical order."""
    declaration = _DECLS[name]
    labels = [label for label, _ in declaration.bullets]
    expected = [edge_class.value for edge_class in required_classes(declaration)]
    assert labels == expected, f"{name}: bullet labels {labels} != required {expected}"


@pytest.mark.parametrize("name", _NAMES)
def test_structural_secondary_outcomes_are_pinned(name: str) -> None:
    """Every degenerate outcome a Degenerate-denominator bullet asserts (a ``NaN`` or an infinity) is witnessed by a
    Degenerate-denominator pin of that kind — the claim⇔pin link, so a narrated regime cannot go without a fixed case.
    """
    declaration = _DECLS[name]
    witnessed = degenerate_witness_kinds(declaration)
    for label, body in declaration.bullets:
        if label != EdgeClass.DEGENERATE_DENOMINATOR.value:
            continue
        missing = asserted_outcome_kinds(body) - witnessed
        assert not missing, (
            f"{name}: the Degenerate-denominator bullet asserts {sorted(missing)} with no pin witnessing it — "
            f"add a Degenerate-denominator pin whose expected lanes carry that outcome"
        )


# --- the conditional Note headers and the Raises naming (per-declaration couplings over the registry) ---------


def _note_headers(declaration: Declaration) -> set[str]:
    """The declared pre-list Note subheader labels."""
    return {label for label, _ in declaration.notes}


@pytest.mark.parametrize("name", _NAMES)
def test_talib_divergence_header_matches_the_registry(name: str) -> None:
    """The ``Documented TA-Lib divergence`` Note header appears exactly on the functions the declaration marks as a
    deliberate TA-Lib divergence — the header cannot go stale on either side.
    """
    declaration = _DECLS[name]
    carries = "Documented TA-Lib divergence" in _note_headers(declaration)
    talib = declaration.talib
    registered = talib is not None and talib.name == "DOCUMENTED_DIVERGENCE"
    assert carries == registered, (
        f"{name}: TA-Lib divergence header present={carries} but declared DOCUMENTED_DIVERGENCE={registered}"
    )


@pytest.mark.parametrize("name", _NAMES)
def test_ddof_header_matches_the_signature(name: str) -> None:
    """The ``Degrees of freedom`` Note header appears exactly on the factories that take a ``ddof`` parameter."""
    declaration = _DECLS[name]
    carries = "Degrees of freedom" in _note_headers(declaration)
    declared = "ddof" in inspect.signature(declaration.factory).parameters
    assert carries == declared, f"{name}: Degrees-of-freedom header present={carries} but ddof parameter={declared}"


@pytest.mark.parametrize("name", _NAMES)
def test_raises_section_names_every_counterexample(name: str) -> None:
    """Every validation counterexample's parameter is named in the declared Raises prose, so no guard goes undocumented.

    The shared ``TypeError`` line carries no counterexample parameter, so the per-function ``raises_prose`` is the whole
    Raises text a counterexample parameter can appear in.
    """
    declaration = _DECLS[name]
    if not declaration.raises:
        return
    prose = _flat(declaration.raises_prose)
    offenders = sorted({param for override, _ in declaration.raises for param in override if param not in prose})
    assert not offenders, f"{name}: the Raises prose does not name {offenders}"


# --- the signature surface: scalar knobs keyword-only ---------------------------------------------------------


@pytest.mark.parametrize("name", _NAMES)
def test_scalar_knobs_are_keyword_only(name: str) -> None:
    """
    Verifies the positional surface is only the ``pl.Expr`` inputs and the ``window*`` lookbacks — every other scalar
    knob is keyword-only, so a call site always names it.
    """
    factory = _DECLS[name].factory
    for parameter in inspect.signature(factory).parameters.values():
        if parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            continue
        positional_kind = parameter.annotation is pl.Expr or (
            parameter.annotation is int and parameter.name.startswith("window")
        )
        assert positional_kind, f"{parameter.name}: positional but neither a pl.Expr input nor a window lookback"


# --- the Returns body: opener template and warm-up formula ----------------------------------------------------

# The observed output shape, read straight off the declaration, mapped to the Returns opener's three template forms.
_SHAPE_FORM = {"REDUCING": "reducing", "STRUCT": "struct", "SERIES": "elementwise"}


@pytest.mark.parametrize("name", _NAMES)
def test_returns_opener_matches_shape(name: str) -> None:
    """The declared Returns body opens with the canonical template for the function's declared output shape."""
    declaration = _DECLS[name]
    first = _flat(declaration.returns_body)
    shape = _SHAPE_FORM[declaration.shape.name]
    if shape == "reducing":
        assert re.match(r"A single ``Float64`` value( in ``\[.+?\]``)?: the", first), (
            f"{name}: Returns opener {first[:80]!r} does not open with the reducing template"
        )
        assert "(one value in ``select``, one per group under ``.over``)" in first, (
            f"{name}: the reducing Returns body lacks the ``select``/``.over`` clause"
        )
    elif shape == "struct":
        assert re.match(r"A struct ``pl\.Expr`` with (two|three|four) ``Float64`` fields, the same length as", first), (
            f"{name}: Returns opener {first[:80]!r} does not open with the struct template"
        )
    else:
        assert re.match(r"The .+ for each row", first), (
            f"{name}: Returns opener {first[:80]!r} does not open with the elementwise template"
        )


# The sentence forms a Returns body states its warm-up in; the captured group is the backticked formula whose value, at
# the canonical parameters, must equal the declaration's warm-up.
_WARMUP_PATTERNS: tuple[str, ...] = (
    r"first ``([^`]+)`` (?:rows|values) are ``null``",
    r"(?:Values|values) are ``null`` until [^(]*\(the first ``([^`]+)`` rows\)",
    r"``([^`]+)`` rows of ``null``",
    r"[Tt]he first (?:value|row) is ``null``()",
    r"Row ``0`` is ``null``()",
)

# Warm-ups substitution cannot check: adxr composes its length from adx's in words, and hma's formula reads a
# prose-defined intermediate. Shrink this set by rephrasing to a formula in the canonical parameters; an entry whose
# Returns becomes checkable fails as stale.
_WARMUP_PROSE: frozenset[str] = frozenset({"adxr", "hma"})


def _eval_warmup(formula: str, params: dict[str, int]) -> int:
    """The integer value of a backticked warm-up formula under the canonical parameters (an empty capture is the
    one-row shift, i.e. ``1``).
    """
    if not formula:
        return 1
    node = ast.parse(formula, mode="eval").body
    operators: dict[type[ast.operator], Callable[[int, int], int]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
    }

    def evaluate(expr: ast.expr) -> int:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, int):
            return expr.value
        if isinstance(expr, ast.Name) and expr.id in params:
            return params[expr.id]
        if isinstance(expr, ast.BinOp) and type(expr.op) in operators:
            return operators[type(expr.op)](evaluate(expr.left), evaluate(expr.right))
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
            return -evaluate(expr.operand)
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "max":
            return max(evaluate(argument) for argument in expr.args)
        message = f"unsupported warm-up formula node: {ast.dump(expr)}"
        raise ValueError(message)

    return evaluate(node)


@pytest.mark.parametrize("name", _NAMES)
def test_returns_warmup_matches_the_spec(name: str) -> None:
    """The declared Returns warm-up formula reproduces the declaration's warm-up under the canonical parameters."""
    declaration = _DECLS[name]
    if declaration.warmup is None or not isinstance(declaration.warmup, int):
        return
    text = _flat(declaration.returns_body)
    formula = next((m.group(1) for p in _WARMUP_PATTERNS if (m := re.search(p, text)) is not None), None)
    params = {key: value for key, value in declaration.params.items() if isinstance(value, int)}
    if name in _WARMUP_PROSE:
        if formula is None:
            return
        try:
            _eval_warmup(formula, params)
        except ValueError:
            return
        pytest.fail(f"{name}: now checkable by substitution — remove its stale _WARMUP_PROSE entry")
    assert formula is not None, f"{name}: no checkable warm-up sentence found in Returns"
    declared = _eval_warmup(formula, params)
    assert declared == declaration.warmup, (
        f"{name}: the Returns formula ``{formula}`` gives {declared} at the canonical parameters, but the "
        f"declaration states warmup={declaration.warmup}"
    )
