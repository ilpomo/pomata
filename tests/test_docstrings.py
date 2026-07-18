"""
Source-only truth-couplers for the public docstring template.

Each guard here binds one part of a public factory's docstring to a fact the declaration or the signature already
states, so the two cannot drift apart: the Args entries to the signature, the NaN vocabulary to the declared NaN
behavior, the Returns opener to the declared shape, the edge-case bullet labels to the classes the declaration
activates, each asserted degenerate outcome to a pin that witnesses it, each demanded scenario to an executed Examples
block, the warm-up sentence to the declared warm-up, the Raises section to the declared counterexamples, the TA-Lib
divergence header to the declared relation, the ``ddof`` header to the signature, and the opener variant to the
mirror-oracle disclosure. The *phrasing* of a bullet is the author's; only its truth is coupled here. None of this is
reachable by ruff's pydocstyle shell checks or the doctest gate, so it is proven from the source, the same way the
ladder proves the declarations at import.
"""

import ast
import inspect
import operator
import re
from collections.abc import Callable

import polars as pl
import pytest

import pomata.indicators
import pomata.metrics
import pomata.pnl
import tests.all_declarations as _registered
from tests.support.edge_classes import (
    EdgeClass,
    asserted_outcome_kinds,
    degenerate_witness_kinds,
    required_classes,
    scenario_witnesses,
)
from tests.support.registry import registry_all
from tests.test_oracle_docstrings import STRUCTURAL_MIRRORS, discloses_mirror

# ``all_declarations`` is imported only to run its registration side effects; nothing is referenced from it directly.
del _registered

_FAMILIES = {"indicators": pomata.indicators, "pnl": pomata.pnl, "metrics": pomata.metrics}
_DECLS = {declaration.name: declaration for declaration in registry_all()}
_NAMES = sorted(_DECLS)

# Persistence markers a latching NaN bullet may use; the bare recovery verb alone is the lie the guard rejects.
_LATCH_MARKERS = ("latch", "contaminat", "poison", "every subsequent", "every later", "never recover")
# A bullet may instead delegate to its latching components — but only without the bare recovery verb.
_DELEGATION_MARKERS = ("documented for each", "inherited from", "inherits")

# The shared parameters whose Args prose is byte-identical across the package, with the sanctioned per-role
# deviants pinned by name: a new deviant (or a silently "healed" one) is a red build.
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


def _flat(text: str) -> str:
    """Whitespace-normalized text, so a template survives any line-wrap position."""
    return " ".join(text.split())


def _module_of(name: str) -> object:
    return _FAMILIES[_DECLS[name].family]


def _doc(name: str) -> str:
    doc = inspect.getdoc(getattr(_module_of(name), name))
    assert doc, f"{name} has no docstring"
    return doc


def _note(doc: str) -> str:
    return doc.split("\nNote:\n", 1)[1].split("\nSee Also:\n", 1)[0]


def _bullets(note: str) -> list[tuple[str, str]]:
    """Every ``- **Label** — text`` bullet of the Note, in order."""
    raw = re.split(r"^ *- \*\*", note, flags=re.MULTILINE)[1:]
    return [(part.split("**", 1)[0], part.split("**", 1)[1]) for part in raw]


def _args_entries(doc: str) -> list[tuple[str, str]]:
    """The Args entries in order, each joined to one normalized line."""
    block = doc.split("\nArgs:\n", 1)[1].split("\nReturns:\n", 1)[0]
    entries: list[tuple[str, str]] = []
    for line in block.splitlines():
        match = re.match(r"    (\w+): (.*)", line)
        if match:
            entries.append((match.group(1), match.group(2)))
        elif entries and line.startswith("        "):
            entries[-1] = (entries[-1][0], entries[-1][1] + " " + line.strip())
    return entries


# The observed output shape, read straight off the declaration, mapped to the Returns opener's three template forms.
_SHAPE_FORM = {"REDUCING": "reducing", "STRUCT": "struct", "SERIES": "elementwise"}


@pytest.mark.parametrize("name", _NAMES)
def test_args_match_signature(name: str) -> None:
    """The Args entries name every parameter, in signature order, with no extras."""
    documented = [param for param, _ in _args_entries(_doc(name))]
    declared = list(inspect.signature(getattr(_module_of(name), name)).parameters)
    assert documented == declared, f"{name}: Args {documented} != signature {declared}"


@pytest.mark.parametrize("name", _NAMES)
def test_nan_vocabulary_matches_policy(name: str) -> None:
    """A latching NaN is described with a persistence marker, never with the bare recovery verb alone."""
    note = _flat(_note(_doc(name))).lower()
    if _DECLS[name].behavior_nan.name == "LATCHES":
        stated = any(marker in note for marker in _LATCH_MARKERS)
        delegated = any(marker in note for marker in _DELEGATION_MARKERS) and "propagates" not in note
        assert stated or delegated, f"{name}: LATCHES note states no persistence and no clean delegation"


@pytest.mark.parametrize("name", _NAMES)
def test_returns_opener_matches_shape(name: str) -> None:
    """The Returns opener follows the canonical template for the function's declared output shape."""
    doc = _doc(name)
    first = doc.split("\nReturns:\n", 1)[1].split("\nRaises:\n", 1)[0].strip().splitlines()[0].strip()
    shape = _SHAPE_FORM[_DECLS[name].shape.name]
    if shape == "reducing":
        assert re.match(r"A single ``Float64`` value( in ``\[.+?\]``)?: the", first), f"{name}: {first[:80]}"
        block = doc.split("\nReturns:\n", 1)[1].split("\nRaises:\n", 1)[0]
        assert "(one value in ``select``, one per group under ``.over``)" in _flat(block), name
    elif shape == "struct":
        assert re.match(r"A struct ``pl\.Expr`` with (two|three|four) ``Float64`` fields, the same length as", first), (
            f"{name}: {first[:80]}"
        )
    else:
        assert re.match(r"The .+ for each row", first), f"{name}: {first[:80]}"


def _entry_texts(param: str) -> dict[str, str]:
    return {name: text for name in _NAMES for entry, text in _args_entries(_doc(name)) if entry == param}


@pytest.mark.parametrize("param", sorted(_SHARED_PARAMS))
def test_shared_param_prose_is_pinned(param: str) -> None:
    """Every shared parameter's Args prose equals the modal text, except the pinned per-role deviants."""
    texts = _entry_texts(param)
    counts: dict[str, int] = {}
    for text in texts.values():
        counts[text] = counts.get(text, 0) + 1
    modal = max(counts, key=lambda text: counts[text])
    deviants = frozenset(name for name, text in texts.items() if text != modal)
    pinned = _SHARED_PARAMS[param]
    assert deviants == pinned, f"{param}: deviants {sorted(deviants)} != pinned {sorted(pinned)}"


# The canonical Raises clause per shared-validator parameter, pinned whitespace-flattened: every docstring whose
# Raises block names the parameter must carry one of its allowed fragments verbatim. One shared validator, one
# phrasing — the ``risk_free_rate`` tuple admits the two sanctioned role variants (the ``< -1`` bound where the rate
# is de-annualized through ``per_period_rate``; the combined finiteness clauses where it is only ``validate_finite``).
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


def _flat_raises(name: str) -> str:
    """The docstring's Raises block, whitespace-flattened, empty when the function declares none."""
    doc = _doc(name)
    if "\nRaises:\n" not in doc:
        return ""
    return _flat(doc.split("\nRaises:\n", 1)[1].split("\n\n", 1)[0])


@pytest.mark.parametrize("param", sorted(_RAISES_FRAGMENTS))
def test_shared_raises_prose_is_pinned(param: str) -> None:
    """Every Raises block naming a shared-validator parameter carries its canonical clause verbatim."""
    fragments = _RAISES_FRAGMENTS[param]
    offenders = [
        name
        for name in _NAMES
        if f"``{param}``" in _flat_raises(name) and not any(fragment in _flat_raises(name) for fragment in fragments)
    ]
    assert not offenders, f"{param}: non-canonical Raises clause in {offenders}"


# --- the Examples edge scenarios: every demanded witness demonstrated -----------------------------------------

# A demanded scenario whose executed block is byte-identical to one already shown for an earlier class of the same
# function (a single-row window-one witness is its own insufficient-sample witness): the duplicate block is not
# repeated under a second heading, so the class stays documented in the Note without its own Examples witness.
# Shrink-only: an entry whose class heading appears anyway fails the sweep loudly.
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

_SCENARIO_INTRO = re.compile(
    r"^    \*\*(Domain|Insufficient sample|Degenerate denominator|Non-finite input|window == 1)\*\* — ",
    re.MULTILINE,
)

# The token each asserted outcome kind must show in a scenario's printed output line. The ``zero`` pattern matches
# an exact ``0.0`` / ``-0.0`` lane without matching the digits of an ordinary value such as ``10.0`` or ``0.01``.
_KIND_TOKENS: dict[str, str] = {
    "null": r"None",
    "nan": r"nan",
    "inf": r"inf",
    "zero": r"(?<![\d.])-?0\.0(?![\d])",
}


def _examples_text(name: str) -> str:
    """The Examples section body, indented as ``inspect.getdoc`` yields it."""
    return _doc(name).split("\nExamples:\n", 1)[1]


def _scenario_segments(name: str) -> list[tuple[str, str]]:
    """Each bold-labeled edge scenario as ``(class label, segment text)``, in order of appearance."""
    text = _examples_text(name)
    matches = list(_SCENARIO_INTRO.finditer(text))
    segments: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segments.append((match.group(1), text[match.start() : end]))
    return segments


@pytest.mark.parametrize("name", _NAMES)
def test_examples_demonstrate_every_asserted_outcome(name: str) -> None:
    """Each edge scenario the declaration demands is shown, labeled, in taxonomy order, with its outcome printed."""
    demanded = [
        (edge_class, label, kinds)
        for edge_class, label, kinds in scenario_witnesses(_DECLS[name])
        if (name, edge_class.value, label) not in _MERGED_SCENARIOS
    ]
    merged_labels = {
        edge_value
        for spec_name, edge_value, _ in _MERGED_SCENARIOS
        if spec_name == name and not any(entry[0].value == edge_value for entry in demanded)
    }
    segments = _scenario_segments(name)
    shown = [label for label, _ in segments]
    assert shown == [edge_class.value for edge_class, _, _ in demanded], (
        f"{name}: scenario labels {shown} != demanded {[entry[0].value for entry in demanded]}"
    )
    assert not (merged_labels & set(shown)), f"{name}: a merged scenario exists anyway — shrink _MERGED_SCENARIOS"
    for (edge_class, label, kinds), (_, segment) in zip(demanded, segments, strict=True):
        output = segment.strip().splitlines()[-1]
        for kind in kinds:
            assert re.search(_KIND_TOKENS[kind], output), (
                f"{name}: the {edge_class.value} scenario ({label}) prints {output!r}, not the asserted {kind}"
            )


# --- the edge-case list: labels demanded by the declaration; asserted outcomes witnessed by pins -------------


def _edge_bullets(note: str) -> list[tuple[str, str]]:
    """Each edge-case bullet as ``(label, body)``, the body whitespace-flattened with the ``—`` separator stripped; a
    bullet's content ends at the first blank line, so a paragraph following the list is not folded into the last bullet.
    """
    return [(label, _flat(rest.split("\n\n", 1)[0]).removeprefix("— ")) for label, rest in _bullets(note)]


@pytest.mark.parametrize("name", _NAMES)
def test_edge_bullets_match_the_required_classes(name: str) -> None:
    """The edge-case bullet labels are exactly the classes the declaration demands, in canonical order."""
    labels = [label for label, _ in _edge_bullets(_note(_doc(name)))]
    expected = [edge_class.value for edge_class in required_classes(_DECLS[name])]
    assert labels == expected, f"{name}: bullet labels {labels} != required {expected}"


@pytest.mark.parametrize("name", _NAMES)
def test_structural_secondary_outcomes_are_pinned(name: str) -> None:
    """Every degenerate outcome a Degenerate-denominator bullet asserts (a ``NaN`` or an infinity) is witnessed by a
    Degenerate-denominator pin of that kind — the claim⇔pin link, so a narrated regime cannot go without a fixed case.
    """
    declaration = _DECLS[name]
    witnessed = degenerate_witness_kinds(declaration)
    for label, text in _edge_bullets(_note(_doc(name))):
        if label != EdgeClass.DEGENERATE_DENOMINATOR.value:
            continue
        missing = asserted_outcome_kinds(text) - witnessed
        assert not missing, (
            f"{name}: the Degenerate-denominator bullet asserts {sorted(missing)} with no pin witnessing it — "
            f"add a Degenerate-denominator pin whose expected lanes carry that outcome"
        )


# --- the pre-list explanatory sub-headers: TA-Lib divergence and ddof, coupled to the declaration / signature ---


def _prelist_sections(name: str) -> list[tuple[str, str]]:
    """Each explanatory sub-header of the Note with its first following paragraph, whitespace-flattened."""
    lines = _note(_doc(name)).splitlines()
    sections: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        match = re.fullmatch(r"\*\*([^*\n]+?)\*\*", line.strip())
        if match is None:
            continue
        start = index + 1
        while start < len(lines) and not lines[start].strip():
            start += 1
        stop = start
        while stop < len(lines) and lines[stop].strip():
            stop += 1
        sections.append((match.group(1), _flat(" ".join(lines[start:stop]))))
    return sections


def _prelist_headers(name: str) -> set[str]:
    return {header for header, _ in _prelist_sections(name)}


@pytest.mark.parametrize("name", _NAMES)
def test_talib_divergence_header_matches_the_registry(name: str) -> None:
    """The ``Documented TA-Lib divergence`` sub-header appears exactly on the functions the declaration marks as a
    deliberate TA-Lib divergence — the header cannot go stale on either side.
    """
    carries = "Documented TA-Lib divergence" in _prelist_headers(name)
    talib = _DECLS[name].talib
    registered = talib is not None and talib.name == "DOCUMENTED_DIVERGENCE"
    assert carries == registered, (
        f"{name}: TA-Lib divergence header present={carries} but declared DOCUMENTED_DIVERGENCE={registered}"
    )


@pytest.mark.parametrize("name", _NAMES)
def test_ddof_header_matches_the_signature(name: str) -> None:
    """The ``Degrees of freedom`` sub-header appears exactly on the factories that take a ``ddof`` parameter."""
    carries = "Degrees of freedom" in _prelist_headers(name)
    declared = "ddof" in inspect.signature(getattr(_module_of(name), name)).parameters
    assert carries == declared, f"{name}: Degrees-of-freedom header present={carries} but ddof parameter={declared}"


def test_opener_variants_are_the_path_dependent_set() -> None:
    """The references that disclose a structural-mirror nature are exactly the path-dependent set: only a function
    genuinely without an independent oracle admits it, and every such function does.
    """
    disclosing = {name for name, declaration in _DECLS.items() if discloses_mirror(declaration)}
    assert disclosing == set(STRUCTURAL_MIRRORS), (
        f"disclosing {sorted(disclosing)} != path-dependent set {sorted(STRUCTURAL_MIRRORS)}"
    )


@pytest.mark.parametrize("name", _NAMES)
def test_note_headers_are_their_own_paragraph(name: str) -> None:
    """
    Every non-bullet bold label in the Note is a header alone on its line, separated from its content by a blank line,
    so Sphinx renders it as its own paragraph instead of joining it to the sentence below. A ``**Label:**`` colon
    header, a ``**Label** --`` / ``**Label** —`` dash connective, or a header whose content abuts it with no blank
    line is rejected. A list item keeps its ``- **Label** — text`` form and is exempt.
    """
    lines = _note(_doc(name)).splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("- "):
            continue
        match = re.match(r"\*\*([^*]+?)\*\*(.*)", stripped)
        if match is None:
            continue
        label, rest = match.group(1), match.group(2)
        assert not label.endswith(":"), f"{name}: header **{label}** still carries a trailing colon"
        assert not rest.strip(), f"{name}: header **{label}** carries inline text on its own line"
        following = lines[index + 1] if index + 1 < len(lines) else ""
        assert not following.strip(), f"{name}: header **{label}** is joined to its content — no blank line follows"


@pytest.mark.parametrize("name", _NAMES)
def test_raises_section_names_every_counterexample(name: str) -> None:
    """Every validation counterexample's parameter is named in the Raises section, so no guard goes undocumented."""
    declaration = _DECLS[name]
    if not declaration.raises:
        return
    raises = _flat_raises(name)
    offenders = sorted({param for override, _ in declaration.raises for param in override if param not in raises})
    assert not offenders, f"{name}: Raises does not name {offenders}"


@pytest.mark.parametrize("name", _NAMES)
def test_scalar_knobs_are_keyword_only(name: str) -> None:
    """
    Verifies the positional surface is only the ``pl.Expr`` inputs and the ``window*`` lookbacks — every other
    scalar knob is keyword-only, so a call site always names it.
    """
    factory = getattr(_module_of(name), name)
    for parameter in inspect.signature(factory).parameters.values():
        if parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            continue
        positional_kind = parameter.annotation is pl.Expr or (
            parameter.annotation is int and parameter.name.startswith("window")
        )
        assert positional_kind, f"{parameter.name}: positional but neither a pl.Expr input nor a window lookback"


# The sentence forms a Returns block states its warm-up in; the captured group is the backticked formula whose
# value, at the canonical parameters, must equal the declaration's warm-up.
_WARMUP_PATTERNS: tuple[str, ...] = (
    r"first ``([^`]+)`` (?:rows|values) are ``null``",
    r"(?:Values|values) are ``null`` until [^(]*\(the first ``([^`]+)`` rows\)",
    r"``([^`]+)`` rows of ``null``",
    r"[Tt]he first (?:value|row) is ``null``()",
    r"Row ``0`` is ``null``()",
)

# Warm-ups substitution cannot check: adxr composes its length from adx's in words, and hma's formula reads a
# prose-defined intermediate. Shrink this set by rephrasing to a formula in the canonical parameters; an entry
# whose Returns becomes checkable fails as stale.
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


def _returns_block(name: str) -> str:
    """The full Returns section, flattened to one line."""
    tail = _doc(name).split("\nReturns:\n", 1)[1]
    cuts = [tail.find("\n" + section) for section in ("Raises:", "Note:") if "\n" + section in tail]
    return " ".join(tail[: min(cuts)].split()) if cuts else " ".join(tail.split())


@pytest.mark.parametrize("name", _NAMES)
def test_returns_warmup_matches_the_spec(name: str) -> None:
    """The Returns warm-up formula reproduces the declaration's warm-up under the canonical parameters."""
    declaration = _DECLS[name]
    if declaration.warmup is None or not isinstance(declaration.warmup, int):
        return
    text = _returns_block(name)
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
