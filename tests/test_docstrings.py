"""
Source-only conformance guards for the public docstring template.

Every public factory's docstring follows one template: the seven Google sections in one order, the Args entries in
signature order, the byte-identical ``TypeError`` line, a sanctioned Note opener (a family-level **Precision** /
**Correctness** form or the function's pinned variant), the edge-case bullet labels its declarations demand in
canonical order, each bullet worded to its class canon (or pinned as a deviant), the latch marker every
``NanPolicy.LATCHES`` docstring must carry, a whitelisted set of explanatory sub-headers above the edge-case list,
the canonical Returns opener per output shape, byte-identical Args prose for every shared parameter (with the
sanctioned per-role deviants pinned by name), a Raises section that names every validation counterexample's
parameter, and a Returns warm-up formula that reproduces the spec's declared warm-up when the canonical parameters
are substituted into it. None of this is reachable by ruff's pydocstyle shell checks or the doctest gate, so it is
proven here, from the source, the same way the spec ladder proves the suite's own declarations at import.
"""

import ast
import inspect
import operator
import re
from collections.abc import Callable
from types import ModuleType

import polars as pl
import pytest
from tests.all_specs import ALL_SPECS
from tests.support import COLUMN_X, synthesize_call
from tests.support.edge_classes import DEVIANT_BULLETS, bullet_matches, expected_bullet, required_classes

import pomata.indicators
import pomata.metrics
import pomata.pnl
from pomata._policy import POLICIES, NanPolicy

_FAMILIES = {"indicators": pomata.indicators, "pnl": pomata.pnl, "metrics": pomata.metrics}
_SECTIONS = ("Args", "Returns", "Raises", "Note", "See Also", "References", "Examples")
_TYPE_ERROR_LINE = "TypeError: If any input is not a ``pl.Expr``."
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


def _module_of(name: str) -> ModuleType:
    for module in _FAMILIES.values():
        if name in module.__all__:
            return module
    raise AssertionError(f"{name} is in no family __all__")


def _family_of(name: str) -> str:
    for family, module in _FAMILIES.items():
        if name in module.__all__:
            return family
    raise AssertionError(name)


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


def _shape(name: str) -> str:
    """The output shape observed from a probe, exactly as the spec engine observes it."""
    factory = getattr(_module_of(name), name)
    positional, keywords = synthesize_call(factory)
    frame = pl.DataFrame({COLUMN_X: pl.Series([float(value) for value in range(1, 9)], dtype=pl.Float64)})
    out = frame.select(factory(*positional, **keywords).alias("o"))
    if out.height == 1:
        return "reducing"
    return "struct" if isinstance(out.schema["o"], pl.Struct) else "elementwise"


_NAMES = sorted(POLICIES)
_SPECS = {spec.name: spec for spec in ALL_SPECS}


@pytest.mark.parametrize("name", _NAMES)
def test_sections_present_and_ordered(name: str) -> None:
    """Every docstring carries exactly the seven template sections, in the one canonical order."""
    doc = _doc(name)
    headers = [line.strip() for line in doc.splitlines() if line.strip().endswith(":")]
    found = tuple(header.rstrip(":") for header in headers if header.rstrip(":") in _SECTIONS)
    assert found == _SECTIONS, f"{name}: sections {found}"


@pytest.mark.parametrize("name", _NAMES)
def test_args_match_signature(name: str) -> None:
    """The Args entries name every parameter, in signature order, with no extras."""
    documented = [param for param, _ in _args_entries(_doc(name))]
    declared = list(inspect.signature(getattr(_module_of(name), name)).parameters)
    assert documented == declared, f"{name}: Args {documented} != signature {declared}"


@pytest.mark.parametrize("name", _NAMES)
def test_type_error_line_is_canonical(name: str) -> None:
    """The Raises section carries the byte-identical TypeError line."""
    assert _TYPE_ERROR_LINE in _flat(_doc(name)), name


@pytest.mark.parametrize("name", _NAMES)
def test_note_opener_matches_family(name: str) -> None:
    """The Note opens with **Precision** for indicators and **Correctness** for metrics and pnl."""
    opener = re.search(r"\*\*(.+?)\*\*", _note(_doc(name)))
    expected = "Precision" if _family_of(name) == "indicators" else "Correctness"
    assert opener is not None, f"{name}: the Note opens with no bold marker"
    assert opener.group(1) == expected, f"{name}: opener {opener.group(1)}"


@pytest.mark.parametrize("name", _NAMES)
def test_nan_vocabulary_matches_policy(name: str) -> None:
    """A latching NaN is described with a persistence marker, never with the bare recovery verb alone."""
    _, nan_policy = POLICIES[name]
    note = _flat(_note(_doc(name))).lower()
    if nan_policy is NanPolicy.LATCHES:
        stated = any(marker in note for marker in _LATCH_MARKERS)
        delegated = any(marker in note for marker in _DELEGATION_MARKERS) and "propagates" not in note
        assert stated or delegated, f"{name}: LATCHES note states no persistence and no clean delegation"


@pytest.mark.parametrize("name", _NAMES)
def test_returns_opener_matches_shape(name: str) -> None:
    """The Returns opener follows the canonical template for the function's observed output shape."""
    doc = _doc(name)
    first = doc.split("\nReturns:\n", 1)[1].split("\nRaises:\n", 1)[0].strip().splitlines()[0].strip()
    shape = _shape(name)
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


# The canonical Examples opening: ``>>> import polars as pl`` then the function's own family import, optionally
# preceded by the stdlib import an example genuinely needs (the seven Hilbert-cycle functions import ``math``).
# The one sanctioned alias is pinned: a healed (or new) alias is a red build.
_EXAMPLES_ALIASED: frozenset[str] = frozenset({"modigliani_risk_adjusted_performance"})


def _examples_import_header(name: str) -> list[str]:
    """The leading ``>>>`` import statements of the Examples block, in order."""
    block = _doc(name).split("\nExamples:\n", 1)[1]
    header: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith(">>> "):
            if header:
                break
            continue
        statement = stripped[4:]
        if statement.startswith(("import ", "from ")):
            header.append(statement)
        else:
            break
    return header


@pytest.mark.parametrize("name", _NAMES)
def test_examples_open_with_the_canonical_imports(name: str) -> None:
    """The Examples block opens with the polars import then the function's own bare family import."""
    header = _examples_import_header(name)
    if header and re.fullmatch(r"import \w+", header[0]) and header[0] != "import polars as pl":
        header = header[1:]  # a stdlib import the example needs may lead
    family = _family_of(name)
    expected_self = f"from pomata.{family} import {name}"
    if name in _EXAMPLES_ALIASED:
        assert len(header) == 2, f"{name}: header {header}"
        assert header[0] == "import polars as pl", f"{name}: header {header}"
        assert re.fullmatch(rf"{re.escape(expected_self)} as \w+", header[1]), (
            f"{name}: pinned as aliased, got {header[1]!r}"
        )
    else:
        assert header == ["import polars as pl", expected_self], f"{name}: header {header}"


# --- the source-form conventions: the factory body and its docstring literal, swept from the same registry ---


def _function_source(name: str) -> str:
    """The factory's own source text, for the literal-level facts the parsed docstring cannot carry."""
    return inspect.getsource(getattr(_module_of(name), name))


def _function_def(name: str) -> ast.FunctionDef:
    """The factory's parsed definition, for the body-shape sweeps."""
    node = ast.parse(_function_source(name)).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


@pytest.mark.parametrize("name", _NAMES)
def test_edge_case_block_is_the_terminal_subheader(name: str) -> None:
    """Verifies ``**Edge-case behavior:**`` is the LAST bold sub-header of the Note — explanations precede it."""
    headers = re.findall(r"\*\*([^*\n]+?):\*\*", _note(_doc(name)))
    assert headers, "the Note carries no bold sub-header"
    assert headers[-1] == "Edge-case behavior", f"the Note's sub-headers end with {headers[-1]!r}"


@pytest.mark.parametrize("name", _NAMES)
def test_docstring_literal_is_raw(name: str) -> None:
    """Verifies the public docstring is a raw string, so the math blocks' backslashes never need doubling."""
    source = _function_source(name)
    index = source.index('"""')
    assert source[index - 1] == "r", "the public docstring literal is not raw"


@pytest.mark.parametrize("name", _NAMES)
def test_body_returns_name_keep(name: str) -> None:
    """Verifies every top-level return keeps the landing column: the expression chain ends in ``.name.keep()``."""
    returns = [node for node in _function_def(name).body if isinstance(node, ast.Return)]
    assert returns, "no top-level return"
    for node in returns:
        call = node.value
        assert isinstance(call, ast.Call), "the return is not a call chain"
        assert isinstance(call.func, ast.Attribute), "the return does not end in .name.keep()"
        assert call.func.attr == "keep", "the return does not end in .name.keep()"
        assert isinstance(call.func.value, ast.Attribute), "the return does not end in .name.keep()"
        assert call.func.value.attr == "name", "the return does not end in .name.keep()"


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


def _phase_of(statement: ast.stmt) -> str:
    """The body phase one statement belongs to: docstring, normalization, validation, or computation."""
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
        return "docstring"
    if (
        isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "float64_expr"
    ):
        return "normalize"
    if (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id.startswith("validate_")
    ):
        return "validate"
    return "compute"


@pytest.mark.parametrize("name", _NAMES)
def test_body_keeps_the_three_phases(name: str) -> None:
    """
    Verifies the body never interleaves its phases: every ``float64_expr`` normalization precedes every
    ``validate_*`` guard, and every guard precedes the first computation statement.
    """
    rank = {"normalize": 0, "validate": 1, "compute": 2}
    phases = [_phase_of(statement) for statement in _function_def(name).body]
    indices = [rank[phase] for phase in phases if phase != "docstring"]
    assert indices == sorted(indices), f"the body phases interleave: {[p for p in phases if p != 'docstring']}"


# --- the edge-case list: labels demanded by the declarations, wording held to the class canon ---


def _edge_bullets(note: str) -> list[tuple[str, str]]:
    """Each edge-case bullet as ``(label, body)``, the body whitespace-flattened with the ``—`` separator stripped."""
    return [(label, _flat(rest).removeprefix("— ")) for label, rest in _bullets(note)]


@pytest.mark.parametrize("name", _NAMES)
def test_edge_bullets_match_the_required_classes(name: str) -> None:
    """The edge-case bullet labels are exactly the classes the spec's declarations demand, in canonical order."""
    labels = [label for label, _ in _edge_bullets(_note(_doc(name)))]
    expected = [edge_class.value for edge_class in required_classes(_SPECS[name])]
    assert labels == expected, f"{name}: bullet labels {labels} != required {expected}"


@pytest.mark.parametrize("name", _NAMES)
def test_edge_bullet_phrasing_is_canonical(name: str) -> None:
    """Every edge-case bullet is worded to its class canon (or pinned as a deviant), so the phrasing cannot drift."""
    spec = _SPECS[name]
    for label, text in _edge_bullets(_note(_doc(name))):
        if (name, label) in DEVIANT_BULLETS:
            continue
        assert bullet_matches(spec, label, text), (
            f"{name}: the {label!r} bullet is not canonical.\n"
            f"  got:      {text}\n"
            f"  expected: {expected_bullet(spec, label)}"
        )


# The whitelist of explanatory sub-headers a Note may carry above its ``**Edge-case behavior:**`` list — the exact
# set the corpus uses today. Each is a bold ``**Label:**`` paragraph header naming a documented convention (seeding,
# input handling, a clamp, a sign choice, ...); an unlisted header is an ad-hoc section the guard rejects. Shrink-only.
_PRELIST_LABELS: frozenset[str] = frozenset(
    {
        "Anchoring",
        "Clamp convention",
        "Classification",
        "Composition",
        "Degrees of freedom",
        "Displacement (no lookahead)",
        "Flat start",
        "Gaussian assumption",
        "Historical, not parametric",
        "Inception",
        "Inputs",
        "Long / flat",
        "Moving average",
        "No lookahead (alignment is the caller's)",
        "Off-funding bars",
        "Period rounding",
        "Scaling",
        "Seeding",
        "Sign",
        "Sign convention",
        "Tie-break and seeding",
        "Warm-up",
        "Zero return",
        "Zero-range bars",
    }
)


@pytest.mark.parametrize("name", _NAMES)
def test_prelist_labels_are_whitelisted(name: str) -> None:
    """Every explanatory sub-header above the edge-case list is a sanctioned label — no ad-hoc section names."""
    headers = re.findall(r"\*\*([^*\n]+?):\*\*", _note(_doc(name)))
    offenders = [header for header in headers if header != "Edge-case behavior" and header not in _PRELIST_LABELS]
    assert not offenders, f"{name}: non-whitelisted pre-list sub-headers {offenders}"


# The sanctioned Note openers. The family-level prefixes carry the great majority: an indicator "agrees with its
# independent reference oracle", a reducing metric or a pnl accounting line "the result is checked against an
# independent reference oracle", a rolling metric "each window matches an independent reference oracle (". The
# functions whose oracle can only mirror a path-dependent recurrence (the Ehlers cycle cluster, ``kama``,
# ``parabolic_sar``) open on their own measured wording, pinned per name. Shrink-only.
_OPENER_PREFIXES: tuple[str, ...] = (
    "**Precision** -- agrees with its independent reference oracle",
    "**Correctness** -- the result is checked against an independent reference oracle",
    "**Correctness** -- each window matches an independent reference oracle (",
)
_CYCLE_OPENER = "**Precision** -- the fixed FIR smoothing and quadrature stages are computed independently"
_OPENER_VARIANTS: dict[str, str] = {
    **dict.fromkeys(
        (
            "dominant_cycle_period",
            "dominant_cycle_phase",
            "hilbert_phasor",
            "hilbert_trendline",
            "mama",
            "sine_wave",
            "trend_mode",
        ),
        _CYCLE_OPENER,
    ),
    "kama": (
        "**Precision** -- the efficiency ratio and adaptive smoothing constant are checked against "
        "an independent reference"
    ),
    "parabolic_sar": "**Precision** -- the parabolic SAR is a path-dependent stop-and-reverse recurrence",
}


@pytest.mark.parametrize("name", _NAMES)
def test_note_opener_is_canonical(name: str) -> None:
    """The Note opens with a sanctioned family form (or the function's pinned variant), so the opener cannot drift."""
    opener = _flat(_note(_doc(name)).strip().split("\n\n", 1)[0])
    sanctioned = (_OPENER_VARIANTS[name],) if name in _OPENER_VARIANTS else _OPENER_PREFIXES
    assert any(opener.startswith(prefix) for prefix in sanctioned), f"{name}: non-canonical Note opener: {opener[:120]}"


@pytest.mark.parametrize("name", _NAMES)
def test_raises_section_names_every_counterexample(name: str) -> None:
    """Every validation counterexample's parameter is named in the Raises section, so no guard goes undocumented."""
    spec = _SPECS[name]
    if not spec.raises:
        return
    raises = _flat_raises(name)
    offenders = sorted({param for override, _ in spec.raises for param in override if param not in raises})
    assert not offenders, f"{name}: Raises does not name {offenders}"


# The sentence forms a Returns block states its warm-up in; the captured group is the backticked formula whose
# value, at the canonical parameters, must equal the spec's declared warm-up.
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
    """The Returns warm-up formula reproduces the spec's declared warm-up under the canonical parameters."""
    spec = _SPECS[name]
    if spec.warmup is None or not isinstance(spec.warmup, int):
        return
    text = _returns_block(name)
    formula = next((m.group(1) for p in _WARMUP_PATTERNS if (m := re.search(p, text)) is not None), None)
    params = {key: value for key, value in spec.params.items() if isinstance(value, int)}
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
    assert declared == spec.warmup, (
        f"{name}: the Returns formula ``{formula}`` gives {declared} at the canonical parameters, but the spec "
        f"declares warmup={spec.warmup}"
    )
