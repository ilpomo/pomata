"""
Source-only conformance guards for the public docstring template.

Every public factory's docstring follows one template: the seven Google sections in one order, the Args entries in
signature order, the byte-identical ``TypeError`` line, the family Note opener (**Precision** for indicators,
**Correctness** for metrics and pnl), the edge-case bullet order (Null before NaN, Partitioning last), the latch
marker every ``NanPolicy.LATCHES`` docstring must carry, the canonical Returns opener per output shape, and
byte-identical Args prose for every shared parameter (with the sanctioned per-role deviants pinned by name). None of
this is reachable by ruff's pydocstyle shell checks or the doctest gate, so it is proven here, from the source, the
same way the spec ladder proves the suite's own declarations at import.
"""

import inspect
import re
from types import ModuleType

import polars as pl
import pytest
from tests.support import COLUMN_X, synthesize_call

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
def test_edge_bullets_ordered(name: str) -> None:
    """In the edge-case list the Null bullet precedes the NaN bullet, and Partitioning closes the list."""
    labels = [label for label, _ in _bullets(_note(_doc(name)))]
    assert labels, f"{name}: no Note bullets parsed"
    null_at = next((index for index, label in enumerate(labels) if label.startswith("Null")), None)
    nan_at = next((index for index, label in enumerate(labels) if label.startswith("NaN")), None)
    if null_at is not None and nan_at is not None:
        assert null_at < nan_at, f"{name}: NaN bullet before Null bullet"
    if "Partitioning" in labels:
        assert labels[-1] == "Partitioning", f"{name}: Partitioning is not the last bullet"


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
