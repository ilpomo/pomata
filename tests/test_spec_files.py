"""
The spec-file conventions, swept from the registry: a spec is pure data, so its FORM is part of the contract — a
reason that stops telling the truth, a declared band without its rationale, or a file off the canonical path is
drift the ladder itself cannot see. Each sweep enumerates 100% of its site set: the registry (``ALL_SPECS``) for
the declared values, and the ``tests/<family>/<name>.py`` glob for the file-level form, with the two site sets
pinned to each other by the layout bijection below.
"""

import ast
import re
from pathlib import Path

import pytest
from tests.all_specs import ALL_SPECS
from tests.support.spec import ScaleExempt, Spec, spec_id

_TESTS_DIR = Path(__file__).parent

# One file per function under tests/<family>/<name>.py: the canonical home. The all_specs.py bijection checks the
# spec OBJECTS against the public __all__, not where they live — this glob closes that blind spot.
_SPEC_FILES: dict[str, Path] = {
    path.stem: path
    for family in ("indicators", "metrics", "pnl")
    for path in sorted((_TESTS_DIR / family).glob("*.py"))
    if path.stem != "test_bespoke"
}
_SPEC_FILE_NAMES = sorted(_SPEC_FILES)

# Claims about what the harness can or cannot reach do not belong in a declared reason: the harness moves (a probe
# retuned, a fuzz domain widened, a rung added) and the prose silently rots. A reason justifies its case on the
# case's own terms; a conditioning cut states its measured boundary, never a capability claim about the fuzz, the
# probe, or the rungs. The patterns bind each claim shape to a harness subject, so a domain fact about the kernel
# ("the residual never reaches this output") or a frozen golden stays legal; ``\s+`` joins every word because a
# reason may wrap across concatenated string literals with incidental double spaces.
_HARNESS_CAPABILITY_CLAIMS: tuple[str, ...] = (
    r"\bunreachable\s+by\s+the\s+(?:fuzz|probe)\b",
    r"\bthe\s+(?:fuzz|probe|property\s+tiers?|generic\s+rungs?)\b[^;—]*\bnever\b",
    r"\bnever\b[^;—]*\bby\s+(?:the\s+|any\s+)?(?:fuzz|probe|property\s+tier|generic\s+rung)",
    r"\bthe\s+(?:fuzz|probe)\s+\w+\s+is\b",
    r"\b(?:fuzz|probe)\b[^;—]*\bpositive-only\b",
    r"\bpositive-only\b[^;—]*\b(?:fuzz|probe)\b",
    r"\bthe\s+probe\b[^;—]*\balways\s+positive\b",
    r"\botherwise\s+untested\b",
    r"\buntested\s+by\b",
    r"\bnever\s+reached\s+by\b",
    r"\b(?:fuzz|probe)\b[^;—]*\bcannot\b",
    r"\bkeeps?\s+out\s+of\s+the\s+property\s+(?:tier|fuzz)\b",
)
_CLAIM_PATTERN = re.compile("|".join(_HARNESS_CAPABILITY_CLAIMS), re.IGNORECASE)

# The canonical constructor order of a spec file — reader-optimized, NOT the dataclass field order: identity
# (factory, inputs, params, shape, fields), the windowed contract (warmup, lands_on, raises, flow_horizon), the
# oracle block kept whole (the adapter, the declared bands, then the property-tier filter that closes it), the scale
# claim, the declared deviations, the golden block kept whole (params, input, output, rounding), the recomposition,
# and the pins last. Every spec file lists its keywords as an in-order subsequence of this tuple; a new field earns
# its slot here the first time a spec uses it.
_CANONICAL_KWARGS: tuple[str, ...] = (
    "factory",
    "inputs",
    "params",
    "shape",
    "fields",
    "warmup",
    "lands_on",
    "raises",
    "flow_horizon",
    "oracle",
    "oracle_adapter",
    "oracle_rel_tol",
    "oracle_abs_tol",
    "conditioning",
    "scale",
    "all_null",
    "flow_deviation",
    "golden_params",
    "golden_input",
    "golden_output",
    "golden_round",
    "component_expr",
    "cost_degree",
    "pins",
)
_CANONICAL_ORDER: dict[str, int] = {key: index for index, key in enumerate(_CANONICAL_KWARGS)}

_SNAKE_CASE = re.compile(r"[a-z0-9]+(_[a-z0-9]+)*")


def _declared_reasons(spec: Spec) -> list[tuple[str, str]]:
    """Every free-text reason a spec declares, labeled by where it lives."""
    reasons = [(f"pin {pin.label!r}", pin.reason) for pin in spec.pins]
    if isinstance(spec.scale, ScaleExempt):
        reasons.append(("scale exemption", spec.scale.reason))
    if spec.all_null is not None:
        reasons.append(("all_null deviant", spec.all_null.reason))
    if spec.flow_deviation:
        reasons.append(("flow deviation", spec.flow_deviation))
    return reasons


def test_spec_files_mirror_the_registry() -> None:
    """Verifies the one-file-per-function layout: the ``tests/<family>/<name>.py`` stems equal the registry names."""
    assert sorted(spec.name for spec in ALL_SPECS) == _SPEC_FILE_NAMES


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_reasons_state_no_harness_capability(spec: Spec) -> None:
    """Verifies no declared reason claims what the fuzz or the probe can reach — the drift-prone sentence class."""
    for where, reason in _declared_reasons(spec):
        match = _CLAIM_PATTERN.search(reason)
        assert match is None, f"{where}: the harness-capability claim {match.group(0)!r} does not belong in a reason"


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_pin_labels_are_snake_case(spec: Spec) -> None:
    """Verifies every pin label is snake_case, so the derived pytest id stays readable and grep-able."""
    for pin in spec.pins:
        assert _SNAKE_CASE.fullmatch(pin.label), f"pin label {pin.label!r} is not snake_case"


@pytest.mark.parametrize("spec", ALL_SPECS, ids=spec_id)
def test_pin_reasons_carry_no_trailing_period(spec: Spec) -> None:
    """Verifies pin reasons stay lowercase clauses without a terminating period, like every declared reason."""
    for pin in spec.pins:
        assert not pin.reason.strip().endswith("."), f"pin {pin.label!r}: the reason ends with a period"


@pytest.mark.parametrize("name", _SPEC_FILE_NAMES)
def test_module_docstring_names_its_function(name: str) -> None:
    """Verifies each spec file opens with the canonical one-line banner naming its public function."""
    path = _SPEC_FILES[name]
    docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    prefix = f"Spec for ``pomata.{path.parent.name}.{name}`` — "
    assert docstring is not None, "the spec file has no module docstring"
    assert docstring.startswith(prefix), f"docstring does not open with {prefix!r}"


@pytest.mark.parametrize("name", _SPEC_FILE_NAMES)
def test_spec_constant_names_its_function(name: str) -> None:
    """Verifies the module-level ``Spec`` assignment target is the upper-cased function name."""
    tree = ast.parse(_SPEC_FILES[name].read_text(encoding="utf-8"))
    targets = [
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "Spec"
        for target in node.targets
        if isinstance(target, ast.Name)
    ]
    assert targets == [name.upper()], f"Spec constant {targets} != [{name.upper()!r}]"


@pytest.mark.parametrize("name", _SPEC_FILE_NAMES)
def test_spec_kwargs_follow_the_field_order(name: str) -> None:
    """Verifies every ``Spec(...)`` call lists its keywords as an in-order subsequence of the dataclass fields."""
    tree = ast.parse(_SPEC_FILES[name].read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Spec"
    ]
    assert calls, "no Spec(...) call found"
    for call in calls:
        keys = [keyword.arg for keyword in call.keywords if keyword.arg is not None]
        indices = [_CANONICAL_ORDER[key] for key in keys]
        assert indices == sorted(indices), f"keywords off the canonical field order: {keys}"


@pytest.mark.parametrize("name", _SPEC_FILE_NAMES)
def test_declared_bands_carry_a_rationale(name: str) -> None:
    """Verifies every ``oracle_*_tol`` departure sits under a one-line comment saying why (CORRECTNESS.md's rule)."""
    lines = _SPEC_FILES[name].read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith(("oracle_rel_tol=", "oracle_abs_tol=")):
            continue
        above = lines[index - 1].lstrip()
        assert above.startswith(("#", "oracle_rel_tol=")), f"line {index + 1}: a declared band without its rationale"
