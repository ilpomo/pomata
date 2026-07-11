"""
The test-naming and presence grammar, enforced against the declared policies.

Four guarantees in one source-only walk, keyed on the public surface:

- **presence** -- every public function's test file carries at least one interior-``null`` test, one interior-``NaN``
  test, and one ``matches_reference`` test. A function shipped without one of these edge anchors is a red build, not the
  next audit's finding.
- **canonical names do not lie** -- a reserved ``test_null_*`` / ``test_nan_*`` name is used only by a function whose
  declared policy (:mod:`tests.support.policies`) it names, and a scale-family rung is spelled from the one scale
  vocabulary and sits in a ``Test*Properties`` class. Descriptive per-input names (``test_null_in_volume_propagates``)
  are left free; only the canonical ones are held.
- **null precedes nan** -- within a function's test file the interior-``null`` flow anchor comes before the
  interior-``NaN`` one, the canonical Edge order (tests/README.md §4). A file that runs its ``nan`` anchor first
  is a red build.
- **missing precedes scale** -- within a function's ``Test*Properties`` class the
  ``matches_reference_under_missing_data`` rung comes before the scale rung, the canonical Properties order
  (tests/README.md §4). A file that runs its scale rung first is a red build.

Scale carries no presence mandate: a dimensionless ratio or a rolling metric legitimately has no scale test, so scale is
name-only. All detectors are ``test_``-scoped, so a helper like ``apply_domi``**``nan``**``t_cycle_period`` cannot be
mistaken for a NaN test.
"""

import ast
from pathlib import Path

import pytest
from tests.support.policies import POLICIES, NanPolicy, NullPolicy

from pomata import indicators, metrics, pnl

_TESTS_ROOT = Path(__file__).parent
_PACKAGES = {"indicators": indicators, "pnl": pnl, "metrics": metrics}

_RESERVED_NULL = {
    "test_null_skipped": NullPolicy.SKIPPED,
    "test_null_propagates": NullPolicy.PROPAGATES,
    "test_null_in_window_is_null": NullPolicy.IN_WINDOW_IS_NULL,
    "test_null_absorbed": NullPolicy.ABSORBED,
    "test_null_bridged": NullPolicy.BRIDGED,
    "test_null_latches": NullPolicy.LATCHES,
}
_RESERVED_NAN = {
    "test_nan_poisons": NanPolicy.POISONS,
    "test_nan_propagates": NanPolicy.PROPAGATES,
    "test_nan_latches": NanPolicy.LATCHES,
}
_SCALE_CANONICAL = frozenset(
    {
        "test_scale_homogeneity",
        "test_scale_invariance",
        "test_price_scale_homogeneity",
        "test_price_scale_invariance",
        "test_volume_scale_homogeneity",
        "test_volume_scale_invariance",
        "test_additive_shift_invariance",
    }
)
_SCALE_WORDS = ("homogen", "invari", "shift")


def _test_file(name: str) -> Path:
    for family, package in _PACKAGES.items():
        if name in package.__all__:
            return _TESTS_ROOT / family / f"test_{name}.py"
    raise KeyError(name)


def _methods(path: Path) -> list[tuple[str, str]]:
    """Every ``(class, method)`` defined inside a class in a test module."""
    module = ast.parse(path.read_text(encoding="utf-8"))
    return [
        (node.name, item.name)
        for node in module.body
        if isinstance(node, ast.ClassDef)
        for item in node.body
        if isinstance(item, ast.FunctionDef)
    ]


def _is_null(method: str) -> bool:
    return (
        method.startswith("test_")
        and "null" in method
        and not any(x in method for x in ("all_null", "warmup", "warm_up"))
    )


def _is_nan(method: str) -> bool:
    return method.startswith("test_") and "nan" in method


def _is_scale(method: str) -> bool:
    return method.startswith("test_") and (method.startswith("test_scale_") or any(w in method for w in _SCALE_WORDS))


def _is_scale_canonical(method: str) -> bool:
    return method in _SCALE_CANONICAL or method.startswith("test_scale_homogeneity_in_")


_METHODS: dict[str, list[tuple[str, str]]] = {name: _methods(_test_file(name)) for name in POLICIES}


def _reserved_usages() -> list[tuple[str, str, object, object]]:
    usages: list[tuple[str, str, object, object]] = []
    for name in sorted(POLICIES):
        null_policy, nan_policy = POLICIES[name]
        for _, method in _METHODS[name]:
            if method in _RESERVED_NULL:
                usages.append((name, method, _RESERVED_NULL[method], null_policy))
            elif method in _RESERVED_NAN:
                usages.append((name, method, _RESERVED_NAN[method], nan_policy))
    return usages


def _scale_usages() -> list[tuple[str, str, str]]:
    return [(name, cls, method) for name in sorted(POLICIES) for cls, method in _METHODS[name] if _is_scale(method)]


_RESERVED = _reserved_usages()
_SCALE = _scale_usages()


def test_the_scan_covered_every_function() -> None:
    """Guards the guard: every function's test file was parsed, and reserved / scale usages were actually found."""
    assert len(_METHODS) == len(POLICIES)
    assert _RESERVED
    assert _SCALE


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_interior_null_test_present(name: str) -> None:
    """Verifies the function's test file carries at least one interior-``null`` test."""
    assert any(_is_null(method) for _, method in _METHODS[name]), f"{name}: no interior-null test"


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_interior_nan_test_present(name: str) -> None:
    """Verifies the function's test file carries at least one interior-``NaN`` test."""
    assert any(_is_nan(method) for _, method in _METHODS[name]), f"{name}: no interior-nan test"


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_matches_reference_present(name: str) -> None:
    """Verifies the function's test file carries at least one ``matches_reference`` test."""
    assert any(method.startswith("test_matches_reference") for _, method in _METHODS[name]), (
        f"{name}: no reference test"
    )


@pytest.mark.parametrize(
    ("name", "method", "reserved", "declared"), _RESERVED, ids=[f"{n}::{m}" for n, m, _, _ in _RESERVED]
)
def test_reserved_name_matches_policy(name: str, method: str, reserved: object, declared: object) -> None:
    """Verifies a canonical ``test_null_*`` / ``test_nan_*`` name is used only by a function declaring its policy."""
    assert reserved is declared, f"{name}: {method} is canonical for {reserved} but {name} declares {declared}"


@pytest.mark.parametrize(("name", "cls", "method"), _SCALE, ids=[f"{n}::{m}" for n, _, m in _SCALE])
def test_scale_name_is_canonical(name: str, cls: str, method: str) -> None:
    """Verifies every scale-family rung is drawn from the scale vocabulary and sits in a ``Test*Properties`` class."""
    assert _is_scale_canonical(method), f"{name}: {method} is not a canonical scale-rung name"
    assert cls.endswith("Properties"), f"{name}: {method} sits in {cls}, not a *Properties class"


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_null_anchor_precedes_nan_anchor(name: str) -> None:
    """Verifies a function's interior-``null`` flow anchor precedes its interior-``NaN`` one (the Edge order, §4)."""
    methods = [method for _, method in _METHODS[name]]
    null_index = next((index for index, method in enumerate(methods) if method.startswith("test_null_")), None)
    nan_index = next((index for index, method in enumerate(methods) if method.startswith("test_nan_")), None)
    if null_index is not None and nan_index is not None:
        assert null_index < nan_index, (
            f"{name}: the null anchor ({methods[null_index]}) must precede the nan anchor ({methods[nan_index]})"
        )


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_missing_data_precedes_scale(name: str) -> None:
    """Verifies the ``matches_reference_under_missing_data`` rung precedes the scale rung (the Properties order, §4)."""
    methods = [method for _, method in _METHODS[name]]
    missing_index = next(
        (index for index, method in enumerate(methods) if method == "test_matches_reference_under_missing_data"),
        None,
    )
    scale_index = next((index for index, method in enumerate(methods) if _is_scale(method)), None)
    if missing_index is not None and scale_index is not None:
        assert missing_index < scale_index, (
            f"{name}: the missing-data rung ({methods[missing_index]}) must precede "
            f"the scale rung ({methods[scale_index]})"
        )


# ---- The full within-tier order and the one-rung-one-name law (§4), enforced in full ------------------------------

# One rung, one spelling (§4's naming law): the left spelling is forbidden, the right one is canonical. `window_one`
# is the one deliberately suffixed stem (the rung states the identity it collapses to), so its PLAIN form is the
# forbidden one; every other stem is plain, so its suffixed forms are forbidden.
_FORBIDDEN_SPELLINGS: dict[str, str] = {
    "test_single_row_is_nan": "test_single_row",
    "test_single_row_is_zero": "test_single_row",
    "test_single_row_is_one": "test_single_row",
    "test_single_row_is_null": "test_single_row",
    "test_single_row_starts_at_zero": "test_single_row",
    "test_window_one": "test_window_one_<identity>",
    "test_infinity_propagates": "test_consecutive_infinities_make_nan",
    "test_unordered_windows_raise": "test_misordered_windows_raise",
    "test_fast_exceeds_slow_raises": "test_fast_above_slow_raises",
    "test_window_fast_above_slow_raises": "test_fast_above_slow_raises",
    "test_windows_below_one_raises": "test_window_below_one_raises",
    "test_golden_master_adjust": "test_golden_master_adjusted",
    "test_matches_reference_adjust": "test_matches_reference_adjusted",
    "test_constant_series_is_constant": "test_constant_series",
}


# Literal-prefix ranks for the §4 Edge ladder; the prefixes are mutually exclusive, so lookup order is irrelevant.
_EDGE_PREFIX_RANKS: tuple[tuple[str, int], ...] = (
    ("test_empty", 1),
    ("test_all_null", 3),
    ("test_null", 4),
    ("test_nan", 5),
    ("test_warmup", 6),
    ("test_no_warmup", 6),
    ("test_window_exceeds_length", 7),
    ("test_window_equals_length", 8),
    ("test_window_one", 9),
    ("test_constant_window", 10),
)


def _edge_rank(method: str) -> int:
    """The §4 Edge rank of a canonical rung; a bespoke (function-specific) rung ranks last by design."""
    if method.endswith(("_raises", "_raise")):
        return 0
    if method == "test_single_row":
        return 2
    for prefix, rank in _EDGE_PREFIX_RANKS:
        if method.startswith(prefix):
            return rank
    return 11  # bespoke / singularity guard: after the shared ladder


def _contract_rank(method: str) -> int:
    if method == "test_returns_expr":
        return 0
    if method.startswith("test_lazy_eager"):
        return 2
    if method.startswith("test_over_"):
        return 3
    return 4


def _correctness_rank(method: str) -> int:
    if method.startswith("test_matches_reference"):
        return 0
    if method.startswith("test_golden_master"):
        return 1
    return 2


def _properties_rank(method: str) -> int:
    if method == "test_matches_reference_for_any_input":
        return 0
    if method == "test_matches_reference_under_missing_data":
        return 1
    if _is_scale(method):
        return 2
    if method == "test_matches_reference_at_large_magnitude":
        return 3
    return 4


_TIER_RANKS = {
    "Contract": _contract_rank,
    "Edge": _edge_rank,
    "Correctness": _correctness_rank,
    "Properties": _properties_rank,
}
_TIER_ORDER = ("Contract", "Edge", "Correctness", "Properties")


def _tier_of(cls: str) -> str | None:
    for tier in _TIER_ORDER:
        if cls.endswith(tier):
            return tier
    return None


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_no_forbidden_spelling(name: str) -> None:
    """Verifies every rung uses its one canonical spelling (§4's naming law)."""
    for _, method in _METHODS[name]:
        assert method not in _FORBIDDEN_SPELLINGS, (
            f"{name}: {method} is a forbidden spelling; the canonical rung is {_FORBIDDEN_SPELLINGS[method]}"
        )
        assert not method.startswith("test_bounded_in_"), (
            f"{name}: {method} is a forbidden spelling; the canonical rung is test_bounded (the docstring names the "
            f"range)"
        )


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_tier_classes_in_order(name: str) -> None:
    """Verifies the four tier classes appear in the canonical Contract -> Edge -> Correctness -> Properties order."""
    tiers = [tier for cls, _ in _METHODS[name] if (tier := _tier_of(cls)) is not None]
    seen: list[str] = []
    for tier in tiers:
        if not seen or seen[-1] != tier:
            seen.append(tier)
    assert seen == sorted(set(seen), key=_TIER_ORDER.index), f"{name}: tier classes run {seen}"


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_within_tier_rung_order(name: str) -> None:
    """Verifies each tier lays its rungs out in the §4 order (bespoke rungs close the tier, after the shared ladder)."""
    for tier in _TIER_ORDER:
        methods = [method for cls, method in _METHODS[name] if _tier_of(cls) == tier]
        ranks = [_TIER_RANKS[tier](method) for method in methods]
        assert ranks == sorted(ranks), (
            f"{name}: the {tier} tier runs {methods} (ranks {ranks}); the §4 order is non-decreasing"
        )
