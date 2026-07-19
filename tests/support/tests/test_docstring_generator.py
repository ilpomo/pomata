"""
Self-tests of :mod:`tests.support.docstring` — the declaration-driven docstring-tail generator.

These pin the mechanism, not the whole corpus (the corpus fidelity is held by the round-trip guard): the greedy wrap and
its atomic code spans, the family- and shape-keyed template selection, the family-aware See Also qualifier, the
References ordering, that the generator emits only lines present verbatim in the source it reproduces, that the tail
splits and re-assembles without disturbing the human head, and that the executed-example capture reproduces every
committed golden output.
"""

import math
import re

import pytest

from tests.all_declarations import ALL_DECLARATIONS  # importing runs every ``suite_*`` registration
from tests.regenerate_docstrings import current_tail, locate, rewrite
from tests.support.declaration import Declaration, Shape
from tests.support.docstring import execute_example, lane_literal, tail_for
from tests.support.registry import registry_for

_NON_STRUCT_GOLDENS = [d for d in ALL_DECLARATIONS if d.golden is not None and d.shape is not Shape.STRUCT]


# --- wrapping (exercised through the public tail) ---


def test_generated_lines_fit_the_source_budget_or_are_lone_urls() -> None:
    """No generated line exceeds the 120-column budget, except a bullet holding a single unbreakable URL."""
    for declaration in ALL_DECLARATIONS:
        for line in tail_for(declaration).splitlines():
            assert len(line) <= 120 or line.strip().startswith("- http"), (
                f"{declaration.name}: generated line exceeds the 120-column budget: {line!r}"
            )


def test_wrapped_reference_continuation_is_indented_and_atomic() -> None:
    """A long citation wraps at the 10-column continuation indent without splitting an inline code span."""
    tail = tail_for(registry_for("pnl")["turnover"])
    assert "          Superior Returns and Controlling Risk* (2nd ed.). McGraw-Hill." in tail


# --- template selection ---


def test_opener_is_family_keyed() -> None:
    """The pnl and metrics openers use Correctness (pnl naming warm-up); indicators use Precision."""
    pnl = tail_for(registry_for("pnl")["turnover"])
    metrics = tail_for(registry_for("metrics")["total_return"])
    indicators = tail_for(registry_for("indicators")["sma"])
    assert "**Correctness**" in pnl
    assert "checked against an independent reference oracle on every input" in " ".join(pnl.split())
    assert "warm-up where applicable" in " ".join(pnl.split())  # the pnl-only clause
    assert "warm-up where applicable" not in " ".join(metrics.split())
    assert "**Precision**" in indicators


def test_rolling_metric_opens_on_the_twin_sentence() -> None:
    """A rolling twin opens on the window-agreement sentence naming its reducing twin, not the family opener."""
    tail = tail_for(registry_for("metrics")["alpha_rolling"])
    assert "Each window matches an independent reference oracle (the reducing :func:`alpha` over the window)." in tail


def test_edge_bullets_are_emitted_from_declaration_data() -> None:
    """Every declared edge bullet is rendered ``- **Label** — body``, in order, including the Partitioning bullet."""
    tail = tail_for(registry_for("indicators")["sma"])
    assert "- **Null** — a window containing a ``null`` yields ``null``" in tail
    assert "- **window == 1** — the one-point mean is the input itself, so the SMA reproduces the input." in tail
    assert "**Partitioning** — wrap the call in ``.over(...)`` for a multi-series panel" in tail


def test_note_postscript_renders_after_the_edge_list() -> None:
    """A ``note_postscript`` renders as a standalone paragraph after the Edge-case list, not folded into a bullet."""
    lines = tail_for(registry_for("indicators")["trend_mode"]).splitlines()
    edge = next(i for i, line in enumerate(lines) if line.strip() == "**Edge-case behavior**")
    postscript = next(i for i, line in enumerate(lines) if "The underlying phase branch guards" in line)
    assert postscript > edge  # after the Edge-case bullets, not before
    assert lines[postscript].startswith("        The underlying phase branch")  # base indent: a standalone paragraph
    assert lines[postscript - 1] == ""  # blank-line separated from the bullet list


def test_returns_body_and_raises_prose_are_emitted() -> None:
    """The Returns body and the per-function ValueError clause are rendered verbatim from declaration data."""
    tail = tail_for(registry_for("indicators")["sma"])
    assert "The SMA for each row, the same length as ``expr``." in tail
    assert "    ValueError: If ``window < 1``." in tail


def test_args_prose_overrides_the_shared_table() -> None:
    """A per-function ``args_prose`` entry drives its parameter; a shared parameter falls back to the mined table."""
    tail = tail_for(registry_for("indicators")["sma"])
    assert "window: Number of observations in the moving window. Must be ``>= 1``." in tail  # args_prose
    assert 'expr: Input series, typically a price column (e.g. ``pl.col("close")``).' in tail  # mined template


def test_opener_extension_and_override() -> None:
    """A note_extension continues the family opener; a bespoke opener replaces it entirely."""
    extended = tail_for(registry_for("indicators")["adx"])
    assert "float-conditioning limit beyond it. It is scale-invariant under a positive common rescaling" in " ".join(
        extended.split()
    )
    override = tail_for(registry_for("indicators")["dominant_cycle_period"])
    assert "**Precision**" in override
    assert "The fixed FIR smoothing and quadrature stages are computed independently" in override


def test_reference_url_is_the_fourth_bucket() -> None:
    """A non-DOI / non-Wikipedia reference URL is rendered as its own References bullet."""
    tail = tail_for(registry_for("indicators")["aroon"])
    assert "- https://chartschool.stockcharts.com/" in tail


def test_examples_header_carries_extra_imports_and_alias() -> None:
    """The Examples import header renders the extra imports and the import alias from declaration data."""
    cycle = tail_for(registry_for("indicators")["dominant_cycle_period"])
    assert "        >>> import math" in cycle
    assert "        >>> import polars as pl" in cycle
    aliased = tail_for(registry_for("metrics")["modigliani_risk_adjusted_performance"])
    assert ">>> from pomata.metrics import modigliani_risk_adjusted_performance as m_squared" in aliased


def test_see_also_qualifier_is_family_aware() -> None:
    """A See Also target is bare within the family and ``~pomata.<family>.<name>`` across families."""
    tail = tail_for(registry_for("metrics")["total_return"])
    assert "- :func:`cagr`:" in tail  # same family: bare
    assert "- :func:`~pomata.pnl.equity_curve`:" in tail  # cross family: qualified


def test_references_are_ordered_reference_doi_wikipedia() -> None:
    """When present, the citation precedes the DOI, which precedes the Wikipedia URL."""
    declaration = next(d for d in ALL_DECLARATIONS if d.reference and d.doi and d.wikipedia)
    tail = tail_for(declaration).splitlines()
    idx_ref = next(i for i, line in enumerate(tail) if declaration.reference[:20] in line)
    idx_doi = next(i for i, line in enumerate(tail) if declaration.doi in line)
    idx_wiki = next(i for i, line in enumerate(tail) if declaration.wikipedia in line)
    assert idx_ref < idx_doi < idx_wiki


# --- structure and byte-faithfulness ---


def test_tail_has_the_seven_headers_in_order() -> None:
    """Every tail carries the seven Google sections, in canonical order."""
    tail = tail_for(registry_for("indicators")["sma"])
    positions = [tail.index(f"    {header}") for header in ("Args:", "Returns:", "Raises:", "Note:")]
    positions += [tail.index(f"    {header}") for header in ("See Also:", "References:", "Examples:")]
    assert positions == sorted(positions)


def test_generator_invents_no_prose() -> None:
    """Every generated line's content appears verbatim in the source — the generator may re-wrap, never invent.

    A line-for-line match is not required — only the *content*: every generated line must be a contiguous fragment of
    the source, so the generator never fabricates a sentence a declaration field does not hold.
    """
    declaration = registry_for("indicators")["sma"]
    doc = declaration.factory.__doc__ or ""
    source = re.sub(r"\s+", " ", doc)
    for line in tail_for(declaration).splitlines():
        collapsed = " ".join(line.split())
        assert not collapsed or collapsed in source


def test_type_error_line_is_the_shared_constant() -> None:
    """Every tail's Raises section carries the one shared TypeError line."""
    for family in ("pnl", "metrics", "indicators"):
        name = next(iter(registry_for(family)))
        assert "    TypeError: If any input is not a ``pl.Expr``." in tail_for(registry_for(family)[name])


# --- head preservation (the regenerator's split / re-assemble) ---


def test_locate_splits_at_args_and_rewrite_preserves_the_head() -> None:
    """The tail split starts at ``Args:`` and re-assembling with the current tail reproduces the file byte-for-byte."""
    declaration = registry_for("indicators")["sma"]
    span = locate(declaration)
    assert span.lines[span.tail_start].strip() == "Args:"
    round_trip = rewrite(span, "\n".join(current_tail(span)))
    assert round_trip == span.path.read_text(encoding="utf-8")


# --- executed-example capture (the outputs are truth) ---


@pytest.mark.parametrize("declaration", _NON_STRUCT_GOLDENS, ids=lambda d: d.name)
def test_execution_capture_reproduces_the_golden(declaration: Declaration) -> None:
    """Running the factory over the golden inputs reproduces the committed golden output, rounded as declared."""
    golden = declaration.golden
    assert golden is not None
    assert isinstance(golden.output, tuple)  # a non-struct golden's output is a single lane
    columns = {role: list(golden.inputs[role]) for role in declaration.inputs}
    captured = execute_example(declaration, columns, overrides=golden.params)
    rounded = [_round(value, golden.round_to) for value in captured]
    expected = [_round(value, golden.round_to) for value in golden.output]
    assert _lanes_equal(rounded, expected)


def test_lane_literal_spells_out_none_nan_and_infinities() -> None:
    """A lane literal round-trips ``None``, ``NaN``, and ``±inf`` as source that re-parses to the same values."""
    rendered = lane_literal([1.0, None, math.nan, math.inf, -math.inf])
    assert rendered == '[1.0, None, float("nan"), float("inf"), float("-inf")]'


# --- helpers ---


def _round(value: float | None, digits: int) -> float | None:
    """Round a lane value, leaving ``None`` and non-finite values untouched (they never round)."""
    if value is None or math.isnan(value) or math.isinf(value):
        return value
    return round(value, digits)


def _lanes_equal(left: list[float | None], right: list[float | None]) -> bool:
    """Lane equality with ``NaN == NaN`` (a captured ``NaN`` matches a committed ``NaN``)."""
    if len(left) != len(right):
        return False
    return all(
        (a is None and b is None) or (a is not None and b is not None and math.isnan(a) and math.isnan(b)) or a == b
        for a, b in zip(left, right, strict=True)
    )
