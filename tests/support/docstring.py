"""
The docstring generator: the tail of a public function's docstring, built as a pure function of its declaration.

The head of a docstring — the summary line and the LaTeX formula, everything above ``Args:`` — is human-authored and
left untouched. Everything from ``Args:`` to the closing quotes (the *tail*) is a function of the declaration: the
per-parameter ``args_prose`` (falling back to the mined shared table) drives ``Args``, ``returns_body`` drives
``Returns``, ``raises_prose`` the ValueError clause of ``Raises``, the Note opener template plus ``note_extension``,
``notes``, and the whole ``bullets`` list drive ``Note``, and the reference fields plus ``see_also`` supply
``References`` / ``See Also`` verbatim. The Examples section is rendered in full: the import header (with
``example_imports`` and ``example_alias``), each scenario's intro, and — for every declared ``Example`` — its frame,
its call in the canonical style, and the executed output, run at generation so a captured value can never drift from
what the doctest prints.

Only the family-shared prose is a *template mined from the corpus by majority* (the Note opener body per family, the
TypeError line, the shared Args descriptions); the sanctioned opener variant for a rolling metric is keyed off the
declaration. Every per-function remainder is declaration data, so the generator never invents prose — where a datum is
absent it omits the line, and the parity harness records what remains.

:func:`tail_for` returns the tail in *source form* (a four-space base indent, matching the ``.py`` docstring body), so
the regenerator can splice it straight back under the human head.
"""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Mapping, Sequence

import polars as pl

import pomata.indicators
import pomata.metrics
import pomata.pnl
from tests.support.declaration import Declaration, Example, ScalarParam, Shape, build_expr
from tests.support.frames import materialize

# --- layout constants ---

BASE = "    "  # the docstring body's base indent in source (the function body is one level in)
_WIDTH = 120  # the source line budget the corpus prose wraps to

# --- family-keyed template constants (mined from the corpus by majority) ---

# The Note opener paragraph. pnl and metrics open on **Correctness**, indicators on **Precision**; the pnl clause names
# warm-up, the metrics clause does not. A rolling metric opens on the twin-agreement sentence instead.
_OPENER_LABEL = {"pnl": "Correctness", "metrics": "Correctness", "indicators": "Precision"}
_OPENER_BODY = {
    "pnl": (
        "The result is checked against an independent reference oracle on every input, and every edge case (missing "
        "data, boundaries, and warm-up where applicable) is given a defined behavior."
    ),
    "metrics": (
        "The result is checked against an independent reference oracle on every input, and every edge case (missing "
        "data and boundaries) is given a defined behavior."
    ),
    "indicators": (
        "Agrees with its independent reference oracle to ten significant figures (a ``1e-10`` band) on any finite "
        "input within a sane dynamic range; the documentation's *Correctness* page gives the method and the "
        "float-conditioning limit beyond it."
    ),
}

_TYPE_ERROR = "TypeError: If any input is not a ``pl.Expr``."

# The em-dash separating an edge bullet's label from its body: ``- **Label** — sentence.`` (the corpus idiom).
_BULLET_DASH = "—"

# Args parameter descriptions mined from the corpus by majority: a parameter whose description is shared across sites
# (an input-price role, a returns series, the annualization knob) becomes a template constant here. A parameter with no
# dominant description (``window`` alone carries seventeen wordings) is left out and surfaces in the parity report as a
# per-site remainder that must become declaration data.
_ARG_DESCRIPTIONS = {
    "expr": 'Input series, typically a price column (e.g. ``pl.col("close")``).',
    "high": 'High-price series (e.g. ``pl.col("high")``).',
    "low": 'Low-price series (e.g. ``pl.col("low")``).',
    "open": 'Open-price series (e.g. ``pl.col("open")``).',
    "close": 'Close-price series (e.g. ``pl.col("close")``).',
    "volume": 'Traded-volume series (e.g. ``pl.col("volume")``).',
    "weight": (
        "Signed weight, the fraction of capital held (e.g. ``1.0`` fully long, ``-0.5`` half short); "
        "``|weight| > 1`` is leverage."
    ),
    "benchmark": "Benchmark per-bar return series, as fractions, aligned row-for-row with ``returns``.",
    "returns": "Per-bar net return series, as fractions (e.g. from :func:`~pomata.pnl.returns_net`).",
    "periods_per_year": "Observations per year for annualization (canonically ``252`` for daily). Must be ``>= 1``.",
}


# --- wrapping ---


def _tokenize(text: str) -> list[str]:
    """Split ``text`` on spaces, keeping an inline ``code`` span (which may contain spaces) as one atomic token."""
    tokens: list[str] = []
    parts = text.split(" ")
    i = 0
    while i < len(parts):
        part = parts[i]
        if part.count("``") % 2 == 1:  # opens an inline code span; glue words until it closes
            j = i + 1
            while j < len(parts) and "``" not in parts[j]:
                j += 1
            if j < len(parts):
                j += 1
            tokens.append(" ".join(parts[i:j]))
            i = j
        else:
            tokens.append(part)
            i += 1
    return tokens


def _wrap(text: str, indent: str, cont: str) -> list[str]:
    """Greedy word wrap of ``text`` to the source budget: first line at ``indent``, continuations at ``cont``.

    An over-long unbreakable token (a bare URL) stays on its line rather than being pushed to a lone continuation.
    """
    out: list[str] = []
    cur = indent
    started = False
    for word in _tokenize(text):
        if not started:
            cur = indent + word
            started = True
            continue
        candidate = cur + " " + word
        # Break only when moving the word to a fresh continuation line actually keeps it within budget; an unbreakable
        # over-long token (a bare URL) then stays on the current line and overflows, as the corpus renders it.
        if len(candidate) > _WIDTH and len(cont + word) <= _WIDTH:
            out.append(cur)
            cur = cont + word
        else:
            cur = candidate
    out.append(cur)
    return out


def _paragraph(text: str, indent: str) -> list[str]:
    """A wrapped paragraph whose continuation lines share the first line's indent."""
    return _wrap(text, indent, indent)


# --- References / See Also (verbatim from the prose fields) ---


def _references(declaration: Declaration) -> list[str]:
    """The References section: the citation line, then the DOI, then the Wikipedia URL, each an independent bullet."""
    body: list[str] = []
    for entry in (declaration.reference, declaration.doi, declaration.wikipedia, declaration.reference_url):
        if entry:
            body += _wrap(f"- {entry}", BASE + BASE, BASE + BASE + "  ")
    if not body:
        return []
    return [BASE + "References:", *body]


def _family_of(name: str) -> str:
    """The family a public function name belongs to, for a family-aware ``:func:`` qualifier."""
    for module, family in ((pomata.pnl, "pnl"), (pomata.metrics, "metrics"), (pomata.indicators, "indicators")):
        if name in module.__all__:
            return family
    return ""


def _see_also(declaration: Declaration) -> list[str]:
    """The See Also section: each ``(name, clause)`` rendered as a ``:func:`` bullet, family-aware in its qualifier."""
    if not declaration.see_also:
        return []
    body: list[str] = []
    for name, clause in declaration.see_also:
        family = _family_of(name)
        target = name if family == declaration.family else f"~pomata.{family}.{name}"
        body += _wrap(f"- :func:`{target}`: {clause}", BASE + BASE, BASE + BASE + "  ")
    return [BASE + "See Also:", *body]


# --- Note ---


def _opener(declaration: Declaration) -> tuple[str, str]:
    """The Note opener (label, body).

    The label is family-fixed; the body is a bespoke override when declared, else the rolling twin-agreement
    sentence, else the family template.
    """
    label = _OPENER_LABEL[declaration.family]
    if declaration.opener_override:
        return label, declaration.opener_override
    if declaration.rolling_of is not None:
        twin = declaration.rolling_of.name
        return (
            label,
            f"Each window matches an independent reference oracle (the reducing :func:`{twin}` over the window).",
        )
    return label, _OPENER_BODY[declaration.family]


def _extend(opener_body: str, note_extension: str) -> str:
    r"""Join the opener body with its per-function extension.

    A ``\n\n`` prefix opens a new paragraph; otherwise the extension continues the opener's paragraph, joined with a
    single space. An empty extension leaves the opener body untouched.
    """
    if not note_extension:
        return opener_body
    if note_extension.startswith("\n\n"):
        return opener_body + note_extension
    return opener_body + " " + note_extension


def _subheader(label: str, note_body: str) -> list[str]:
    """A ``**Label**`` subheader followed by its blank-line-separated paragraphs (the opener / pre-list note shape)."""
    block = [BASE + BASE + f"**{label}**"]
    for para in note_body.split("\n\n"):
        block.append("")
        block += _paragraph(para, BASE + BASE)
    return block


def _note(declaration: Declaration) -> list[str]:
    """The Note section: the opener (with its per-function extension), the declared pre-list subheaders, the Edge list.

    The opener body is the family template (or the rolling twin-agreement sentence) followed by ``note_extension`` —
    the per-function sentence(s) the corpus appends. The Edge-case bullets are the whole declared ``bullets`` list,
    verbatim and in order (the behavior axes stay independently verified by the rungs; these are only the prose).
    A ``note_postscript`` renders as a standalone paragraph after the Edge-case list where one trails the list.
    """
    label, opener_body = _opener(declaration)
    body = _subheader(label, _extend(opener_body, declaration.note_extension))
    for note_label, note_body in declaration.notes:
        body.append("")
        body += _subheader(note_label, note_body)
    body.append("")
    body.append(BASE + BASE + "**Edge-case behavior**")
    body.append("")
    for bullet_label, bullet_body in declaration.bullets:
        body += _wrap(f"- **{bullet_label}** {_BULLET_DASH} {bullet_body}", BASE + BASE, BASE + BASE + "  ")
    if declaration.note_postscript:
        body.append("")
        body += _paragraph(declaration.note_postscript, BASE + BASE)
    return [BASE + "Note:", *body]


# --- Raises ---


def _raises(declaration: Declaration) -> list[str]:
    """The Raises section: the shared TypeError line, then the per-function ValueError clause where one is declared."""
    body = [BASE + BASE + _TYPE_ERROR]
    if declaration.raises_prose:
        body += _wrap(declaration.raises_prose, BASE + BASE, BASE + BASE + "    ")
    return [BASE + "Raises:", *body]


# --- Returns (the per-function body, verbatim) ---


def _returns(declaration: Declaration) -> list[str]:
    """The Returns section: the declared body, its blank-line-separated paragraphs each greedily wrapped.

    A paragraph that opens a struct's field list (``- ``field`` — ...``) is split back into one bullet per field —
    each field ends its clause with a period before the next ``- ``field`` marker, while any interior ``-`` sits inside
    a code span — and each bullet wraps under a two-space hanging indent so it renders as a list, not a run-on
    paragraph.
    """
    body: list[str] = []
    for i, para in enumerate(declaration.returns_body.split("\n\n")):
        if i:
            body.append("")
        if para.startswith("- "):
            for bullet in re.split(r"(?<=\.) (?=- ``)", para):
                body += _wrap(bullet, BASE + BASE, BASE + BASE + "  ")
        else:
            body += _paragraph(para, BASE + BASE)
    return [BASE + "Returns:", *body]


# --- Args (per-parameter prose overriding the mined shared table) ---


def _args(declaration: Declaration) -> list[str]:
    """The Args section: one entry per factory parameter, from ``args_prose`` where declared, else the mined table.

    Every parameter whose wording is per-function carries an ``args_prose`` entry, and every shared parameter is in
    the mined table, so no parameter is ever left undescribed.
    """
    body: list[str] = []
    for pname in inspect.signature(declaration.factory).parameters:
        description = declaration.args_prose.get(pname) or _ARG_DESCRIPTIONS.get(pname)
        if description is not None:
            body += _wrap(f"{pname}: {description}", BASE + BASE, BASE + BASE + "    ")
    return [BASE + "Args:", *body]


# --- Examples (rendered in the canonical idiom and executed at generation) ---

_EX = BASE + BASE  # the eight-space indent of an Examples ``>>>`` code line


def _partition_literal(labels: tuple[str, ...]) -> str:
    """The panel partition column as the run-length idiom ``["A"] * 5 + ["B"] * 5`` (its labels are contiguous runs)."""
    runs: list[tuple[str, int]] = []
    for label in labels:
        if runs and runs[-1][0] == label:
            runs[-1] = (label, runs[-1][1] + 1)
        else:
            runs.append((label, 1))
    return " + ".join(f'["{label}"] * {count}' for label, count in runs)


def _display_columns(declaration: Declaration, example: Example) -> dict[str, str]:
    """Ordered ``{display column: source literal}`` for the frame dict — the partition first, then the input roles."""
    columns: dict[str, str] = {}
    if example.partition:
        columns[example.partition_col] = _partition_literal(example.partition)
    for role in declaration.inputs:
        name = declaration.example_columns.get(role, role)
        columns[name] = lane_literal(list(example.inputs[role]))
    return columns


def _column_lines(name: str, literal: str) -> list[str]:
    """One frame-dict column: inline when it fits, else a plain-list lane exploded one element per line."""
    inline = _EX + f'...         "{name}": {literal},'
    plain_list = literal.startswith("[") and literal.endswith("]") and " + " not in literal and " * " not in literal
    if len(inline) <= _WIDTH or not plain_list:
        return [inline]
    elements = literal[1:-1].split(", ")
    return [
        _EX + f'...         "{name}": [',
        *(_EX + f"...             {element}," for element in elements),
        _EX + "...         ],",
    ]


def _frame_lines(columns: dict[str, str]) -> list[str]:
    """The ``>>> frame = pl.DataFrame(...)`` lines: inline for a single column that fits, multi-line otherwise."""
    inline = "pl.DataFrame({" + ", ".join(f'"{name}": {literal}' for name, literal in columns.items()) + "})"
    if len(columns) == 1 and len(_EX + ">>> frame = " + inline) <= _WIDTH:
        return [_EX + ">>> frame = " + inline]
    body = [_EX + ">>> frame = pl.DataFrame(", _EX + "...     {"]
    for name, literal in columns.items():
        body += _column_lines(name, literal)
    body += [_EX + "...     }", _EX + "... )"]
    return body


def _call_parts(declaration: Declaration, example: Example) -> tuple[str, list[str]]:
    """The call's ``(name, args)`` — the input columns then the params as keywords — for inline or wrapped rendering."""
    name = declaration.example_alias or declaration.name
    args = [f'pl.col("{declaration.example_columns.get(role, role)}")' for role in declaration.inputs]
    args += [f"{key}={value!r}" for key, value in example.params.items()]
    return name, args


def _call(declaration: Declaration, example: Example) -> str:
    """The factory call ``name(pl.col("close"), window=3)`` — the inputs as columns, then the params as keywords."""
    name, args = _call_parts(declaration, example)
    return f"{name}({', '.join(args)})"


def _example_frame(declaration: Declaration, example: Example) -> pl.DataFrame:
    """The eager ``Float64`` frame a scenario runs against (the partition column stays its own string column)."""
    data: dict[str, pl.Series] = {}
    if example.partition:
        data[example.partition_col] = pl.Series(example.partition_col, list(example.partition))
    for role in declaration.inputs:
        name = declaration.example_columns.get(role, role)
        data[name] = pl.Series(name, list(example.inputs[role]), dtype=pl.Float64)
    return pl.DataFrame(data)


def _round_expr(expr: pl.Expr, round_to: int | None) -> pl.Expr:
    return expr.round(round_to) if round_to is not None else expr


def _execute(declaration: Declaration, example: Example) -> list[str]:
    """The executed output line(s), formatted exactly as the doctest prints them (commitment 2: outputs are truth).

    Builds the same expression the canonical code renders and materializes it, so a captured value can never drift
    from what running the emitted doctest produces: one line per displayed struct field, else the single lane.
    """
    frame = _example_frame(declaration, example)
    expr = declaration.factory(
        *(pl.col(declaration.example_columns.get(role, role)) for role in declaration.inputs),
        **example.params,
    )
    if example.partition:
        expr = expr.over(example.partition_col)
    if declaration.shape is Shape.STRUCT:
        return [
            repr(frame.select(_round_expr(expr.struct.field(name), example.round_to).alias("_"))["_"].to_list())
            for name in example.fields
        ]
    column = frame.select(_round_expr(expr, example.round_to).alias("_"))["_"]
    if declaration.shape is Shape.REDUCING:
        return [repr(column.unique().sort().to_list() if example.partition else column.item())]
    return [repr(column.to_list())]


def _binding_lines(declaration: Declaration, example: Example, suffix: str) -> list[str]:
    """``>>> expr = name(args)suffix`` — one line when it fits, else the call wrapped across continuation lines."""
    name, args = _call_parts(declaration, example)
    flat = f">>> expr = {name}({', '.join(args)}){suffix}"
    if len(_EX + flat) <= _WIDTH:
        return [_EX + flat]
    joined = _EX + "...     " + ", ".join(args)
    if len(joined) <= _WIDTH:
        return [_EX + f">>> expr = {name}(", joined, _EX + f"... ){suffix}"]
    return [_EX + f">>> expr = {name}(", *(_EX + f"...     {arg}," for arg in args), _EX + f"... ){suffix}"]


def _display_line(display: str) -> list[str]:
    """A ``>>> frame.<accessor>(...)[...]`` display line — one line when it fits, else the subscript wrapped."""
    line = _EX + ">>> " + display
    if len(line) <= _WIDTH:
        return [line]
    head, _, rest = display.partition("[")
    key, _, terminal = rest.partition("]")
    return [_EX + ">>> " + head + "[", _EX + "...     " + key, _EX + "... ]" + terminal]


def _scenario_lines(declaration: Declaration, example: Example) -> list[str]:
    """One scenario's rendered ``>>>`` code and executed output — the canonical idiom keyed by shape and partition."""
    if example.verbatim:
        return [_EX + line for line in example.verbatim]
    lines = _frame_lines(_display_columns(declaration, example))
    name = declaration.example_alias or declaration.name
    outputs = _execute(declaration, example)
    over = f'.over("{example.partition_col}")' if example.partition else ""
    rounding = f".round({example.round_to})" if example.round_to is not None else ""
    # a panel broadcasts a series with ``.with_columns``; a reduction (one row per group) uses ``.select``
    accessor = "with_columns" if example.partition and declaration.shape is not Shape.REDUCING else "select"
    if declaration.shape is Shape.STRUCT:
        lines += _binding_lines(declaration, example, "")
        for field_name, output in zip(example.fields, outputs, strict=True):
            field_expr = f'expr{over}.struct.field("{field_name}"){rounding}'
            lines += _display_line(f'frame.{accessor}({field_name}={field_expr})["{field_name}"].to_list()')
            lines.append(_EX + output)
        return lines
    # series / reducing: a ``.item()`` for a printable scalar, a list otherwise (a null reduction prints nothing via
    # ``.item()``, so it is shown as a list). The call is inlined when it fits, else hoisted to an ``expr`` binding.
    null_reduction = declaration.shape is Shape.REDUCING and not example.partition and outputs[0] == "None"
    expression = f"{_call(declaration, example)}{over}{rounding}"

    def display(token: str) -> str:
        if declaration.shape is Shape.REDUCING and not example.partition and not null_reduction:
            return f"frame.select({token}).item()"
        reducing_panel = bool(example.partition) and declaration.shape is Shape.REDUCING
        terminal = "unique().sort().to_list()" if reducing_panel else "to_list()"
        return f'frame.{accessor}({name}={token})["{name}"].{terminal}'

    inline = display(expression)
    if len(_EX + ">>> " + inline) <= _WIDTH:
        lines += _display_line(inline)
    else:
        # Hoist the bare call to ``expr`` and carry ``.over(...).round(...)`` in the display, so the binding is a plain
        # call ruff wraps cleanly (no trailing method chain to reparenthesize) — the idiom the struct branch uses.
        lines += _binding_lines(declaration, example, "")
        lines += _display_line(display(f"expr{over}{rounding}"))
    lines.append(_EX + ("[None]" if null_reduction else outputs[0]))
    return lines


def _examples(declaration: Declaration) -> list[str]:
    """The Examples section: the optional block intro, the import header, then each scenario rendered and executed.

    The import header carries the per-function extra imports (``import math`` for the closed-form cycle checks) and
    the import alias (``as m_squared``). Each scenario is preceded by a blank line and, where declared, its prose
    intro; a bespoke scenario (a computed cycle frame, the struct ``.columns`` demonstration) is emitted verbatim.
    """
    lines = [BASE + "Examples:"]
    if declaration.intro_basic:
        lines += _paragraph(declaration.intro_basic, BASE + BASE)
        lines.append("")  # docutils needs a blank line between the intro paragraph and the ``>>>`` import block
    for statement in declaration.example_imports:
        lines.append(_EX + f">>> {statement}")
    lines.append(_EX + ">>> import polars as pl")
    alias = f" as {declaration.example_alias}" if declaration.example_alias else ""
    lines.append(_EX + f">>> from pomata.{declaration.family} import {declaration.name}{alias}")
    lines.append(_EX + ">>>")
    for index, example in enumerate(declaration.examples):
        if index:  # the first scenario follows the import header directly; later ones are blank-line separated
            lines.append("")
        if example.intro:
            lines += _paragraph(example.intro, BASE + BASE)
            lines.append("")
        lines += _scenario_lines(declaration, example)
    return lines


# --- Examples (single-lane execution helper, for the golden-capture self-test) ---


def lane_literal(values: Sequence[float | None]) -> str:
    """A lane rendered as a Python list literal, spelling out ``None`` / ``nan`` / ``±inf`` so it round-trips."""
    parts: list[str] = []
    for value in values:
        if value is None:
            parts.append("None")
        elif math.isnan(value):
            parts.append('float("nan")')
        elif math.isinf(value):
            parts.append('float("inf")' if value > 0.0 else 'float("-inf")')
        else:
            parts.append(repr(value))
    return "[" + ", ".join(parts) + "]"


def execute_example(
    declaration: Declaration,
    columns: dict[str, list[float | None]],
    overrides: Mapping[str, ScalarParam] | None = None,
) -> list[float | None]:
    """Run the factory over ``columns`` (under the declared params, with ``overrides``) and return the output lane.

    Commitment 2: the example outputs are truth; the generator executes the scenario at generation time so a captured
    value can never drift from the committed one without the parity harness flagging it. ``overrides`` carries a
    scenario's per-call parameters (a golden's ``params`` or a pin's ``params_override``).
    """
    return materialize(columns, build_expr(declaration, **(overrides or {})))


# --- assembly ---


def tail_for(declaration: Declaration) -> str:
    """The docstring tail — ``Args:`` through the closing quotes — as source-form text built from the declaration.

    Reproduces the prose sections (Args, Returns, Raises, the Note opener with its extension and the whole Edge-case
    bullet list, See Also, References) verbatim from the declaration's data and mined templates, and the Examples
    section in full — the import header, each scenario's intro, and every scenario's frame, canonical-style call, and
    executed output.
    """
    sections = [
        _args(declaration),
        _returns(declaration),
        _raises(declaration),
        _note(declaration),
        _see_also(declaration),
        _references(declaration),
        _examples(declaration),
    ]
    return "\n\n".join("\n".join(block) for block in sections if block)
