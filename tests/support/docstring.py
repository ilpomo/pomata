"""
The docstring generator: the tail of a public function's docstring, built as a pure function of its declaration.

The head of a docstring — the summary line and the LaTeX formula, everything above ``Args:`` — is human-authored and
left untouched. Everything from ``Args:`` to the closing quotes (the *tail*) is a function of the declaration: the
per-parameter ``args_prose`` (falling back to the mined shared table) drives ``Args``, ``returns_body`` drives
``Returns``, ``raises_prose`` the ValueError clause of ``Raises``, the Note opener template plus ``note_extension``,
``notes``, and the whole ``bullets`` list drive ``Note``, and the reference fields plus ``see_also`` supply
``References`` / ``See Also`` verbatim. The Examples section supplies the import header (with ``example_imports`` and
``example_alias``) and the scenario intros; the scenario frames, calls, and executed outputs are regenerated in a
following pass once the corpus call style is normalized (the style is not rule-derivable from data today).

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
from collections.abc import Mapping, Sequence

import pomata.indicators
import pomata.metrics
import pomata.pnl
from tests.support.declaration import Declaration, ScalarParam, build_expr
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
        body += _paragraph(declaration.raises_prose, BASE + BASE)
    return [BASE + "Raises:", *body]


# --- Returns (the per-function body, verbatim) ---


def _returns(declaration: Declaration) -> list[str]:
    """The Returns section: the declared body, its blank-line-separated paragraphs each greedily wrapped."""
    body: list[str] = []
    for i, para in enumerate(declaration.returns_body.split("\n\n")):
        if i:
            body.append("")
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


# --- Examples (the import header; the frames and outputs are per-function and executed where derivable) ---


def _examples(declaration: Declaration) -> list[str]:
    """The Examples section: the optional basic intro, the import header, then the ``.over`` / null+NaN intros.

    The import header carries the per-function extra imports (``import math`` for the closed-form cycle checks) and
    the import alias (``as m_squared``). The trio intros come from the declaration verbatim. The scenario frames,
    call lines, and executed outputs are regenerated in sub-PR 2 (the call style is not rule-derivable from data —
    a function mixes positional and keyword window arguments across its own examples — so it is normalized first).
    """
    lines = [BASE + "Examples:"]
    if declaration.intro_basic:
        lines += _paragraph(declaration.intro_basic, BASE + BASE)
    for statement in declaration.example_imports:
        lines.append(BASE + BASE + f">>> {statement}")
    lines.append(BASE + BASE + ">>> import polars as pl")
    alias = f" as {declaration.example_alias}" if declaration.example_alias else ""
    lines.append(BASE + BASE + f">>> from pomata.{declaration.family} import {declaration.name}{alias}")
    lines.append(BASE + BASE + ">>>")
    for intro in (declaration.intro_over, declaration.intro_missing):
        if intro:
            lines += _paragraph(intro, BASE + BASE)
    return lines


# --- Examples (regenerated and executed) ---


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
    bullet list, See Also, References) verbatim from the declaration's data and mined templates. The Examples section
    reproduces the import header and the scenario intros; the scenario frames, calls, and executed outputs are the
    remaining per-function content, regenerated in sub-PR 2 once the corpus call style is normalized.
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
