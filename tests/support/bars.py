"""
Bar-tuple unzipping: transpose a list of bars into aligned per-column lists.

The ``coherent_*`` strategies yield one bar as a tuple; a property test draws a list of them and then unzips that list
into the per-column lists the materialize adapters consume. These ``split_*`` helpers are that transposition, one per
bar arity.
"""

from collections.abc import Sequence


def split_pairs[T](rows: Sequence[tuple[T, T]]) -> tuple[list[T], list[T]]:
    """
    Unzip a list of 2-tuples (e.g. ``high`` / ``low`` bars) into two aligned columns.
    """
    return [row[0] for row in rows], [row[1] for row in rows]


def split_triples[T](rows: Sequence[tuple[T, T, T]]) -> tuple[list[T], list[T], list[T]]:
    """
    Unzip a list of 3-tuples (e.g. ``high`` / ``low`` / ``close`` bars) into three aligned columns.
    """
    return [row[0] for row in rows], [row[1] for row in rows], [row[2] for row in rows]


def split_quads[T](rows: Sequence[tuple[T, T, T, T]]) -> tuple[list[T], list[T], list[T], list[T]]:
    """
    Unzip a list of 4-tuples (OHLC or HLCV bars) into four aligned columns.
    """
    return (
        [row[0] for row in rows],
        [row[1] for row in rows],
        [row[2] for row in rows],
        [row[3] for row in rows],
    )
