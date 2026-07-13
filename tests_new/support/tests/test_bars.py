"""
Meta-tests for ``tests_new.support.bars`` — the bar-unzip splits.

These verify the test infrastructure itself: a wrong ``split_*`` would silently mis-feed every multi-input property
tier, so the splits are pinned the same way an indicator is.
"""

from tests_new.support import split_pairs, split_quads, split_triples


class TestSplits:
    """
    The bar-unzip helpers transpose a list of tuples into aligned per-column lists.
    """

    def test_split_pairs(self) -> None:
        """
        Verifies that ``split_pairs`` transposes 2-tuples into two aligned columns.
        """
        assert split_pairs([(1.0, 2.0), (3.0, 4.0)]) == ([1.0, 3.0], [2.0, 4.0])

    def test_split_triples(self) -> None:
        """
        Verifies that ``split_triples`` transposes 3-tuples into three aligned columns.
        """
        assert split_triples([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]) == ([1.0, 4.0], [2.0, 5.0], [3.0, 6.0])

    def test_split_quads(self) -> None:
        """
        Verifies that ``split_quads`` transposes 4-tuples into four aligned columns.
        """
        rows = [(1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)]
        assert split_quads(rows) == ([1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0])

    def test_split_empty(self) -> None:
        """
        Verifies that an empty input yields the right number of empty columns, not a single empty list.
        """
        assert split_pairs([]) == ([], [])
        assert split_triples([]) == ([], [], [])
        assert split_quads([]) == ([], [], [], [])
