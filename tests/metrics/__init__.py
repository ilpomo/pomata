"""
The metrics family: the family dialect that shapes its declarations, plus (as the family lands) one declaration per
public ``pomata.metrics`` function.

The metrics dialect differs from the pnl one in two ways the engine reads directly: a reduction skips an interior
missing value rather than propagating it, and a trailing-window (rolling) function nulls every window the missing value
overlaps. Its rolling functions declare a ``rolling_of`` twin — the reducing or series function they roll per window —
so the twin-coherence rung holds each rolling row to the twin over its trailing window, and an ``Annualization``
convention that the closed-form annualization rung checks where a period-count ratio exists. This package marker carries
no re-export surface of its own.
"""
