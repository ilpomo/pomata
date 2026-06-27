"""
Pytest configuration and shared collection root for the pomata test suite.

This module is intentionally minimal: pytest imports it automatically for every test session, so it is the natural home
for genuine, stateful fixtures should the suite ever need them. The stateless cross-module helpers (``apply_expr`` and
``assert_matches``) deliberately live in the ``tests.support`` package and are imported directly as
``from tests.support import apply_expr, assert_matches`` rather than exposed as fixtures, because a function-scoped
fixture is created once per test function (not once per Hypothesis-generated example) and would leak state across the
examples of a property-based test; plain functions sidestep that and read identically in every tier of the ladder.

See ``tests/README.md`` for the category/marker structure (the per-module test classes and the cross-cutting
``differential`` / ``benchmark`` markers) and ``tests/indicators/oracles`` for the naive reference implementations and
the frozen golden-master datasets that anchor the correctness tier.

It also registers the Hypothesis ``settings`` profiles and selects one from the ``HYPOTHESIS_PROFILE`` environment
variable. Both profiles draw the same derived example count; CI's extra assurance comes from a fresh random seed each
run (and the full OS x Python matrix), not from a larger search.
"""

import os

from hypothesis import settings

# Both profiles draw MAX_EXAMPLES examples per property test: once a property is proven and its edges are pinned by the
# deterministic tier, more random draws buy wall-clock, not confidence (the reasoning is in CORRECTNESS.md). A rare
# failing input — a subnormal magnitude, an ill-conditioned ratio — is driven toward zero by construction, not hunted
# with a larger N; CI's extra assurance comes from a fresh random seed each run (and the full OS x Python matrix). Both
# drop the per-example deadline, a poor flakiness signal under CI load (performance
# belongs to the benchmark tier).
MAX_EXAMPLES = 64  # the single knob; both profiles share it (raise or lower here, never per test)
settings.register_profile("dev", max_examples=MAX_EXAMPLES, deadline=None)
settings.register_profile("ci", max_examples=MAX_EXAMPLES, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
