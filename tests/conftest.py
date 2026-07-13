"""
Pytest configuration for the declarative contract suite.

pytest imports this automatically for every ``tests`` session, so it is where the Hypothesis ``settings`` profiles
are registered and one is selected from the ``HYPOTHESIS_PROFILE`` environment variable, so the property tier draws
the pinned example count instead of Hypothesis' defaults. See ``tests/README.md`` for the ladder.
"""

import os

from hypothesis import settings

# Both profiles draw MAX_EXAMPLES examples per property test: once a property is proven and its edges are pinned by the
# deterministic tier, more random draws buy wall-clock, not confidence. A rare failing input — a subnormal magnitude, an
# ill-conditioned ratio — is driven toward zero by construction, not hunted with a larger N; CI's extra assurance comes
# from a fresh random seed each run (and the full OS x Python matrix). Both drop the per-example deadline, a poor
# flakiness signal under CI load (performance belongs to the benchmark tier).
MAX_EXAMPLES = 64  # the single knob; both profiles share it (raise or lower here, never per test)
settings.register_profile("dev", max_examples=MAX_EXAMPLES, deadline=None)
settings.register_profile("ci", max_examples=MAX_EXAMPLES, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
