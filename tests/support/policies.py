"""
The suite's view of the declared null / NaN policies — a re-export of the package's policy registry.

The declarations themselves live in :mod:`pomata._policy`, the single source of truth, so the package and the suite
can never disagree on a function's ``(null_policy, nan_policy)``. :mod:`tests.test_policies` proves each declaration
against the code on every run. See ``tests/README.md``.
"""

from pomata._policy import NO_ORACLE, POLICIES, NanPolicy, NullPolicy

__all__ = ("NO_ORACLE", "POLICIES", "NanPolicy", "NullPolicy")
