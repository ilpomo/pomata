"""
The pnl family: one declaration per public ``pomata.pnl`` function, plus the family dialect that shapes them.

Each declaration file states one function's contract by calling :func:`tests.pnl.harness.suite_pnl` with the
family enums (:mod:`tests.pnl.enums`) and a naive reference oracle (:mod:`tests.pnl.oracles`). The call
registers the declaration as a side effect, so importing :mod:`tests.pnl.all_pnl` populates the pnl registry the
generic rungs then parametrize over. This package marker carries no re-export surface of its own.
"""
