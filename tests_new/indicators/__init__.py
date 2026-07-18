"""
The indicators family: one declaration per public ``pomata.indicators`` function, plus the family dialect that shapes
them.

Each declaration file states one function's contract by calling :func:`tests_new.indicators.harness.suite_indicators`
with the family enums (:mod:`tests_new.indicators.enums`) and a naive reference oracle
(:mod:`tests_new.indicators.oracles`). The call registers the declaration as a side effect, so importing
:mod:`tests_new.indicators.all_indicators` populates the indicators registry the generic rungs then parametrize over.
This package marker carries no re-export surface of its own.
"""
