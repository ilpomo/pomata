"""
The contract framework's public surface — the machinery and the capability mixins a contract composes.

Shared low-level helpers (asserts, tolerances, strategies, frame utilities) stay in :mod:`tests.support` and are
imported from there: this package adds only what the declarative redesign introduces. See ``tests_new/DESIGN.md``.
"""

from tests_new.support.contracts import (
    REGISTRY,
    Contract,
    ContractCorrectness,
    ContractProperties,
    ContractReducing,
    ContractSeries,
    ContractStruct,
    ContractWindowed,
    probe_frame,
)

__all__ = (
    "REGISTRY",
    "Contract",
    "ContractCorrectness",
    "ContractProperties",
    "ContractReducing",
    "ContractSeries",
    "ContractStruct",
    "ContractWindowed",
    "probe_frame",
)
