"""
The spec framework's public surface — the frozen data types a contract declares and the engine the rungs delegate to.

Shared low-level helpers (asserts, tolerances, strategies, frame utilities) stay in :mod:`tests.support` and are
imported from there: this package adds only what the declarative redesign introduces. See ``tests_new/DESIGN.md``.
"""

from tests_new.support.spec import (
    SPEC_LANE,
    SPEC_SCALAR,
    Deviant,
    ScaleAxis,
    ScaleExempt,
    Shape,
    Spec,
    SpecPin,
    actual_lanes,
    build_expr,
    flat,
    fuzz_frames,
    horizon,
    lane_series,
    probe_frame,
    probe_length,
    reference_lanes,
    spec_id,
    widest_warmup,
    widest_window,
)

__all__ = (
    "SPEC_LANE",
    "SPEC_SCALAR",
    "Deviant",
    "ScaleAxis",
    "ScaleExempt",
    "Shape",
    "Spec",
    "SpecPin",
    "actual_lanes",
    "build_expr",
    "flat",
    "fuzz_frames",
    "horizon",
    "lane_series",
    "probe_frame",
    "probe_length",
    "reference_lanes",
    "spec_id",
    "widest_warmup",
    "widest_window",
)
