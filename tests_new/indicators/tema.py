"""Spec for ``pomata.indicators.tema`` — the triple EMA lag-correction, gap-bridging, NaN-latching, degree-1."""

from tests_new.indicators.oracles import tema_reference
from tests_new.support.spec import ScaleAxis, Shape, Spec, SpecPin

from pomata.indicators import tema

TEMA = Spec(
    factory=tema,
    inputs=("expr",),
    params={"window": 2},
    shape=Shape.SERIES,
    warmup=3,
    raises=(({"window": 0}, r"window must be >= 1"),),
    oracle=tema_reference,
    # A linear combination of three linear EMA passes scales linearly with the series (tests/indicators/test_tema.py
    # ::TestTemaProperties::test_scale_homogeneity).
    scale=(ScaleAxis(roles=("expr",), degree=1),),
    golden_input={"expr": (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)},
    golden_output=(None, None, None, 8.0, 10.0, 12.0),
    pins=(
        SpecPin(
            label="null_bridged",
            inputs={"expr": (1.0, None, 3.0, 4.0, 5.0)},
            expected=(None, None, None, None, 5.037037037037038),
            params_override={"window": 2},
            reason="the exact recovery value after an interior null bridges the cascade "
            "(test_tema.py::TestTemaEdge::test_null_bridged)",
        ),
        SpecPin(
            label="window_one_identity",
            inputs={"expr": (1.0, 2.0, 3.0, 4.0)},
            expected=(1.0, 2.0, 3.0, 4.0),
            params_override={"window": 1},
            reason="window=1 collapses each of the three nested EMAs to the identity "
            "(test_tema.py::TestTemaEdge::test_window_one_is_identity)",
        ),
        SpecPin(
            label="all_zero_series",
            inputs={"expr": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
            expected=(None, None, None, None, None, None, 0.0, 0.0),
            params_override={"window": 3},
            reason="the degenerate all-zero series stays exactly zero after warm-up "
            "(test_tema.py::TestTemaEdge::test_all_zero_series_is_zero)",
        ),
        SpecPin(
            label="constant_series",
            inputs={"expr": (7.0, 7.0, 7.0, 7.0, 7.0, 7.0)},
            expected=(None, None, None, 7.0, 7.0, 7.0),
            params_override={"window": 2},
            reason="TEMA of a constant recovers exactly that constant after warm-up "
            "(test_tema.py::TestTemaEdge::test_constant_series)",
        ),
        SpecPin(
            label="window_three_golden",
            inputs={"expr": (3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0)},
            expected=(
                None,
                None,
                None,
                None,
                None,
                None,
                3.296296296296297,
                5.399016203703705,
                5.081452546296297,
                3.234953703703704,
            ),
            params_override={"window": 3},
            reason="a second frozen golden master at window=3 (test_tema.py::TestTemaCorrectness"
            "::test_golden_master_window_three)",
        ),
        SpecPin(
            label="golden_adjusted",
            inputs={"expr": (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)},
            expected=(None, None, None, 8.118158284023668, 10.090675959328463, 12.055955303250178),
            params_override={"window": 2, "adjust": True},
            reason="the frozen golden under adjust=True finite-window unbiased weighting "
            "(test_tema.py::TestTemaCorrectness::test_golden_master_adjusted)",
        ),
    ),
)
