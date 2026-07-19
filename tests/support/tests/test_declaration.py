"""
Self-tests of the declaration constructor: every conditional requirement bites on a counterexample.

The declaration language's core guarantee is that a contract cannot lie by omission. These tests take one valid
template and, with :func:`dataclasses.replace`, change exactly one field to something wrong, proving each is rejected
by :meth:`Declaration.__post_init__`. If any check ever regressed into a silent no-op, one of these would go green
where it must be red.
"""

import dataclasses
import enum

import polars as pl
import pytest

from tests.support.declaration import Declaration, Deviant, Golden, Pin, ScaleAxis, Shape


class _Behavior(enum.Enum):
    PROPAGATES = "propagates"
    BRIDGED = "bridged"


class _Space(enum.Enum):
    CASH = "cash"


class _Sign(enum.Enum):
    LONG_SHORT = "long_short"


class _NonFinite(enum.Enum):
    IEEE_FLOW = "ieee_flow"


def widget(price: pl.Expr) -> pl.Expr:
    """A stand-in factory: ``__post_init__`` never calls it, it only reads its ``__name__``."""
    return price


def reference_widget(price: list[float | None]) -> list[float | None]:
    """The stand-in oracle, named ``reference_widget`` so the oracle-name guard is satisfied."""
    return list(price)


def wrong_name(price: list[float | None]) -> list[float | None]:
    """A stand-in oracle with the wrong name, to trip the oracle-name guard."""
    return list(price)


def _always(_frame: pl.DataFrame) -> bool:
    """A conditioning predicate that admits everything — enough to test the pairing guard."""
    return True


_VALID = Declaration(
    family="pnl",
    factory=widget,
    inputs=("price",),
    params={},
    shape=Shape.SERIES,
    behavior_null=_Behavior.PROPAGATES,
    behavior_nan=_Behavior.PROPAGATES,
    space=_Space.CASH,
    sign=_Sign.LONG_SHORT,
    nonfinite=_NonFinite.IEEE_FLOW,
    oracle=reference_widget,
    scaling=(ScaleAxis(roles=("price",), degree=0),),
)


def test_valid_declaration_constructs() -> None:
    """The template is valid and derives its name from the factory."""
    assert _VALID.name == "widget"
    assert _VALID.landing == "price"


class TestInputAndOracle:
    """The input-role and oracle-name guards."""

    def test_empty_inputs_rejected(self) -> None:
        """An empty input tuple dies at construction — the probe frame needs at least one role."""
        with pytest.raises(ValueError, match=r"inputs must be non-empty roles"):
            dataclasses.replace(_VALID, inputs=())

    def test_unknown_role_rejected(self) -> None:
        """An input the probe frame cannot build dies at construction."""
        with pytest.raises(ValueError, match=r"unknown: \['mystery'\]"):
            dataclasses.replace(_VALID, inputs=("mystery",), scaling=(ScaleAxis(roles=("mystery",), degree=0),))

    def test_wrong_oracle_name_rejected(self) -> None:
        """An oracle not named ``reference_{name}`` dies at construction."""
        with pytest.raises(ValueError, match=r"the oracle must be named 'reference_widget'"):
            dataclasses.replace(_VALID, oracle=wrong_name)


class TestFieldsAndWarmup:
    """The struct-fields and warm-up-form guards."""

    def test_struct_without_fields_rejected(self) -> None:
        """A struct that names no fields dies at construction."""
        with pytest.raises(ValueError, match=r"a struct must declare its ordered fields"):
            dataclasses.replace(_VALID, shape=Shape.STRUCT)

    def test_fields_on_non_struct_rejected(self) -> None:
        """A non-struct that declares fields dies at construction."""
        with pytest.raises(ValueError, match=r"only a struct declares fields"):
            dataclasses.replace(_VALID, fields=("a",))

    def test_warmup_mapping_on_non_struct_rejected(self) -> None:
        """A per-field warm-up mapping on a series dies at construction — only a struct's form."""
        with pytest.raises(ValueError, match=r"a per-field warm-up mapping is keyed by a struct's fields"):
            dataclasses.replace(_VALID, warmup={"a": 2})

    def test_bare_int_warmup_on_struct_rejected(self) -> None:
        """A struct declaring one bare warm-up int dies at construction — the form is per field."""
        with pytest.raises(ValueError, match=r"a struct declares its warm-up per field"):
            dataclasses.replace(
                _VALID,
                shape=Shape.STRUCT,
                fields=("a", "b"),
                warmup=2,
                scaling=(ScaleAxis(roles=("price",), degree={"a": 1, "b": 1}),),
            )


class TestScaling:
    """The scale-claim guards."""

    def test_empty_scale_tuple_rejected(self) -> None:
        """An empty scale tuple dies at construction — the exemption must be a reasoned ``ScaleExempt``."""
        with pytest.raises(ValueError, match=r"an empty scale tuple is never allowed"):
            dataclasses.replace(_VALID, scaling=())

    def test_scale_axis_role_outside_inputs_rejected(self) -> None:
        """A scale axis naming a role that is not an input dies at construction."""
        with pytest.raises(ValueError, match=r"a scale axis names input roles"):
            dataclasses.replace(_VALID, scaling=(ScaleAxis(roles=("mystery",), degree=0),))

    def test_bare_int_degree_on_struct_axis_rejected(self) -> None:
        """A struct scale axis with one bare degree dies at construction — the form is per field."""
        with pytest.raises(ValueError, match=r"a struct's scale axis declares one degree per field"):
            dataclasses.replace(
                _VALID, shape=Shape.STRUCT, fields=("a", "b"), scaling=(ScaleAxis(roles=("price",), degree=1),)
            )

    def test_degree_mapping_on_non_struct_axis_rejected(self) -> None:
        """A per-field degree mapping on a single-lane output dies at construction."""
        with pytest.raises(ValueError, match=r"only a struct's scale axis maps degrees per field"):
            dataclasses.replace(_VALID, scaling=(ScaleAxis(roles=("price",), degree={"a": 1}),))


class TestGoldenPinsDeviant:
    """The golden-shape, pin, and deviant guards."""

    def test_golden_inputs_off_the_roles_rejected(self) -> None:
        """A golden input keyed off the declared roles dies at construction."""
        with pytest.raises(ValueError, match=r"golden inputs .* must match inputs"):
            dataclasses.replace(_VALID, golden=Golden(inputs={"close": (1.0,)}, output=(1.0,)))

    def test_mapping_golden_output_on_non_struct_rejected(self) -> None:
        """A per-field golden output on a series dies at construction."""
        with pytest.raises(ValueError, match=r"a golden output is a per-field mapping iff"):
            dataclasses.replace(_VALID, golden=Golden(inputs={"price": (1.0,)}, output={"a": (1.0,)}))

    def test_duplicate_pin_labels_rejected(self) -> None:
        """Two pins sharing a label die at construction — the label is the pytest id suffix."""
        pin = Pin(label="twin", inputs={"price": (1.0,)}, expected=(1.0,), reason="x")
        with pytest.raises(ValueError, match=r"pin labels must be unique"):
            dataclasses.replace(_VALID, pins=(pin, pin))

    def test_pin_inputs_off_the_roles_rejected(self) -> None:
        """A pin whose input lanes are keyed off the declared roles dies at construction."""
        with pytest.raises(ValueError, match=r"inputs \['close'\] must match \['price'\]"):
            dataclasses.replace(_VALID, pins=(Pin(label="p", inputs={"close": (1.0,)}, expected=(1.0,), reason="x"),))

    def test_pin_expected_shape_off_the_shape_rejected(self) -> None:
        """A pin whose expected lanes are per-field on a series dies at construction."""
        with pytest.raises(ValueError, match=r"expected is a per-field mapping iff"):
            dataclasses.replace(
                _VALID, pins=(Pin(label="p", inputs={"price": (1.0,)}, expected={"a": (1.0,)}, reason="x"),)
            )

    def test_deviant_empty_reason_rejected(self) -> None:
        """A deviant with a blank reason dies at construction."""
        with pytest.raises(ValueError, match=r"a deviant must carry a non-empty reason"):
            dataclasses.replace(_VALID, deviant=Deviant(expected=(None,), reason="   "))


class TestConditioningPairing:
    """A conditioning filter and its witnessing pin must come as a pair."""

    def test_conditioning_without_a_covering_pin_rejected(self) -> None:
        """A conditioning filter without its witnessing pin dies at construction."""
        with pytest.raises(ValueError, match=r"no exclusion without a fixed case"):
            dataclasses.replace(_VALID, conditioning=_always)

    def test_covering_pin_without_a_conditioning_filter_rejected(self) -> None:
        """A covers_conditioning pin with no filter dies at construction — a stale coverage claim."""
        with pytest.raises(ValueError, match=r"claim to cover a conditioning filter, but none is declared"):
            dataclasses.replace(
                _VALID,
                pins=(Pin(label="p", inputs={"price": (1.0,)}, expected=(1.0,), reason="x", covers_conditioning=True),),
            )


class TestParamsRaises:
    """Declared params without validation counterexamples would make the raises rung a no-op."""

    def test_params_without_raises_rejected(self) -> None:
        """Declared params without validation counterexamples die at construction."""
        with pytest.raises(ValueError, match=r"declares params but no raises"):
            dataclasses.replace(_VALID, params={"window": 3})


class TestProse:
    """The documentation-prose guards: a cross-reference must resolve, a note subheader must be labeled."""

    def test_unknown_see_also_target_rejected(self) -> None:
        """A See Also naming a function that is in no family ``__all__`` dies at construction."""
        with pytest.raises(ValueError, match=r"see_also names 'no_such_function', which is not a public function"):
            dataclasses.replace(_VALID, see_also=(("no_such_function", "a clause"),))

    def test_empty_note_label_rejected(self) -> None:
        """A note subheader with a blank label dies at construction."""
        with pytest.raises(ValueError, match=r"a note subheader must carry a non-empty label"):
            dataclasses.replace(_VALID, notes=(("   ", "a body"),))

    def test_note_label_with_trailing_colon_rejected(self) -> None:
        """A note subheader label ending in a colon dies at construction (the bold header renders no punctuation)."""
        with pytest.raises(ValueError, match=r"note subheader 'Seeding:' must not end with a colon"):
            dataclasses.replace(_VALID, notes=(("Seeding:", "a body"),))

    def test_opener_override_with_note_extension_rejected(self) -> None:
        """An opener override alongside a note extension dies at construction (nothing is left to extend)."""
        with pytest.raises(ValueError, match=r"opener_override replaces the whole opener body"):
            dataclasses.replace(_VALID, opener_override="A bespoke opener.", note_extension="An extension.")

    def test_empty_bullet_label_rejected(self) -> None:
        """An edge-case bullet with a blank label dies at construction."""
        with pytest.raises(ValueError, match=r"an edge-case bullet must carry a non-empty label"):
            dataclasses.replace(_VALID, bullets=(("   ", "a body"),))

    def test_reference_url_that_is_a_doi_rejected(self) -> None:
        """A DOI or Wikipedia URL misfiled into the fourth-bucket ``reference_url`` dies at construction."""
        with pytest.raises(ValueError, match=r"reference_url is the non-DOI/non-Wikipedia bucket"):
            dataclasses.replace(_VALID, reference_url="https://doi.org/10.1000/xyz")

    def test_args_prose_off_the_signature_rejected(self) -> None:
        """An ``args_prose`` key that is not a factory parameter dies at construction (a stale parameter name)."""
        with pytest.raises(ValueError, match=r"args_prose describes \['no_such_param'\], which are not parameters"):
            dataclasses.replace(_VALID, args_prose={"no_such_param": "a description"})
