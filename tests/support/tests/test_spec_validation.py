"""
Self-tests of the spec engine: every conditional requirement bites on a counterexample built inline.

The spec language's core guarantee is that a declaration cannot lie by omission. These tests construct deliberately
wrong specs and prove each is rejected at construction — a missing required field by the language itself (a native
``TypeError``), and every conditional rule by :meth:`Spec.__post_init__`. If any check ever regressed into a silent
no-op, one of these would go green where it must be red.
"""

import polars as pl
import pytest
from tests.support import Deviant, ScaleAxis, ScaleExempt, Shape, Spec, SpecPin

from pomata._policy import POLICIES, NanPolicy, NullPolicy
from pomata.indicators import sma


def _oracle(*_args: object, **_kwargs: object) -> object:
    """A stand-in oracle: ``__post_init__`` never calls it, it only checks the golden shapes around it."""
    return []


class TestNativeCompleteness:
    """The required fields have no default, so the language itself refuses an incomplete spec."""

    def test_missing_required_field_is_a_native_type_error(self) -> None:
        """Verifies omitting a required field dies at construction, naming the field."""
        with pytest.raises(TypeError, match=r"factory"):
            Spec(  # type: ignore[call-arg]
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
            )

    def test_scale_exempt_without_reason_is_a_native_type_error(self) -> None:
        """Verifies ``ScaleExempt`` cannot be built without its mandatory reason."""
        with pytest.raises(TypeError, match=r"reason"):
            ScaleExempt()  # type: ignore[call-arg]

    def test_deviant_without_reason_is_a_native_type_error(self) -> None:
        """Verifies ``Deviant`` cannot be built without its mandatory reason."""
        with pytest.raises(TypeError, match=r"reason"):
            Deviant(expected=(1.0,))  # type: ignore[call-arg]


class TestConditionalRequirements:
    """Every conditional rule is checked in ``__post_init__`` and proven on a counterexample."""

    def test_struct_without_fields_is_rejected(self) -> None:
        """Verifies a struct that names no fields dies at construction."""
        with pytest.raises(ValueError, match=r"a struct must declare its ordered fields"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.STRUCT,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output={"a": (1.0,)},
            )

    def test_reducing_with_a_warmup_is_rejected(self) -> None:
        """Verifies a reduction that declares a warm-up dies at construction."""
        with pytest.raises(ValueError, match=r"a reduction has no warm-up"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.REDUCING,
                warmup=2,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
            )

    def test_params_without_raises_is_rejected(self) -> None:
        """Verifies declared params without validation counterexamples die at construction (the rung would no-op)."""
        with pytest.raises(ValueError, match=r"declares params but no raises"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={"window": 3},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
            )

    def test_cost_degree_below_one_is_rejected(self) -> None:
        """Verifies a declared polynomial cost degree below one dies at construction."""
        with pytest.raises(ValueError, match=r"cost_degree must be >= 1"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
                cost_degree=0,
            )

    def test_empty_scale_tuple_is_rejected(self) -> None:
        """Verifies an empty scale tuple dies at construction — the exemption must be a reasoned ``ScaleExempt``."""
        with pytest.raises(ValueError, match=r"empty scale tuple is never allowed"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=(),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
            )

    def test_scale_axis_role_outside_inputs_is_rejected(self) -> None:
        """Verifies a scale axis naming a role that is not an input dies at construction."""
        with pytest.raises(ValueError, match=r"a scale axis names input roles"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=(ScaleAxis(roles=("mystery",), degree=1),),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
            )

    def test_unknown_input_role_is_rejected(self) -> None:
        """Verifies an input the probe frame cannot build dies at construction."""
        with pytest.raises(ValueError, match=r"the probe frame can build; unknown: \['mystery'\]"):
            Spec(
                factory=sma,
                inputs=("mystery",),
                params={},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"mystery": (1.0,)},
                golden_output=(1.0,),
            )

    def test_name_outside_the_policy_registry_is_rejected(self) -> None:
        """Verifies a factory whose name has no declared policy dies at construction (the name is derived)."""

        def unregistered(expr: pl.Expr) -> pl.Expr:
            return expr

        with pytest.raises(ValueError, match=r"no declared policy in pomata._policy"):
            Spec(
                factory=unregistered,
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
            )

    def test_fields_on_a_non_struct_are_rejected(self) -> None:
        """Verifies a non-struct that declares fields dies at construction."""
        with pytest.raises(ValueError, match=r"only a struct declares fields"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
                fields=("a",),
            )

    def test_bare_int_warmup_on_a_struct_is_rejected(self) -> None:
        """Verifies a struct declaring one bare warm-up int dies at construction — the form is per field."""
        with pytest.raises(ValueError, match=r"a struct declares its warm-up per field"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.STRUCT,
                warmup=2,
                fields=("a", "b"),
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output={"a": (1.0,), "b": (1.0,)},
            )

    def test_warmup_mapping_on_a_non_struct_is_rejected(self) -> None:
        """Verifies a per-field warm-up mapping on a series dies at construction — only a struct's form."""
        with pytest.raises(ValueError, match=r"keyed by a struct's fields, and only a struct's"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                warmup={"a": 2},
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
            )

    def test_warmup_mapping_with_wrong_keys_is_rejected(self) -> None:
        """Verifies a struct warm-up mapping keyed off the declared fields dies at construction."""
        with pytest.raises(ValueError, match=r"keyed by a struct's fields, and only a struct's"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.STRUCT,
                warmup={"a": 2, "wrong": 3},
                fields=("a", "b"),
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output={"a": (1.0,), "b": (1.0,)},
            )

    def test_bare_int_degree_on_a_struct_axis_is_rejected(self) -> None:
        """Verifies a struct scale axis with one bare degree dies at construction — the form is per field."""
        with pytest.raises(ValueError, match=r"a struct's scale axis declares one degree per field"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.STRUCT,
                fields=("a", "b"),
                scale=(ScaleAxis(roles=("expr",), degree=1),),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output={"a": (1.0,), "b": (1.0,)},
            )

    def test_degree_mapping_on_a_non_struct_axis_is_rejected(self) -> None:
        """Verifies a per-field degree mapping on a single-lane output dies at construction."""
        with pytest.raises(ValueError, match=r"only a struct's scale axis maps degrees per field"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=(ScaleAxis(roles=("expr",), degree={"a": 1}),),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
            )

    def test_golden_input_keys_off_the_inputs_are_rejected(self) -> None:
        """Verifies a golden input keyed off the declared roles dies at construction."""
        with pytest.raises(ValueError, match=r"golden_input keys .* must match inputs"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"close": (1.0,)},
                golden_output=(1.0,),
            )

    def test_mapping_golden_output_on_a_non_struct_is_rejected(self) -> None:
        """Verifies a per-field golden output on a series dies at construction."""
        with pytest.raises(ValueError, match=r"golden_output is a per-field mapping iff the shape is a struct"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output={"a": (1.0,)},
            )

    def test_flat_golden_output_on_a_struct_is_rejected(self) -> None:
        """Verifies a flat golden output on a struct dies at construction."""
        with pytest.raises(ValueError, match=r"golden_output is a per-field mapping iff the shape is a struct"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.STRUCT,
                fields=("a", "b"),
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
            )

    def test_duplicate_pin_labels_are_rejected(self) -> None:
        """Verifies two pins sharing a label die at construction — the label is the pytest id suffix."""
        pin = SpecPin(label="twin", inputs={"expr": (1.0,)}, expected=(1.0,), reason="x")
        with pytest.raises(ValueError, match=r"pin labels must be unique"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
                pins=(pin, pin),
            )

    def test_pin_inputs_off_the_roles_are_rejected(self) -> None:
        """Verifies a pin whose input lanes are keyed off the declared roles dies at construction."""
        with pytest.raises(ValueError, match=r"inputs \['close'\] must match \['expr'\]"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
                pins=(SpecPin(label="p", inputs={"close": (1.0,)}, expected=(1.0,), reason="x"),),
            )

    def test_pin_expected_shape_off_the_spec_shape_is_rejected(self) -> None:
        """Verifies a pin whose expected lanes are per-field on a series dies at construction."""
        with pytest.raises(ValueError, match=r"expected is a per-field mapping iff the shape is a struct"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
                pins=(SpecPin(label="p", inputs={"expr": (1.0,)}, expected={"a": (1.0,)}, reason="x"),),
            )

    def test_conditioning_without_a_covering_pin_is_rejected(self) -> None:
        """Verifies a conditioning filter without its witnessing pin dies at construction."""
        with pytest.raises(ValueError, match=r"no exclusion without a fixed case"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
                conditioning=lambda _frame: True,
            )

    def test_covering_pin_without_a_conditioning_filter_is_rejected(self) -> None:
        """Verifies a covers_conditioning pin on a spec with no filter dies at construction — a stale coverage claim."""
        with pytest.raises(ValueError, match=r"claim to cover a conditioning filter, but none is declared"):
            Spec(
                factory=sma,
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
                pins=(
                    SpecPin(label="p", inputs={"expr": (1.0,)}, expected=(1.0,), reason="x", covers_conditioning=True),
                ),
            )

    def test_empty_inputs_are_rejected(self) -> None:
        """Verifies an empty input tuple dies at construction — the probe frame needs at least one role."""
        with pytest.raises(ValueError, match=r"inputs must be non-empty roles"):
            Spec(
                factory=sma,
                inputs=(),
                params={},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={},
                golden_output=(1.0,),
            )

    def test_name_outside_every_public_all_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies a policy-registered name absent from every public ``__all__`` dies at construction."""

        def orphaned(expr: pl.Expr) -> pl.Expr:
            return expr

        monkeypatch.setitem(POLICIES, "orphaned", (NullPolicy.SKIPPED, NanPolicy.POISONS))
        with pytest.raises(ValueError, match=r"the derived name is in no public __all__"):
            Spec(
                factory=orphaned,
                inputs=("expr",),
                params={},
                shape=Shape.SERIES,
                scale=ScaleExempt(reason="x"),
                oracle=_oracle,
                golden_input={"expr": (1.0,)},
                golden_output=(1.0,),
            )
