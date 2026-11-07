"""Eager get_type_hints must resolve string annotations across Python versions."""

from __future__ import annotations

import inspect
from typing import ForwardRef, _eval_type

import tilelang.testing
from tilelang.language.eager.builder import _EVAL_TYPE_HAS_TYPE_PARAMS, get_type_hints


def test_eval_type_type_params_detection_matches_typing():
    assert ("type_params" in inspect.signature(_eval_type).parameters) == _EVAL_TYPE_HAS_TYPE_PARAMS


def test_get_type_hints_resolves_simple_string_annotation():
    def kernel(x: "int"):
        pass

    assert get_type_hints(kernel)["x"] is int


def test_get_type_hints_resolves_forward_ref_string():
    def kernel(xs: "list[int]"):
        pass

    hints = get_type_hints(kernel)
    assert hints["xs"] == list[int]


def test_get_type_hints_passes_empty_type_params_when_required():
    # Isolate the typing call that Python 3.15 requires, without depending on
    # TileLang's dtype string special-case path.
    value = ForwardRef("int", module=__name__)
    if _EVAL_TYPE_HAS_TYPE_PARAMS:
        resolved = _eval_type(value, globalns=globals(), localns={}, type_params=())
    else:
        resolved = _eval_type(value, globalns=globals(), localns={})
    assert resolved is int


if __name__ == "__main__":
    tilelang.testing.main()
