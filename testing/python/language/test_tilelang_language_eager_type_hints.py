"""Eager get_type_hints must resolve string annotations across Python versions."""

from __future__ import annotations

import inspect
import sys
from typing import ForwardRef, TypeVar, _eval_type

import pytest

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


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12+")
def test_get_type_hints_resolves_function_type_parameter():
    namespace = {}
    exec("def kernel[T](x: 'T'):\n    pass\n", namespace)
    kernel = namespace["kernel"]
    (type_param,) = kernel.__type_params__

    assert get_type_hints(kernel)["x"] is type_param


def test_get_type_hints_forwards_required_type_params(monkeypatch):
    # Python 3.15 makes type_params a required _eval_type argument. Simulate
    # that signature so the forwarding logic is covered on every Python.
    import tilelang.language.eager.builder as builder

    forwarded = []

    def fake_eval_type(value, globalns, localns, *, type_params):
        forwarded.append(type_params)
        return int

    monkeypatch.setattr(builder, "_eval_type", fake_eval_type)
    monkeypatch.setattr(builder, "_EVAL_TYPE_HAS_TYPE_PARAMS", True)

    def kernel(x: "int"):
        pass

    type_param = TypeVar("T")
    kernel.__type_params__ = (type_param,)

    assert builder.get_type_hints(kernel)["x"] is int
    assert forwarded == [(type_param,)]


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
