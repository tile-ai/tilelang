"""Historical decorator spelling over TileLang's native F1 frontend."""

from __future__ import annotations

from tvm_tirx.tirx.script import macro


def prim_func(func=None, private=False, check_well_formed=False):
    """Build a native Function while retaining the traditional decorator call."""

    def decorate(function):
        from tilelang.language.eager.builder import prim_func as build
        from tvm_tirx.tirx.analysis import check_well_formed as verify

        result = build(function)
        if private:
            result = result.without_attr("global_symbol")
        if check_well_formed:
            verify(result)
        return result

    return decorate if func is None else decorate(func)


__all__ = ["prim_func", "macro"]
