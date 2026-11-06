"""Ascend execution scopes built from native TIRx regions.

Allocations are lexical Bind statements inside the region that owns them. They
never need the old SBlock allocation hoisting/stealing protocol.
"""

from __future__ import annotations

from tilelang.language import _source_scope
from tilelang.language import frame as _loop_frame

from contextvars import ContextVar


from tvm_tirx.tirx.script import ir_builder

__all__ = [
    "Cube",
    "CubeFrame",
    "SimdVF",
    "SimdVFFrame",
    "SimtVF",
    "SimtVFFrame",
    "Vector",
    "VectorFrame",
    "inside_simdvf",
]


def _block(name, annotations=None):
    return _source_scope._ascend_block(name, annotations)


class CubeFrame(_loop_frame.LoopFrame):
    """Lexical AIC scope, represented by an ordinary native region."""

    def __init__(self):
        super().__init__([_block("CUBE")])

    def __enter__(self):
        super().__enter__()
        return None


def Cube() -> CubeFrame:
    """Construct a Cube (AIC) execution region frame."""
    return CubeFrame()


class VectorFrame(_loop_frame.LoopFrame):
    """Lexical AIV scope with a native cthread body parameter."""

    def __init__(self, vector_count):
        if vector_count not in (1, 2):
            raise ValueError(f"Vector core count must be 1 or 2, got {vector_count}")
        self.vector_count = int(vector_count)
        thread = ir_builder.launch_thread("cthread", self.vector_count)
        self.sid_var = thread.body_params[0]
        super().__init__([_block("VECTOR", {"vector_count": self.vector_count}), thread])

    def __enter__(self):
        super().__enter__()
        return self.sid_var


def Vector(vector: int = 2) -> VectorFrame:
    """Construct an AIV execution scope and expose its sub-core index."""
    return VectorFrame(vector)


def _next_vf_source_index(kind: str) -> int:
    from tilelang.language.eager.builder import Builder

    builder = Builder.current()
    if builder is None:
        return 0
    counter_attr = f"_ascend_{kind}_source_counter"
    source_index = getattr(builder, counter_attr, 0)
    setattr(builder, counter_attr, source_index + 1)
    return source_index


def _vf_annotations(kind, latency):
    annotations = {"tl.vf_source_index": _next_vf_source_index(kind)}
    if latency > 0:
        annotations["tl.vf_latency"] = int(latency)
    return annotations


class SimtVFFrame(_loop_frame.LoopFrame):
    """A SIMT helper region containing three native launch_thread scopes."""

    def __init__(self, thread_extents, latency=0):
        self.thread_extents = list(thread_extents)
        threads = [ir_builder.launch_thread("threadIdx." + axis, extent) for axis, extent in zip("xyz", self.thread_extents)]
        self.thread_vars = [thread.body_params[0] for thread in threads]
        super().__init__(
            [
                _block("SIMT_VF", _vf_annotations("simtvf", latency)),
                *threads,
            ]
        )

    def __enter__(self):
        super().__enter__()
        from .kernel import SimtVFContext, push_simtvf_context

        push_simtvf_context(SimtVFContext(self.thread_vars, self.thread_extents))
        return self

    def __exit__(self, ptype, value, trace):
        from .kernel import pop_simtvf_context

        try:
            return super().__exit__(ptype, value, trace)
        finally:
            pop_simtvf_context()


def SimtVF(threads: int | list[int] | tuple = 128, latency: int = 0) -> SimtVFFrame:
    """Construct a SIMT vector-fragment scope with one to three thread axes."""
    if isinstance(threads, int):
        normalized = [threads, 1, 1]
    elif isinstance(threads, (list, tuple)):
        if len(threads) > 3:
            raise ValueError("SimtVF supports at most 3 thread dimensions")
        normalized = list(threads) + [1] * (3 - len(threads))
    else:
        raise TypeError(f"threads must be int, list, or tuple, got {type(threads)}")
    if any(t <= 0 for t in normalized):
        raise ValueError("SimtVF requires all thread dimensions > 0")
    return SimtVFFrame(normalized, int(latency))


_inside_simdvf = ContextVar("tilelang_inside_simdvf", default=0)


class SimdVFFrame(_loop_frame.LoopFrame):
    """Register-level SIMD helper scope, without any synthetic thread axes."""

    def __init__(self, latency=0):
        super().__init__(
            [
                _block("SIMD_VF", _vf_annotations("simdvf", latency)),
            ]
        )
        self._token = None

    def __enter__(self):
        from .simd import _reset_default_mask_cache

        if not inside_simdvf():
            _reset_default_mask_cache()
        super().__enter__()
        self._token = _inside_simdvf.set(_inside_simdvf.get() + 1)
        return self

    def __exit__(self, ptype, value, trace):
        from .simd import _reset_default_mask_cache

        try:
            return super().__exit__(ptype, value, trace)
        finally:
            _inside_simdvf.reset(self._token)
            if not inside_simdvf():
                _reset_default_mask_cache()


def inside_simdvf() -> bool:
    """Whether this builder context is inside a register-level SIMD scope."""
    return _inside_simdvf.get() > 0


def SimdVF(latency: int = 0) -> SimdVFFrame:
    """Construct an Ascend SIMD vector-fragment scope."""
    return SimdVFFrame(int(latency))
