"""Execution-region frames for Ascend NPU."""

from __future__ import annotations

from tilelang import _ffi_api
from tvm.ffi import register_object
from tvm.tirx.script.builder.frame import TIRFrame

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


@register_object("tl.CubeFrame")
class CubeFrame(TIRFrame):
    """Frame for Cube (AIC) execution region."""

    pass


def Cube() -> CubeFrame:
    """Construct a Cube (AIC) execution region frame."""
    return _ffi_api.Cube()  # type: ignore[attr-defined] # pylint: disable=no-member


@register_object("tl.VectorFrame")
class VectorFrame(TIRFrame):
    """Frame for Vector (AIV) execution region.

    In Mix mode, provides a subblock ID variable via ``as`` binding::

        with T.Vector(vector=2) as sid:
            # sid = get_subblockid()
            ...
    """

    def __enter__(self):
        super().__enter__()
        return self.sid_var


def Vector(vector: int = 2) -> VectorFrame:
    """Construct a Vector (AIV) execution region frame.

    Parameters
    ----------
    vector : int
        Number of AIV cores (1 or 2). Default 2.

    Returns
    -------
    res : VectorFrame
        The VectorFrame used to denote an AIV execution region.
    """
    return _ffi_api.Vector(vector)  # type: ignore[attr-defined] # pylint: disable=no-member


def _next_vf_source_index(kind: str) -> int:
    from tilelang.language.eager.builder import Builder

    builder = Builder.current()
    if builder is None:
        # A detached frame can be constructed for inspection without being
        # attached to IR. Normal lowering always constructs it under a Builder.
        return 0

    counter_attr = f"_ascend_{kind}_source_counter"
    source_index = getattr(builder, counter_attr, 0)
    setattr(builder, counter_attr, source_index + 1)
    return source_index


@register_object("tl.SimtVFFrame")
class SimtVFFrame(TIRFrame):
    """Frame for SimtVF (SIMT Vector Fixed) region.

    The C++ frame generates the ``SIMT_VF`` block and its thread-extent
    bindings. The Python frame additionally maintains the active SIMT context.
    """

    def __enter__(self):
        super().__enter__()
        from tilelang.language.kernel import SimtVFContext, push_simtvf_context

        ctx = SimtVFContext(
            thread_vars=list(self.thread_vars),
            thread_extents=[int(e) for e in self.thread_extents],
        )
        push_simtvf_context(ctx)
        return self

    def __exit__(self, ptype, value, trace):
        from tilelang.language.kernel import pop_simtvf_context

        pop_simtvf_context()
        super().__exit__(ptype, value, trace)


def SimtVF(threads: int | list[int] | tuple = 128, latency: int = 0) -> SimtVFFrame:
    """Construct a dedicated SIMT vector-fragment region frame.

    Parameters
    ----------
    threads : int | list[int] | tuple
        Thread count for the SimtVF frame. A sequence may contain up to three
        dimensions; omitted dimensions default to one.
    latency : int
        Measured latency in cycles. Zero uses the scheduler's latency model.

    Returns
    -------
    res : SimtVFFrame
        The SimtVFFrame used to denote a SIMT vector-fragment region.
    """
    if isinstance(threads, int):
        normalized = [threads, 1, 1]
    elif isinstance(threads, list):
        if len(threads) > 3:
            raise ValueError("SimtVF supports at most 3 thread dimensions")
        normalized = threads + [1] * (3 - len(threads))
    elif isinstance(threads, tuple):
        if len(threads) > 3:
            raise ValueError("SimtVF supports at most 3 thread dimensions")
        normalized = list(threads) + [1] * (3 - len(threads))
    else:
        raise TypeError(f"threads must be int, list, or tuple, got {type(threads)}")

    if any(t <= 0 for t in normalized):
        raise ValueError("SimtVF requires all thread dimensions > 0")

    source_index = _next_vf_source_index("simtvf")
    return _ffi_api.SimtVF(  # type: ignore[attr-defined] # pylint: disable=no-member
        normalized, int(latency), source_index
    )


@register_object("tl.SimdVFFrame")
class SimdVFFrame(TIRFrame):
    """Frame for SimdVF (SIMD Vector Fragment) region.

    This is a register-level SIMD region without thread dimensions. The C++
    frame generates the ``SIMD_VF`` block and its scope marker.
    """

    def __enter__(self):
        if _inside_simdvf[0] == 0:
            from .simd import _reset_default_mask_cache

            _reset_default_mask_cache()
        _inside_simdvf[0] += 1
        super().__enter__()
        return self

    def __exit__(self, ptype, value, trace):
        super().__exit__(ptype, value, trace)
        _inside_simdvf[0] -= 1
        if _inside_simdvf[0] == 0:
            from .simd import _reset_default_mask_cache

            _reset_default_mask_cache()


_inside_simdvf = [0]


def inside_simdvf() -> bool:
    """Return whether the current builder is inside a SimdVF region."""
    return _inside_simdvf[0] > 0


def SimdVF(latency: int = 0) -> SimdVFFrame:
    """Construct a SimdVF region for Ascend NPU MicroAPI operations.

    Parameters
    ----------
    latency : int
        Measured latency in cycles. Zero uses the scheduler's latency model.

    Returns
    -------
    res : SimdVFFrame
        The SimdVFFrame used to denote a register-level SIMD region.
    """
    source_index = _next_vf_source_index("simdvf")
    return _ffi_api.SimdVF(  # type: ignore[attr-defined] # pylint: disable=no-member
        int(latency), source_index
    )


def _register_eager_var_scope_frames() -> None:
    from tilelang.language.eager.builder import register_var_scope_frame

    for frame_type in (CubeFrame, VectorFrame, SimtVFFrame, SimdVFFrame):
        register_var_scope_frame(frame_type)


_register_eager_var_scope_frames()
del _register_eager_var_scope_frames
