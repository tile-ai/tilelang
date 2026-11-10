import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.host_mapped import get_host_mapped_params
from tilelang.jit.exceptions import JITNoBuilderError


def test_annotation_is_function_metadata_and_round_trips():
    @T.prim_func
    def kernel(src: T.Tensor((16,), T.float32), dst: T.Tensor((16,), T.float32)):
        T.annotate_host_mapped_buffers(dst)
        with T.Kernel(1, threads=32):
            T.annotate_host_mapped_buffers(src, dst)
            for i in T.Parallel(16):
                dst[i] = src[i]

    assert get_host_mapped_params(kernel) == (0, 1)
    assert all(buf.scope() == "global" for buf in kernel.buffer_map.values())
    restored = tvm.ir.load_json(tvm.ir.save_json(kernel))
    assert get_host_mapped_params(restored) == (0, 1)
    tvm.ir.assert_structural_equal(kernel, restored)


def test_annotation_requires_buffer():
    with pytest.raises(TypeError, match="expects Buffer"):

        @T.prim_func
        def kernel(dst: T.Tensor((16,), T.float32)):
            T.annotate_host_mapped_buffers(1)
            T.evaluate(0)


@pytest.mark.parametrize("scope", ["shared", "global"])
def test_annotation_rejects_internal_allocation(scope):
    with pytest.raises(ValueError, match="global buffers|function parameter"):

        @T.prim_func
        def kernel(dst: T.Tensor((16,), T.float32)):
            with T.Kernel(1, threads=32):
                tmp = T.alloc_global((16,), T.float32, scope=scope)
                T.annotate_host_mapped_buffers(tmp)
                T.evaluate(0)


@pytest.mark.parametrize("backend", ["nvrtc", "cutedsl"])
def test_unsupported_launcher_rejects_annotation(backend):
    from tilelang.engine.host_mapped import validate_host_mapped_params

    @T.prim_func
    def kernel(dst: T.Tensor((16,), T.float32)):
        T.annotate_host_mapped_buffers(dst)
        T.evaluate(0)

    with pytest.raises(ValueError, match="currently require"):
        validate_host_mapped_params(kernel, tvm.target.Target("cuda"), backend)


def test_host_mapped_output_must_be_explicit():
    @T.prim_func
    def kernel(dst: T.Tensor((16,), T.float32)):
        T.annotate_host_mapped_buffers(dst)
        T.evaluate(0)

    with pytest.raises(ValueError, match="preallocated"):
        tilelang.compile(kernel, target="cuda", execution_backend="cython", out_idx=[0])


def test_annotation_requires_builder():
    with pytest.raises(JITNoBuilderError):
        T.annotate_host_mapped_buffers()
