from types import SimpleNamespace

from tilelang import tvm as tvm
import tilelang.ascend.language as T

from tilelang.jit.adapter.cython.adapter import CythonKernelAdapter


CUDA_TARGET = SimpleNamespace(kind=SimpleNamespace(name="cuda"), keys=("cuda",))


def test_cython_adapter_leaves_non_packed_int4_abi_unchanged():
    @T.prim_func
    def main(A: T.StridedTensor[(4, 256), (512, 1), T.int4]):
        T.evaluate(0)

    adapter = CythonKernelAdapter.__new__(CythonKernelAdapter)
    adapter.ir_module = tvm.IRModule({main.attrs["global_symbol"]: main})
    adapter.target = CUDA_TARGET
    static_shapes, static_strides, _ = adapter._process_static_buffer_infos()

    assert list(static_shapes.values())[0][1] == [(0, 4), (1, 256)]
    assert list(static_strides.values())[0][1] == [(0, 512), (1, 1)]
