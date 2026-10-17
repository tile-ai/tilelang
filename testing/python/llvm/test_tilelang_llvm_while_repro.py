import tilelang
import tilelang.testing
import tilelang.language as T


@tilelang.testing.requires_llvm
def test_llvm_kernel_source_generation_with_while() -> None:
    """Regression test: LLVM `T.While` kernels should compile without errors.

    See: https://github.com/tile-ai/tilelang/issues/2202

    Historically, a CPU kernel containing `T.While(...)` inside
    `T.Kernel(...)` could leak the synthetic fallback thread variable
    `v_thread` into host/device splitting. LLVM uses the same scalar lowering
    path for this case, so this keeps equivalent coverage for target="llvm".
    """

    @T.prim_func
    def main(flag: T.Tensor((1,), "int32"), out: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            state = T.alloc_fragment((1,), "int32")
            state[0] = 0

            with T.While(state[0] == 0):
                state[0] = flag[0]

            out[0] = state[0]

    compiled = tilelang.compile(main, target="llvm", execution_backend="tvm_ffi")
    assert compiled is not None
