import tilelang
import tilelang.language as T


def test_cpu_kernel_source_generation_with_while() -> None:
    """Regression test: CPU `T.While` kernels should compile without errors.

    See: https://github.com/tile-ai/tilelang/issues/2202

    Historically, a CPU kernel containing `T.While(...)` inside
    `T.Kernel(..., is_cpu=True)` could leak the synthetic fallback thread
    variable `v_thread` into host/device splitting. That in turn caused CPU
    compilation to fail during packed-API generation or later C wrapper
    generation.

    This test follows the existing compile-only regression style used in the
    repository: success is defined by `tilelang.compile(..., target="c")`
    completing without raising an exception.
    """

    @T.prim_func
    def main(flag: T.Buffer((1,), "int32"), out: T.Buffer((1,), "int32")):
        with T.Kernel(1, is_cpu=True):
            state = T.alloc_fragment((1,), "int32")
            state[0] = 0

            with T.While(state[0] == 0):
                state[0] = flag[0]

            out[0] = state[0]

    compiled = tilelang.compile(main, target="c")
    assert compiled is not None
