# type: ignore

import tilelang
import tilelang.testing
import tilelang.language as T


def debug_print_buffer(M=16, N=16, dtype=T.float16):
    @T.prim_func
    def program(Q: T.Tensor((M, N), dtype)):
        with T.Kernel(4, 4, 2, threads=128 * 2) as (bx, by, bz):
            shared_buf = T.alloc_shared([M, N], dtype)
            T.print(shared_buf)

    jit_kernel = tilelang.compile(program)
    profiler = jit_kernel.get_profiler()
    profiler.run_once()


def test_debug_print_buffer():
    debug_print_buffer(dtype=T.int8)
    debug_print_buffer(dtype=T.int16)
    debug_print_buffer(dtype=T.int32)
    debug_print_buffer(dtype=T.int64)
    debug_print_buffer(dtype=T.uint8)
    debug_print_buffer(dtype=T.uint16)
    debug_print_buffer(dtype=T.uint32)
    debug_print_buffer(dtype=T.uint64)
    debug_print_buffer(dtype=T.float16)
    debug_print_buffer(dtype=T.float32)
    debug_print_buffer(dtype=T.float64)
    debug_print_buffer(dtype=T.bfloat16)


@tilelang.testing.requires_cuda
def test_debug_print_buffer_cuda_fp8():
    debug_print_buffer(dtype=T.float8_e4m3fn)
    debug_print_buffer(dtype=T.float8_e5m2)


@tilelang.testing.requires_rocm
def test_debug_print_buffer_rocm_fp8():
    debug_print_buffer(dtype=T.float8_e4m3fnuz)
    debug_print_buffer(dtype=T.float8_e5m2fnuz)


def debug_print_buffer_conditional(M=16, N=16):
    dtype = T.float16

    @T.prim_func
    def program(Q: T.Tensor((M, N), dtype)):
        with T.Kernel(4, 4, 2, threads=128 * 2) as (bx, by, bz):
            shared_buf = T.alloc_shared([M, N], dtype)

            if bx == 0 and by == 0 and bz == 0:
                T.print(shared_buf)

    jit_kernel = tilelang.compile(program, target="cuda")
    profiler = jit_kernel.get_profiler()
    profiler.run_once()


def test_debug_print_buffer_conditional():
    debug_print_buffer_conditional(16, 16)


def debug_print_value_conditional(M=16, N=16):
    dtype = T.float16

    @T.prim_func
    def program(Q: T.Tensor((M, N), dtype)):
        with T.Kernel(4, 4, 2, threads=128 * 2) as (bx, by, bz):
            tid = T.get_thread_binding()
            if tid == 0:
                T.print(bx + by + bz)

    jit_kernel = tilelang.compile(program, target="cuda")
    profiler = jit_kernel.get_profiler()
    profiler.run_once()


def test_debug_print_value_conditional():
    debug_print_value_conditional(16, 16)


def debug_print_register_files(M=16, N=16):
    dtype = T.float16

    @T.prim_func
    def program(Q: T.Tensor((M, N), dtype)):
        with T.Kernel(4, 4, 2, threads=128 * 2) as (bx, by, bz):
            register_buf = T.alloc_fragment([M, N], dtype)
            for i, j in T.Parallel(M, N):
                T.print(register_buf[i, j])

    jit_kernel = tilelang.compile(program, target="cuda")
    profiler = jit_kernel.get_profiler()
    profiler.run_once()


def test_debug_print_register_files():
    debug_print_register_files(16, 16)


def debug_print_msg(M=16, N=16):
    dtype = T.float16

    @T.prim_func
    def program(Q: T.Tensor((M, N), dtype)):
        with T.Kernel(4, 4, 2, threads=128 * 2) as (bx, by, bz):
            tid = T.get_thread_binding()
            if tid == 0:
                T.print(bx + by + bz, msg="hello world")

    jit_kernel = tilelang.compile(program, target="cuda")
    profiler = jit_kernel.get_profiler()
    profiler.run_once()


def test_debug_print_msg():
    debug_print_msg(16, 16)


if __name__ == "__main__":
    tilelang.testing.main()
