# type: ignore
import pytest
import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang.language.fp8 import determine_fp8_type


def debug_print_buffer(M=16, N=16, dtype=T.float16):
    @T.prim_func
    def program(Q: T.Tensor((M, N), dtype)):
        with T.Kernel(4, 4, 2, threads=128 * 2) as (bx, by, bz):
            shared_buf = T.alloc_shared([M, N], dtype)
            T.print(shared_buf)

    jit_kernel = tilelang.compile(program)
    profiler = jit_kernel.get_profiler()
    profiler.run_once()


@pytest.mark.parametrize(
    "dtype", [T.int8, T.int16, T.int32, T.int64, T.uint8, T.uint16, T.uint32, T.uint64, T.float16, T.float32, T.float64, T.bfloat16]
)
def test_debug_print_buffer(dtype):
    debug_print_buffer(dtype=dtype)


@tilelang.testing.requires_cuda
def test_debug_print_buffer_cuda_fp8():
    debug_print_buffer(dtype=T.float8_e4m3fn)
    debug_print_buffer(dtype=T.float8_e5m2)


@tilelang.testing.requires_rocm
def test_debug_print_buffer_rocm_fp8():
    debug_print_buffer(dtype=determine_fp8_type("e4m3"))
    debug_print_buffer(dtype=determine_fp8_type("e5m2"))


def debug_print_buffer_conditional(M=16, N=16):
    dtype = T.float16

    @T.prim_func
    def program(Q: T.Tensor((M, N), dtype)):
        with T.Kernel(4, 4, 2, threads=128 * 2) as (bx, by, bz):
            shared_buf = T.alloc_shared([M, N], dtype)

            if bx == 0 and by == 0 and bz == 0:
                T.print(shared_buf)

    jit_kernel = tilelang.compile(program)
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

    jit_kernel = tilelang.compile(program)
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

    jit_kernel = tilelang.compile(program)
    profiler = jit_kernel.get_profiler()
    profiler.run_once()


def test_debug_print_register_files():
    debug_print_register_files(16, 16)


def debug_print_msg(M=16, N=16, msg_only=False):
    dtype = T.float16

    @T.prim_func
    def program(Q: T.Tensor((M, N), dtype)):
        with T.Kernel(4, 4, 2, threads=128 * 2) as (bx, by, bz):
            tid = T.get_thread_binding()
            if tid == 0:
                if msg_only:
                    T.print(msg="hello world")
                else:
                    T.print(bx + by + bz, msg="hello world")

    jit_kernel = tilelang.compile(program)
    profiler = jit_kernel.get_profiler()
    profiler.run_once()


def test_debug_print_msg():
    debug_print_msg(16, 16, msg_only=True)
    debug_print_msg(16, 16, msg_only=False)


def test_debug_print_global_buffer_is_printed_once_per_block(capfd):
    """A global buffer is dumped by one thread per block; block selection is
    left to the caller."""

    @T.prim_func
    def program(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(2, threads=32):
            T.print(A)
            B[0] = 1

    jit_kernel = tilelang.compile(program)
    profiler = jit_kernel.get_profiler()
    profiler.run_once()

    import torch

    torch.cuda.synchronize()
    dumped = [line for line in capfd.readouterr().out.splitlines() if "buffer=A" in line]
    assert len(dumped) == 8, f"expected one dump of 4 elements per block, got {len(dumped)} lines"


def test_debug_print_global_buffer_in_conditional_block(capfd):
    """Wrapping T.print in a block guard still prints from that block."""

    @T.prim_func
    def program(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(2, threads=32) as bx:
            if bx == 1:
                T.print(A)
            B[0] = 1

    jit_kernel = tilelang.compile(program)
    profiler = jit_kernel.get_profiler()
    profiler.run_once()

    import torch

    torch.cuda.synchronize()
    dumped = [line for line in capfd.readouterr().out.splitlines() if "buffer=A" in line]
    assert len(dumped) == 4, f"expected a dump of 4 elements from block 1, got {len(dumped)} lines"
    assert all("BlockIdx=(1, 0, 0)" in line for line in dumped), dumped


def test_debug_print_shared_buffer_is_printed_by_one_thread(capfd):
    """A shared buffer is printed by one thread per block, including when the
    block has more than one thread dimension."""

    @T.prim_func
    def program(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=(32, 2)):
            S = T.alloc_shared((4,), "int32")
            T.copy(A, S)
            T.sync_threads()
            T.print(S)
            B[0] = 1

    jit_kernel = tilelang.compile(program)
    profiler = jit_kernel.get_profiler()
    profiler.run_once()

    import torch

    torch.cuda.synchronize()
    dumped = [line for line in capfd.readouterr().out.splitlines() if "buffer=S" in line]
    assert len(dumped) == 4, f"expected a single dump of 4 elements, got {len(dumped)} lines"


if __name__ == "__main__":
    tilelang.testing.main()
