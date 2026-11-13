import tilelang
import tilelang.language as T
import torch
import tilelang.testing
import pytest
from tilelang import tvm

tilelang.testing.set_random_seed()


@tilelang.jit(out_idx=[1])
def parallel_elementwise_static(length=256, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((length,), dtype),
        B: T.Tensor((length,), dtype),
    ):
        with T.Kernel(1, threads=length) as _:
            for i in T.Parallel(length):
                B[i] = A[i] + 1.0

    return main


@tilelang.jit(out_idx=[1])
def parallel_elementwise_dynamic(max_len=512, threads=256, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((max_len,), dtype),
        B: T.Tensor((max_len,), dtype),
        valid_len: T.int32,
    ):
        with T.Kernel(1, threads=threads) as _:
            for i in T.Parallel(max_len):
                B[i] = 0.0
            span = T.min(valid_len, max_len)
            for i in T.Parallel(span):
                B[i] = A[i] - 1.0

    return main


def _require_cuda_tensor(shape, dtype=torch.float32):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        return torch.randn(*shape, device="cuda", dtype=dtype)
    except RuntimeError as err:
        pytest.skip(f"CUDA runtime unavailable: {err}")


PARALLEL_DYNAMIC_VALID_LENGTHS = [0, 13, 200, 600]


def test_parallel_static_extent():
    kernel = parallel_elementwise_static(length=256)
    data = _require_cuda_tensor((256,), torch.float32)
    result = kernel(data)
    torch.testing.assert_close(result, data + 1.0, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("enable_let_inline", [False, True])
@pytest.mark.parametrize("rows,cols,threads,thread_start", [(1, 2, 2, 0), (2, 8, 4, 0), (3, 5, 8, 0), (2, 8, 8, 4)])
def test_parallel_let_indices(rows, cols, threads, thread_start, enable_let_inline):
    if not torch.cuda.is_available():
        pytest.skip("CUDA or ROCm runtime unavailable")
    size = 2 * rows * cols + 8

    @T.prim_func
    def main(A: T.Tensor((size,), T.int32), B: T.Tensor((size,), T.int32)):
        with T.Kernel(2, threads=threads) as bx:
            if T.get_thread_binding() >= thread_start:
                for i, j in T.Parallel(rows, cols):
                    offset = bx * rows * cols + i * cols
                    idx = offset + j + 4
                    B[idx] = A[idx] * 3 + idx
                for i, j in T.Parallel(rows, cols):
                    offset = bx * rows * cols + i * cols
                    idx = offset + j + 4
                    B[idx] = B[idx] + 5

    config = {tilelang.PassConfigKey.TL_SIMPLIFY: {tilelang.PassConfigKey.TL_SIMPLIFY_ENABLE_LET_INLINE: enable_let_inline}}
    target = "hip" if torch.version.hip else "cuda"
    kernel = tilelang.compile(main, target=target, execution_backend="tvm_ffi", pass_configs=config)
    data = torch.arange(size, dtype=torch.int32, device="cuda")
    output = torch.full_like(data, -7)
    kernel(data, output)
    expected = torch.full_like(data, -7)
    expected[4:-4] = data[4:-4] * 4 + 5
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "valid_len",
    PARALLEL_DYNAMIC_VALID_LENGTHS,
    ids=[f"valid_len={value}" for value in PARALLEL_DYNAMIC_VALID_LENGTHS],
)
def test_parallel_dynamic_extent(valid_len):
    kernel = parallel_elementwise_dynamic(max_len=512, threads=256)
    data = _require_cuda_tensor((512,), torch.float32)
    out = kernel(data, valid_len)
    reference = torch.zeros_like(data)
    clip = min(valid_len, data.shape[0])
    reference[:clip] = data[:clip] - 1.0
    torch.testing.assert_close(out, reference, atol=1e-5, rtol=1e-5)


@tilelang.jit
def _parallel_vectorize_local_and_var():
    with T.Kernel(1) as _:
        x = T.alloc_fragment([256], T.float32)
        y = T.alloc_fragment([256], T.float32)
        z = T.alloc_var(T.float32)
        for i in T.parallel(256):
            y[i] = x[i] * z


def test_parallel_vectorize_var():
    source = _parallel_vectorize_local_and_var.get_kernel_source()
    # register-register loops vectorize too; the var operand becomes a
    # scalar broadcast inside the packed multiply
    assert "float2" in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("update", ["atomic_add", "read_modify_write", "extern_access_ptr", "extern_address_of"])
def test_replicated_parallel_writes_once(update):
    """A replicated loop (64 iterations on 128 threads) that stores a fragment
    runs its atomics and read-modify-writes on one replica only (#3870)."""
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32), C: T.Tensor((n,), T.float32)):
        with T.Kernel(1, threads=128):
            L = T.alloc_fragment((n,), T.float32)
            for i in T.Parallel(n):
                L[i] = B[i]
                if update == "atomic_add":
                    T.atomic_add(A[i], L[i])
                elif update == "read_modify_write":
                    A[i] = A[i] + L[i]
                elif update == "extern_access_ptr":
                    T.call_extern("float32", "atomicAdd", A.access_ptr("rw", offset=i), L[i])
                else:
                    T.call_extern("float32", "atomicAdd", T.address_of(A[i]), L[i])
                C[i] = L[i]

    kernel = tilelang.compile(main)
    source = kernel.get_kernel_source()
    assert "& 63" in source, "expected a replicated loop layout"
    a = torch.zeros(n, device="cuda")
    b = torch.arange(n, dtype=torch.float32, device="cuda")
    c = torch.empty(n, device="cuda")
    kernel(a, b, c)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(c, b, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("pointer", ["bound_inside", "bound_outside", "only_access"])
def test_replicated_parallel_guards_writes_through_pointers(pointer):
    """A write through a pointer runs on one replica, while binding the pointer
    stays on every replica, also when the loop touches shared/global memory
    only through that pointer (#3870)."""
    n = 64

    @T.prim_func
    def main(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32), C: T.Tensor((n,), T.float32)):
        with T.Kernel(1, threads=128):
            L = T.alloc_fragment((n,), T.float32)
            if pointer == "bound_outside":
                ptr = T.bind(T.access_ptr(A[0], "rw"))
            if pointer == "only_access":
                for i in T.Parallel(n):
                    L[i] = T.cast(i, T.float32)
                    T.call_extern("float32", "extern_update", A.access_ptr("rw", offset=i), L[i])
                for i in T.Parallel(n):
                    C[i] = L[i]
            else:
                for i in T.Parallel(n):
                    L[i] = B[i]
                    if pointer == "bound_inside":
                        ptr = T.bind(T.access_ptr(A[i], "rw"))
                    T.call_extern("float32", "extern_update", ptr, L[i])
                    C[i] = L[i]

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
    with target:
        source = tilelang.lower(main, target=target).kernel_source
    guard = source.index("if ((((int)threadIdx.x) >> 6) == 0) {")
    if pointer != "only_access":
        assert source.index("void* ptr = ") < guard
    assert guard < source.index("extern_update(")
    assert source.index("C[") > source.index("extern_update(")


@tilelang.testing.requires_cuda
def test_replicated_parallel_reads_before_its_update():
    """A value read before the loop's single-replica update of the same buffer
    is only consistent on that replica (#3870)."""
    n = 64

    @T.prim_func
    def snapshot(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32), C: T.Tensor((n,), T.float32)):
        with T.Kernel(1, threads=128):
            L = T.alloc_fragment((n,), T.float32)
            for i in T.Parallel(n):
                L[i] = B[i]
                C[i] = A[i]
                A[i] = L[i]

    kernel = tilelang.compile(snapshot)
    a = torch.arange(n, dtype=torch.float32, device="cuda")
    b = -torch.arange(n, dtype=torch.float32, device="cuda") - 1
    c = torch.empty(n, device="cuda")
    expected_c = a.clone()
    kernel(a, b, c)
    torch.testing.assert_close(c, expected_c, rtol=0, atol=0)
    torch.testing.assert_close(a, b, rtol=0, atol=0)

    @T.prim_func
    def into_fragment(A: T.Tensor((n,), T.float32), C: T.Tensor((n,), T.float32)):
        with T.Kernel(1, threads=128):
            L = T.alloc_fragment((n,), T.float32)
            for i in T.Parallel(n):
                L[i] = A[i]
                A[i] = L[i] + 1.0
                C[i] = L[i]

    with pytest.raises(Exception, match="also updates"):
        tilelang.compile(into_fragment)

    @T.prim_func
    def atomic_condition(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32), C: T.Tensor((n,), T.float32)):
        with T.Kernel(1, threads=128):
            L = T.alloc_fragment((n,), T.float32)
            for i in T.Parallel(n):
                L[i] = B[i]
                if T.atomic_add(A[i], L[i], return_prev=True) > 0:
                    C[i] = L[i]

    with pytest.raises(Exception, match="every replica needs the result"):
        tilelang.compile(atomic_condition)

    @T.prim_func
    def through_pointer(A: T.Tensor((n,), T.float32), C: T.Tensor((n,), T.float32)):
        with T.Kernel(1, threads=128):
            L = T.alloc_fragment((n,), T.float32)
            for i in T.Parallel(n):
                L[i] = T.call_extern("float32", "load", A.access_ptr("r", offset=i))
                T.atomic_add(A[i], 1.0)
                C[i] = L[i]

    @T.prim_func
    def across_iterations(A: T.Tensor((n,), T.float32), C: T.Tensor((n,), T.float32)):
        with T.Kernel(1, threads=128):
            L = T.alloc_fragment((n,), T.float32)
            for i in T.Parallel(n):
                for _ in T.serial(2):
                    L[i] = A[i]
                    T.sync_threads()
                    A[i] = A[i] + L[i]
                C[i] = L[i]

    for program in (through_pointer, across_iterations):
        with pytest.raises(Exception, match="also updates"):
            tilelang.compile(program)

    @T.prim_func
    def atomic_pointer_index(
        A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32), C: T.Tensor((n,), T.float32), Cnt: T.Tensor((1,), T.int32)
    ):
        with T.Kernel(1, threads=128):
            L = T.alloc_fragment((n,), T.float32)
            for i in T.Parallel(n):
                L[i] = B[i]
                ptr = T.bind(T.access_ptr(A[T.atomic_add(Cnt[0], 1, return_prev=True)], "rw"))
                T.call_extern("float32", "extern_update", ptr, L[i])
                C[i] = L[i]

    with pytest.raises(Exception, match="every replica needs the result"):
        tilelang.compile(atomic_pointer_index)


if __name__ == "__main__":
    tilelang.testing.main()
