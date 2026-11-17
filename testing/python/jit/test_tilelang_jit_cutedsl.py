from tilelang import tvm as tvm
import tilelang.language as T
import tilelang.testing
import tilelang
import torch
import pytest


def _is_cutedsl_available():
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
        return True
    except (ImportError, AssertionError):
        return False


pytestmark = pytest.mark.skipif(not _is_cutedsl_available(), reason="CuTeDSL not installed")


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def matmul_jit_kernel(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def matmul_kernel_with_barrier(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    mbars,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            barriers = T.alloc_barrier(mbars)  # noqa: F841
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_jit_kernel(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=3,
    num_threads=128,
):
    program = matmul_jit_kernel(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    matmul_kernel = tilelang.compile(program, out_idx=-1, target="cutedsl")

    in_dtype = T.dtype(in_dtype).as_torch()
    out_dtype = T.dtype(out_dtype).as_torch()

    A = torch.randn(M, K, dtype=in_dtype).cuda()
    B = torch.randn(K, N, dtype=in_dtype).cuda()

    if trans_A:
        A = A.T
    if trans_B:
        B = B.T

    def ref_program(A, B):
        import torch

        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(out_dtype)
        return C

    ref_C = ref_program(A, B)
    C = matmul_kernel(A, B)

    tilelang.testing.torch_assert_close(C, ref_C, atol=1e-2, rtol=1e-2, max_mismatched_ratio=0.05)


@tilelang.testing.requires_cuda
def test_gemm_jit_kernel():
    run_gemm_jit_kernel(
        512,
        1024,
        768,
        False,
        False,
        "float16",
        "float16",
        "float16",
        128,
        256,
        32,
        2,
    )


def run_cutedsl_kernel_do_bench(
    M, N, K, trans_A, trans_B, in_dtype, out_dtype, dtypeAccum, block_M, block_N, block_K, num_stages=3, num_threads=128
):
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    matmul_kernel = tilelang.compile(program, target="cutedsl")

    profiler = matmul_kernel.get_profiler()

    cutedsl_latency = profiler.do_bench(func=matmul_kernel)
    print(f"CuTeDSL Latency: {cutedsl_latency} ms")

    assert cutedsl_latency is not None

    tvm_latency = profiler.do_bench()
    print(f"TVM Latency: {tvm_latency} ms")

    assert tvm_latency is not None


@tilelang.testing.requires_cuda
@pytest.mark.perf
def test_cutedsl_kernel_do_bench():
    run_cutedsl_kernel_do_bench(512, 1024, 768, False, False, "float16", "float16", "float16", 128, 256, 32, 2)


def run_cutedsl_kernel_multi_stream(
    M, N, K, trans_A, trans_B, in_dtype, out_dtype, dtypeAccum, block_M, block_N, block_K, num_stages=3, num_threads=128
):
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    matmul_kernel = tilelang.compile(program, target="cutedsl")
    in_dtype = T.dtype(in_dtype).as_torch()
    out_dtype = T.dtype(out_dtype).as_torch()
    tensor_a = torch.randn(M, K, dtype=in_dtype).cuda()
    tensor_b = torch.randn(K, N, dtype=in_dtype).cuda()

    if trans_A:
        tensor_a = tensor_a.T
    if trans_B:
        tensor_b = tensor_b.T
    tensor_c = torch.randn(M, N, dtype=out_dtype).cuda()

    num_streams = 4
    for _ in range(num_streams):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            matmul_kernel(tensor_a, tensor_b, tensor_c)


@tilelang.testing.requires_cuda
def test_cutedsl_kernel_multi_stream():
    run_cutedsl_kernel_multi_stream(512, 1024, 768, False, False, "float16", "float16", "float16", 128, 256, 32, 2)


def run_cutedsl_dynamic_shape(
    M, N, K, trans_A, trans_B, in_dtype, out_dtype, dtypeAccum, block_M, block_N, block_K, num_stages=3, num_threads=128
):
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    matmul_kernel = tilelang.compile(program, target="cutedsl")
    if isinstance(M, T.Var):
        M = 1024
    if isinstance(N, T.Var):
        N = 1024
    if isinstance(K, T.Var):
        K = 768

    in_dtype = T.dtype(in_dtype).as_torch()
    out_dtype = T.dtype(out_dtype).as_torch()

    tensor_a = torch.randn(M, K, dtype=in_dtype).cuda()
    tensor_b = torch.randn(K, N, dtype=in_dtype).cuda()

    if trans_A:
        tensor_a = tensor_a.T
    if trans_B:
        tensor_b = tensor_b.T
    tensor_c = torch.randn(M, N, dtype=out_dtype).cuda()

    matmul_kernel(tensor_a, tensor_b, tensor_c)

    tensor_ref_c = torch.matmul(tensor_a.to(torch.float), tensor_b.to(torch.float)).to(out_dtype)
    tilelang.testing.torch_assert_close(tensor_c, tensor_ref_c, atol=1e-2, rtol=1e-2, max_mismatched_ratio=0.05)


@tilelang.testing.requires_cuda
def test_cutedsl_dynamic_shape():
    run_cutedsl_dynamic_shape(T.dynamic("m"), 1024, 768, False, False, "float16", "float16", "float16", 128, 256, 32, 2)

    run_cutedsl_dynamic_shape(T.dynamic("m"), T.dynamic("n"), 768, False, False, "float16", "float16", "float16", 128, 256, 32, 2)

    run_cutedsl_dynamic_shape(
        T.dynamic("m"), T.dynamic("n"), T.dynamic("k"), False, False, "float16", "float16", "float16", 128, 256, 32, 2
    )


def check_hopper():
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    compute_capability = props.major, props.minor
    return compute_capability == (9, 0)


@tilelang.testing.requires_cuda
def test_cutedsl_launch_follows_the_torch_stream():
    """The launch must land on the stream torch is recording.

    Passing the legacy default stream launches nothing into the captured graph
    and replays a stale result.

    See https://github.com/tile-ai/tilelang/issues/3693.
    """

    @T.prim_func
    def copy(X: T.Tensor((1024,), "float32"), Y: T.Tensor((1024,), "float32")):
        with T.Kernel(8, threads=128) as bx:
            for i in T.Parallel(128):
                Y[bx * 128 + i] = X[bx * 128 + i] * 2.0

    kernel = tilelang.compile(copy, target="cutedsl")
    x = torch.arange(1024, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)

    kernel(x, y)
    torch.cuda.synchronize()
    assert torch.equal(y, x * 2.0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel(x, y)
    x.fill_(1.0)
    y.fill_(-1.0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, torch.full_like(y, 2.0))


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32"])
@pytest.mark.parametrize("scope", ["global", "shared"])
@pytest.mark.parametrize("threads", [256, 64], ids=["scalar", "vector"])
def test_cutedsl_unsigned_reads_stay_unsigned(dtype, scope, threads):
    """Values read from an unsigned buffer keep their unsigned value.

    See https://github.com/tile-ai/tilelang/issues/2974.
    """
    N = 256
    bits = T.dtype(dtype).bits
    half = (1 << (bits - 1)) - 1
    signed = f"int{bits}"

    @T.prim_func
    def unsigned_reads(
        A: T.Tensor((N,), dtype),
        S: T.Tensor((N,), signed),
        C: T.Tensor((8, N), "int64"),
        F: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1, threads=threads):
            if scope == "shared":
                src = T.alloc_shared((N,), dtype)
                ssrc = T.alloc_shared((N,), signed)
                T.copy(A, src)
                T.copy(S, ssrc)
            else:
                src = A
                ssrc = S
            for i in T.Parallel(N):
                C[0, i] = T.cast(src[i], "int64")
                C[1, i] = T.cast(src[i] > half, "int64")
                C[2, i] = T.cast(src[i] >> (bits - 1), "int64")
                C[3, i] = T.cast(src[i] // 3, "int64")
                C[4, i] = T.cast(T.max(src[i], T.cast(half + 1, dtype)), "int64")
                C[5, i] = T.cast(T.min(src[i], T.cast(half + 1, dtype)), "int64")
                C[6, i] = T.cast(T.reinterpret(ssrc[i], dtype), "int64")
                C[7, i] = T.cast(ssrc[i], "int64")
                F[i] = T.cast(src[i], "float32")

    kernel = tilelang.compile(unsigned_reads, target="cutedsl")
    i = torch.arange(N, dtype=torch.int64, device="cuda")
    a = (i * ((1 << bits) // N) + i % 7) % (1 << bits)
    s = torch.where(a > half, a - (1 << bits), a)
    c = torch.empty(8, N, dtype=torch.int64, device="cuda")
    f = torch.empty(N, dtype=torch.float32, device="cuda")
    kernel(a.to(T.dtype(dtype).as_torch()), s.to(T.dtype(signed).as_torch()), c, f)

    assert torch.equal(c[0], a)
    assert torch.equal(c[1], (a > half).long())
    assert torch.equal(c[2], a >> (bits - 1))
    assert torch.equal(c[3], a // 3)
    assert torch.equal(c[4], a.clamp(min=half + 1))
    assert torch.equal(c[5], a.clamp(max=half + 1))
    assert torch.equal(c[6], a)
    assert torch.equal(c[7], s)
    assert torch.equal(f, a.float())


if __name__ == "__main__":
    tilelang.testing.main()
