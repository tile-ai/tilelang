# pytest correctness_evaluation.py -n 32
import pytest
from tilelang import tvm as tvm
import tilelang.testing
from tilelang import language as T
import torch


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
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope="shared.dyn")
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope="shared.dyn")
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


def _compile_and_check(
    program,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
):
    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            # tilelang.PassConfigKey.TIR_USE_ASYNC_COPY: False,
        },
    )

    print(kernel.get_kernel_source())

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

    def ref_program(A, B):
        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if in_dtype == T.float32:
            A = (A.view(torch.int32) - 0x1000).view(torch.float32)
            B = (B.view(torch.int32) - 0x1000).view(torch.float32)
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)
    print("assert_allclose")


def run_gemm(
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
    num_stages=2,
    num_threads=128,
):
    if block_N >= 256 or block_M >= 256 or block_K >= 256:
        num_stages = 0
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

    _compile_and_check(program, trans_A, trans_B, in_dtype, out_dtype)


def matmul_rs(
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
    A_frag_shape = A_shared_shape

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope="shared.dyn")
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope="shared.dyn")
            A_frag = T.alloc_fragment(A_frag_shape, in_dtype)
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
                T.copy(A_shared, A_frag)
                T.gemm_v2(A_frag, B_shared, C_local, trans_A, trans_B)
                # T.gemm(A_frag, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_rs(
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
    num_stages=2,
    num_threads=128,
):
    if block_N >= 256 or block_M >= 256 or block_K >= 256:
        num_stages = 0
    program = matmul_rs(
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
    _compile_and_check(program, trans_A, trans_B, in_dtype, out_dtype)


def matmul_sr(
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
    B_frag_shape = B_shared_shape

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope="shared.dyn")
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope="shared.dyn")
            B_frag = T.alloc_fragment(B_frag_shape, in_dtype)
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
                T.copy(B_shared, B_frag)
                T.gemm_v2(A_shared, B_frag, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_sr(
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
    num_stages=2,
    num_threads=128,
):
    if block_N >= 256 or block_M >= 256 or block_K >= 256:
        num_stages = 0
    program = matmul_sr(
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

    _compile_and_check(program, trans_A, trans_B, in_dtype, out_dtype)


def matmul_rr(
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
    A_frag_shape = A_shared_shape
    B_frag_shape = B_shared_shape

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope="shared.dyn")
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope="shared.dyn")
            A_frag = T.alloc_fragment(A_frag_shape, in_dtype)
            B_frag = T.alloc_fragment(B_frag_shape, in_dtype)
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
                T.copy(A_shared, A_frag)
                T.copy(B_shared, B_frag)
                T.gemm_v2(A_frag, B_frag, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_rr(
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
    num_stages=2,
    num_threads=128,
):
    if block_N >= 256 or block_M >= 256 or block_K >= 256:
        num_stages = 0
    program = matmul_rr(
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

    _compile_and_check(program, trans_A, trans_B, in_dtype, out_dtype)


M_VALUES = [64, 128, 256]
N_VALUES = [16, 32, 64, 128, 256, 512]
K_VALUES = [16, 32, 64, 128]
K_VALUES_8Bit = [32, 64, 128]
FALSE_TRUE_CASES = (
    [
        pytest.param(
            k,
            T.float16,
            T.float16,
            T.float16,
            id=f"K{k}-float16-float16-float16",
        )
        for k in K_VALUES
    ]
    + [
        pytest.param(
            k,
            T.int8,
            T.int32,
            T.int32,
            id="K32-int8-int32-int32",
        )
        for k in K_VALUES_8Bit
    ]
    + [
        pytest.param(
            k,
            T.float8_e5m2,
            T.float32,
            T.float32,
            id="K32-float8_e5m2-float32-float32",
        )
        for k in K_VALUES_8Bit
    ]
    + [
        pytest.param(
            k,
            T.float8_e4m3fn,
            T.float32,
            T.float32,
            id="K32-float8_e4m3-float32-float32",
        )
        for k in K_VALUES_8Bit
    ]
)


def _ensure_torch_dtypes(*dtype_names):
    import torch

    for name in set(dtype_names):
        if not hasattr(torch, name):
            pytest.skip(f"Torch does not expose dtype {name}")


def run_gemm_rs_false_true(m, n, k, in_dtype, out_dtype, accum_dtype):
    run_gemm_rs(m, n, k * 3, False, True, in_dtype, out_dtype, accum_dtype, m, n, k)


def run_gemm_rs_false_false(m, n, k):
    run_gemm_rs(m, n, k * 3, False, False, T.float16, T.float16, T.float16, m, n, k)


def run_gemm_rs_true_false(m, n, k):
    run_gemm_rs(m, n, k * 3, True, False, T.float16, T.float16, T.float16, m, n, k)


def run_gemm_rs_true_true(m, n, k):
    run_gemm_rs(m, n, k * 3, True, True, T.float16, T.float16, T.float16, m, n, k)


def run_gemm_sr_false_true(m, n, k, in_dtype, out_dtype, accum_dtype):
    run_gemm_sr(m, n, k * 3, False, True, in_dtype, out_dtype, accum_dtype, m, n, k)


def run_gemm_sr_false_false(m, n, k):
    run_gemm_sr(m, n, k * 3, False, False, T.float16, T.float16, T.float16, m, n, k)


def run_gemm_sr_true_false(m, n, k):
    run_gemm_sr(m, n, k * 3, True, False, T.float16, T.float16, T.float16, m, n, k)


def run_gemm_sr_true_true(m, n, k):
    run_gemm_sr(m, n, k * 3, True, True, T.float16, T.float16, T.float16, m, n, k)


def run_gemm_rr_false_true(m, n, k, in_dtype, out_dtype, accum_dtype):
    run_gemm_rr(m, n, k * 3, False, True, in_dtype, out_dtype, accum_dtype, m, n, k)


def run_gemm_rr_false_false(m, n, k):
    run_gemm_rr(m, n, k * 3, False, False, T.float16, T.float16, T.float16, m, n, k)


def run_gemm_rr_true_false(m, n, k):
    run_gemm_rr(m, n, k * 3, True, False, T.float16, T.float16, T.float16, m, n, k)


def run_gemm_rr_true_true(m, n, k):
    run_gemm_rr(m, n, k * 3, True, True, T.float16, T.float16, T.float16, m, n, k)


TRANS_CASES = [
    pytest.param(False, False, id="nn"),
    pytest.param(False, True, id="nt"),
    pytest.param(True, False, id="tn"),
    pytest.param(True, True, id="tt"),
]


@pytest.fixture(scope="module", autouse=True)
def _setup_tilelang_environment():
    tilelang.disable_cache()
    tilelang.testing.set_random_seed(42)


@pytest.mark.parametrize("m", M_VALUES, ids=lambda v: f"M{v}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("k,in_dtype,out_dtype,accum_dtype", FALSE_TRUE_CASES)
def test_gemm_false_true(m, n, k, in_dtype, out_dtype, accum_dtype):
    import torch

    required_torch_attrs = {
        in_dtype,
        out_dtype,
        accum_dtype,
    }
    for attr in required_torch_attrs:
        if not hasattr(torch, attr):
            pytest.skip(f"Torch does not expose dtype {attr}")
    run_gemm(
        m,
        n,
        k * 3,
        False,
        True,
        in_dtype,
        out_dtype,
        accum_dtype,
        m,
        n,
        k,
    )


@pytest.mark.parametrize("m", M_VALUES, ids=lambda v: f"M{v}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("k", K_VALUES, ids=lambda v: f"K{v}")
def test_gemm_false_false(m, n, k):
    run_gemm(
        m,
        n,
        k * 3,
        False,
        False,
        T.float16,
        T.float16,
        T.float16,
        m,
        n,
        k,
    )


@pytest.mark.parametrize("m", M_VALUES, ids=lambda v: f"M{v}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("k", K_VALUES, ids=lambda v: f"K{v}")
def test_gemm_true_false(m, n, k):
    run_gemm(
        m,
        n,
        k * 3,
        True,
        False,
        T.float16,
        T.float16,
        T.float16,
        m,
        n,
        k,
    )


@pytest.mark.parametrize("m", M_VALUES, ids=lambda v: f"M{v}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("k", K_VALUES, ids=lambda v: f"K{v}")
def test_gemm_true_true(m, n, k):
    run_gemm(
        m,
        n,
        k * 3,
        True,
        True,
        T.float16,
        T.float16,
        T.float16,
        m,
        n,
        k,
    )


@pytest.mark.parametrize("m", M_VALUES, ids=lambda v: f"M{v}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("k,in_dtype,out_dtype,accum_dtype", FALSE_TRUE_CASES)
def test_gemm_rs_false_true(m, n, k, in_dtype, out_dtype, accum_dtype):
    _ensure_torch_dtypes(in_dtype, out_dtype, accum_dtype)
    run_gemm_rs_false_true(m, n, k, in_dtype, out_dtype, accum_dtype)


@pytest.mark.parametrize("m", M_VALUES, ids=lambda v: f"M{v}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("k", K_VALUES, ids=lambda v: f"K{v}")
def test_gemm_rs_false_false(m, n, k):
    _ensure_torch_dtypes(T.float16)
    run_gemm_rs_false_false(m, n, k)


@pytest.mark.parametrize("m", M_VALUES, ids=lambda v: f"M{v}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("k", K_VALUES, ids=lambda v: f"K{v}")
def test_gemm_rs_true_false(m, n, k):
    _ensure_torch_dtypes(T.float16)
    run_gemm_rs_true_false(m, n, k)


@pytest.mark.parametrize("m", M_VALUES, ids=lambda v: f"M{v}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("k", K_VALUES, ids=lambda v: f"K{v}")
def test_gemm_rs_true_true(m, n, k):
    _ensure_torch_dtypes(T.float16)
    run_gemm_rs_true_true(m, n, k)


@pytest.mark.parametrize("m", M_VALUES, ids=lambda v: f"M{v}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("k,in_dtype,out_dtype,accum_dtype", FALSE_TRUE_CASES)
def test_gemm_sr_false_true(m, n, k, in_dtype, out_dtype, accum_dtype):
    _ensure_torch_dtypes(in_dtype, out_dtype, accum_dtype)
    run_gemm_sr_false_true(m, n, k, in_dtype, out_dtype, accum_dtype)


@pytest.mark.parametrize("m", M_VALUES, ids=lambda v: f"M{v}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("k", K_VALUES, ids=lambda v: f"K{v}")
def test_gemm_sr_false_false(m, n, k):
    _ensure_torch_dtypes(T.float16)
    run_gemm_sr_false_false(m, n, k)


@pytest.mark.parametrize("m", M_VALUES, ids=lambda v: f"M{v}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("k", K_VALUES, ids=lambda v: f"K{v}")
def test_gemm_sr_true_false(m, n, k):
    _ensure_torch_dtypes(T.float16)
    run_gemm_sr_true_false(m, n, k)


@pytest.mark.parametrize("m", M_VALUES, ids=lambda v: f"M{v}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("k", K_VALUES, ids=lambda v: f"K{v}")
def test_gemm_sr_true_true(m, n, k):
    _ensure_torch_dtypes(T.float16)
    run_gemm_sr_true_true(m, n, k)


@pytest.mark.parametrize("m", M_VALUES, ids=lambda v: f"M{v}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("k,in_dtype,out_dtype,accum_dtype", FALSE_TRUE_CASES)
def test_gemm_rr_false_true(m, n, k, in_dtype, out_dtype, accum_dtype):
    _ensure_torch_dtypes(in_dtype, out_dtype, accum_dtype)
    run_gemm_rr_false_true(m, n, k, in_dtype, out_dtype, accum_dtype)


@pytest.mark.parametrize("m", M_VALUES, ids=lambda v: f"M{v}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("k", K_VALUES, ids=lambda v: f"K{v}")
def test_gemm_rr_false_false(m, n, k):
    _ensure_torch_dtypes(T.float16)
    run_gemm_rr_false_false(m, n, k)


@pytest.mark.parametrize("m", M_VALUES, ids=lambda v: f"M{v}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("k", K_VALUES, ids=lambda v: f"K{v}")
def test_gemm_rr_true_false(m, n, k):
    _ensure_torch_dtypes(T.float16)
    run_gemm_rr_true_false(m, n, k)


@pytest.mark.parametrize("m", M_VALUES, ids=lambda v: f"M{v}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("k", K_VALUES, ids=lambda v: f"K{v}")
def test_gemm_rr_true_true(m, n, k):
    _ensure_torch_dtypes(T.float16)
    run_gemm_rr_true_true(m, n, k)


if __name__ == "__main__":
    tilelang.testing.main()

    # # Test Pass
    # for m in [64, 128, 256]:
    #     for n in [16, 32, 64, 128]:
    #         for k in [16, 32, 64, 128]:
    #             print(f"======================= Test {m} {n} {k} False True =============================")
    #             run_gemm(m, n, k * 3, False, True, T.float16, T.float16, T.float16, m, n, k, 2, 128)
    #             print(f"Test {m} {n} {k} Pass")

    # # Test Pass
    # for m in [64, 128, 256]:
    #     for n in [16, 32, 64, 128]:
    #         for k in [16, 32, 64, 128]:
    #             print(f"======================= Test {m} {n} {k} False False =============================")
    #             run_gemm(m, n, k * 3, False, False, T.float16, T.float16, T.float16, m, n, k, 2, 128)
    #             print(f"Test {m} {n} {k} Pass")

    # # Test Pass
    # for m in [64, 128, 256]:
    #     for n in [16, 32, 64, 128]:
    #         for k in [16, 32, 64, 128]:
    #             print(f"======================= Test {m} {n} {k} True False =============================")
    #             run_gemm(m, n, k * 3, True, False, T.float16, T.float16, T.float16, m, n, k, 2, 128)
    #             print(f"Test {m}, {n} {k} Pass")
    #         print(f"Test {n} Pass")

    # # Test Pass
    # for m in [64, 128, 256]:
    #     for n in [16, 32, 64, 128]:
    #         for k in [16, 32, 64, 128]:
    #             print(f"======================= Test {m} {n} {k} True True =============================")
    #             run_gemm(m, n, k * 3, True, True, T.float16, T.float16, T.float16, m, n, k, 2, 128)
    #             print(f"Test {m}, {n} {k} Pass")
    #         print(f"Test {n} Pass")

    # Test Pass
    # for m in [64, 128, 256]:
    #     for n in [16, 32, 64, 128]:
    #         for k in [16, 32, 64, 128]:
    #             print(f"======================= Test {m} {n} {k} False True =============================")
    #             run_gemm_rs(m, n, k * 3, False, True, T.float16, T.float16, T.float16, m, n, k, 2, 128)
    #             print(f"Test {m} {n} {k} Pass")

    # for n in [16, 32, 64, 128]:
    #     for k in [16, 32, 64, 128]:
    #         run_gemm_rs(64, n, k, False, False, T.float16, T.float16, T.float16, 64, n, k, 0, 256)
    #         print(f"Test {64} {n} {k} Pass")

    # for n in [16, 32, 64, 128]:
    #     for k in [16, 32, 64, 128]:
    #         run_gemm(64, n, k, False, False, T.float16, T.float16, T.float16, 64, n, k, 0, 256)
    #         print(f"Test {64} {n} {k} Pass")
