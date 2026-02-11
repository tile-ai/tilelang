import tilelang
from tilelang import tvm as tvm
import tilelang.language as T
import tilelang.testing

SM100_TARGET = "cuda -arch=sm_100"
SM80_TARGET = "cuda -arch=sm_80"


def vec_add_f32x2(M: int = 128):
    @T.prim_func
    def main(
        A: T.Tensor((M, 2), dtype=T.float32),
        B: T.Tensor((M, 2), dtype=T.float32),
        C: T.Tensor((M, 2), dtype=T.float32),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            for v in T.vectorized(2):
                C[tid, v] = A[tid, v] + B[tid, v]

    return main


def vec_mul_f32x2(M: int = 128):
    @T.prim_func
    def main(
        A: T.Tensor((M, 2), dtype=T.float32),
        B: T.Tensor((M, 2), dtype=T.float32),
        C: T.Tensor((M, 2), dtype=T.float32),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            for v in T.vectorized(2):
                C[tid, v] = A[tid, v] * B[tid, v]

    return main


def vec_fma_f32x2(M: int = 128):
    @T.prim_func
    def main(
        A: T.Tensor((M * 2,), dtype=T.float32),
        B: T.Tensor((M * 2,), dtype=T.float32),
        D: T.Tensor((M * 2,), dtype=T.float32),
        C: T.Tensor((M * 2,), dtype=T.float32),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            idx = T.Ramp(tid * 2, 1, 2)
            C[idx] = T.fma2(A[idx], B[idx], D[idx])

    return main


def auto_vec_add_f32x2(M: int = 128):
    @T.prim_func
    def main(
        A: T.Tensor((M, 2), dtype=T.float32),
        B: T.Tensor((M, 2), dtype=T.float32),
        C: T.Tensor((M, 2), dtype=T.float32),
    ):
        # Use T.Parallel instead of T.vectorized/T.fadd2 and rely on TileLang's
        # vectorization planner to turn the innermost extent-2 axis into a
        # float32x2 vector op, which should lower to tl::fadd2 in CUDA codegen.
        with T.Kernel(1, 1, threads=M) as (bx, by):
            for i, v in T.Parallel(M, 2):
                C[i, v] = A[i, v] + B[i, v]

    return main


def _lower_to_cuda_source(func, target: str = SM100_TARGET) -> str:
    with tvm.transform.PassContext(), tvm.target.Target(target):
        artifact = tilelang.lower(func, target=target)
    assert artifact.kernel_source is not None
    return artifact.kernel_source


@tilelang.testing.requires_cuda
def test_cuda_codegen_fadd2():
    src = _lower_to_cuda_source(vec_add_f32x2(), target=SM100_TARGET)
    assert "tl::fadd2" in src


@tilelang.testing.requires_cuda
def test_cuda_codegen_fmul2():
    src = _lower_to_cuda_source(vec_mul_f32x2(), target=SM100_TARGET)
    assert "tl::fmul2" in src


@tilelang.testing.requires_cuda
def test_cuda_codegen_fma2():
    src = _lower_to_cuda_source(vec_fma_f32x2(), target=SM100_TARGET)
    assert "tl::fma2" in src


@tilelang.testing.requires_cuda
def test_cuda_codegen_auto_vectorize_fadd2():
    src = _lower_to_cuda_source(auto_vec_add_f32x2(), target=SM100_TARGET)
    assert "tl::fadd2" in src


@tilelang.testing.requires_cuda
def test_cuda_codegen_no_fadd2_before_sm100():
    src = _lower_to_cuda_source(vec_add_f32x2(), target=SM80_TARGET)
    assert "tl::fadd2" not in src


if __name__ == "__main__":
    tilelang.testing.main()
