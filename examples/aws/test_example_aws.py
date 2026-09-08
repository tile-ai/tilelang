import tilelang.testing

import flash_attention
import gemm


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_example_gemm():
    gemm.main(kernel="gemm", M=512, N=4096, K=512)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_example_gemm_manual():
    gemm.main(kernel="gemm_manual", M=512, N=4096, K=512)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_example_gemm_ws():
    gemm.main(kernel="gemm_ws", M=512, N=4096, K=512)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_example_flash_attention():
    flash_attention.main(kernel="flash_attention", batch=1, heads=4, seq_len=1024)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_example_flash_attention_manual():
    flash_attention.main(kernel="flash_attention_manual", batch=1, heads=4, seq_len=1024)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_example_flash_attention_ws():
    flash_attention.main(kernel="flash_attention_ws", batch=1, heads=4, seq_len=1024)


if __name__ == "__main__":
    tilelang.testing.main()
