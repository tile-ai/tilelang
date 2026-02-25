cd testing

N=""
N="-n 30"
pytest --verbose --color=yes --durations=0 --showlocals --cache-clear \
  $N --ignore=./python/runtime --ignore=./python/transform \
  ./python/
  python/debug/test_tilelang_debug_print.py::test_debug_print_buffer_rocm_fp8

  python/amd/test_tilelang_gemm_mfma_preshuffle.py::test_assert_tl_matmul
  python/amd/test_tilelang_gemm_mfma_intrinsic.py::test_assert_tl_matmul[128-128-128-float8_e4m3fn-float16-float32-False-True-1]
#   ./python/
#   ./python/



# FAILED python/amd/test_tilelang_gemm_mfma_preshuffle.py::test_assert_tl_matmul[256-256-512-float8_e4m3fnuz-float32-float32-False-True-1-True-False] - AssertionError: Tensor-likes are not close!
# FAILED python/amd/test_tilelang_gemm_mfma_preshuffle.py::test_assert_tl_matmul[256-256-512-float8_e4m3fnuz-float32-float32-False-False-1-True-False] - AssertionError: Tensor-likes are not close!
# FAILED python/amd/test_tilelang_gemm_mfma_preshuffle.py::test_assert_tl_matmul[256-256-512-float8_e4m3fnuz-float32-float32-False-True-2-True-False] - AssertionError: Tensor-likes are not close!
# FAILED python/amd/test_tilelang_gemm_mfma_preshuffle.py::test_assert_tl_matmul[256-256-512-float8_e4m3fnuz-float32-float32-False-False-2-True-False] - AssertionError: Tensor-likes are not close!
# FAILED python/debug/test_tilelang_debug_print.py::test_debug_print_buffer_rocm_fp8 - RuntimeError: Compilation Failed! ['hipcc', '-std=c++17', '-fPIC', '--offload-arch=gfx950', '--shared', '/tmp/tmpo1gb83nj.cpp', '-I/root/tilelang/3rdparty/composable_kernel/include', '-I/root/tilelang/3rdparty/../src', '-o', '/tmp/tmpo1gb83nj.so']