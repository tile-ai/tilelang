# Performance Analyzer

`tilelang.tools.Analyzer` inspects TileLang TIR and estimates its floating-point
work, global-memory traffic, and roofline execution time. It is useful for
checking a kernel's expected arithmetic cost before compiling and benchmarking
it. The result is a static estimate, not a measurement from the GPU.

## Requirements

The analyzer and its NumPy dependency are included in the standard TileLang
installation. No optional package is required.

The device model passed to the analyzer supplies architecture and bandwidth
information. For example, constructing `CUDA("cuda")` queries CUDA device 0, so
that workflow requires a working CUDA runtime and a visible GPU even though the
analyzer does not execute the kernel.

## Quick Start

Create specialized TIR with `get_tir()`, then analyze the resulting `PrimFunc`:

```python
import tilelang
import tilelang.language as T
from tilelang.carver.arch import CUDA
from tilelang.tools import Analyzer

M = N = K = 1024


@tilelang.jit
def matmul(A, B, block_M, block_N, block_K):
    A: T.Tensor((M, K), T.float16)
    B: T.Tensor((N, K), T.float16)
    C = T.empty((M, N), T.float16)

    with T.Kernel(
        T.ceildiv(N, block_N),
        T.ceildiv(M, block_M),
        threads=128,
    ) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), T.float16)
        B_shared = T.alloc_shared((block_N, block_K), T.float16)
        C_local = T.alloc_fragment((block_M, block_N), T.float32)

        T.clear(C_local)
        for k in T.serial(T.ceildiv(K, block_K)):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[bx * block_N, k * block_K], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


tir = matmul.get_tir(block_M=128, block_N=128, block_K=32)
device = CUDA("cuda")
result = Analyzer.analysis(tir, device)

print(f"FLOPs: {result.total_flops}")
print(f"Global bytes: {result.total_global_bytes}")
print(f"Estimated seconds: {result.estimated_time}")
print(f"Model peak TFLOPS: {result.expected_tflops}")
print(f"Model bandwidth (GB/s): {result.expected_bandwidth_GBps}")
```

For this tiled GEMM, `total_flops` should equal `2 * M * N * K`. Memory traffic
is derived from the `T.copy` regions that cross the global-buffer boundary.

## API and Result

The main entry point is:

```python
result = Analyzer.analysis(fn, device)
```

- `fn` is a `tvm.IRModule` or `tvm.tirx.PrimFunc` containing TileLang TIR.
- `device` is a TileLang device description such as
  `tilelang.carver.arch.CUDA("cuda")`.
- The method returns an immutable `AnalysisResult`.

`AnalysisResult` exposes these fields:

| Field | Meaning |
| --- | --- |
| `total_flops` | FLOPs attributed to recognized `T.gemm` calls. |
| `total_global_bytes` | Bytes attributed to recognized `T.copy` calls with a global source or destination. |
| `estimated_time` | Maximum of modeled compute time and memory time when a compute model is available; otherwise memory time. |
| `expected_tflops` | The theoretical peak selected by the built-in compute-capability table, or `None` for an unsupported capability. |
| `expected_bandwidth_GBps` | The bandwidth value obtained from the device model and converted to GB/s. |

The analyzer multiplies each recognized operation by its enclosing non-thread-
bound TIR loop extents and the `blockIdx.x` and `blockIdx.y` grid extents. Its
compute-time model currently has entries for NVIDIA compute capabilities 8.0,
8.6, and 8.9.

## Limitations

- This is a simple roofline estimate. It does not measure latency, occupancy,
  instruction issue, cache behavior, memory coalescing, bank conflicts, or
  pipeline overlap.
- FLOP counting recognizes `T.gemm` calls only. Elementwise arithmetic and
  other intrinsics are not included.
- Memory counting recognizes `T.copy` regions that touch a function parameter
  buffer. Direct buffer loads and stores and other memory intrinsics are not
  included.
- Grid scaling tracks `blockIdx.x` and `blockIdx.y`; a `blockIdx.z` extent is
  not part of the current calculation.
- The built-in peak-TFLOPS table uses fixed architecture values. Treat
  `expected_tflops` and `estimated_time` as comparison aids, not device-specific
  benchmark results.

## Examples

- [GEMM analysis](https://github.com/tile-ai/tilelang/blob/main/examples/analyze/example_gemm_analyze.py)
- [Convolution expressed through tiled GEMM](https://github.com/tile-ai/tilelang/blob/main/examples/analyze/example_conv_analyze.py)
