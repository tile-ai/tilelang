<img src=./images/logo-row.svg />

<div align="center">

# Tile Language
[![PyPI version](https://badge.fury.io/py/tilelang.svg)](https://badge.fury.io/py/tilelang)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tile-ai/tilelang)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/TUrHyJnKPG)
[![Puzzles](https://img.shields.io/badge/🧩_Learn-TileLang_Puzzles-blueviolet)](https://github.com/tile-ai/tilelang-puzzles)
</div>

Tile Language (**tile-lang**) is a concise domain-specific language designed to streamline the development of high-performance GPU/CPU kernels (e.g., GEMM, Dequant GEMM, FlashAttention, LinearAttention). By employing a Pythonic syntax with an underlying compiler infrastructure on top of [TVM](https://tvm.apache.org/), tile-lang allows developers to focus on productivity without sacrificing the low-level optimizations necessary for state-of-the-art performance.

<img src=./images/MatmulExample.png />

## Latest News
- 02/02/2026 🧩: Check out [TileLang Puzzles](https://github.com/tile-ai/tilelang-puzzles), a fun and interactive way to learn TileLang programming with 10 progressively harder puzzles!
- 12/18/2025 🚀: Added [CuTeDSL backend](https://github.com/tile-ai/tilelang/pull/1421) support, enabling compilation to NVIDIA CUTLASS CuTe DSL! Join us in building and optimizing this exciting new backend: [Issue #1454](https://github.com/tile-ai/tilelang/issues/1454).
- 12/17/2025 🔬: Integrated [Z3 theorem prover](https://github.com/tile-ai/tilelang/pull/1367) into TVM Arith Analyzer, bringing SMT-based symbolic reasoning for enhanced optimizations and automatic correctness verification!
- 10/31/2025 🔧: Migrated to [apache-tvm-ffi](https://github.com/tile-ai/tilelang/pull/1108), significantly reducing CPU overhead!
- 10/30/2025 📦: We have released v0.1.6.post2, which is the last version compatible with Python 3.8.
- 10/07/2025 🍎: Added Apple Metal Device support, check out [Pull Request #799](https://github.com/tile-ai/tilelang/pull/799) for details.
- 09/29/2025  🎉: Thrilled to announce that ​​AscendC​​ and ​Ascend​NPU IR​​ backends targeting Huawei Ascend chips are now supported!
Check out the preview here:
🔗 [link](https://github.com/tile-ai/tilelang-ascend).
This includes implementations across two branches:
[ascendc_pto](https://github.com/tile-ai/tilelang-ascend) and
[npuir](https://github.com/tile-ai/tilelang-ascend/tree/npuir).
Feel free to explore and share your feedback!
- 07/04/2025 🚀: Introduced `T.gemm_sp` for 2:4 sparse tensor core support, check out [Pull Request #526](https://github.com/tile-ai/tilelang/pull/526) for details.
- 06/05/2025 ✨: Added [NVRTC Backend](https://github.com/tile-ai/tilelang/pull/461) to significantly reduce compilation time for cute templates!
- 04/14/2025 🚀: Added high-performance FlashMLA implementation for AMD MI300X, achieving performance parity with hand-optimized assembly kernels of Aiter! See [example_mla_amd](./examples/deepseek_mla/amd/README.md) for details.
- 03/03/2025 🚀: Added high-performance MLA Decoding support using only 80 lines of Python code, achieving performance on par with FlashMLA on H100 (see [example_mla_decode.py](./examples/deepseek_mla/example_mla_decode.py))! We also provide [documentation](./examples/deepseek_mla/README.md) explaining how TileLang achieves this.
- 02/15/2025 ✨: Added WebGPU Codegen support, see [Pull Request #86](https://github.com/tile-ai/tilelang/pull/86)!
- 02/12/2025 ✨: Excited to announce the release of [v0.1.0](https://github.com/tile-ai/tilelang/releases/tag/v0.1.0)!
- 02/10/2025 🚀: Added debug tools for TileLang—`T.print` for printing variables/buffers ([docs](https://tilelang.com/tutorials/debug_tools_for_tilelang.html)) and a memory layout plotter ([examples/plot_layout](./examples/plot_layout)).
- 01/20/2025 ✨: We are excited to announce that tile-lang, a dsl for high performance AI workloads, is now open source and available to the public!

## Tested Devices
Although tile-lang aims to be portable across a range of Devices, it has been specifically tested and validated on the following devices: for NVIDIA GPUs, this includes the H100 (with Auto TMA/WGMMA support), A100, V100, RTX 4090, RTX 3090, and RTX A6000; for AMD GPUs, it includes the MI250 (with Auto MatrixCore support) and the MI300X (with Async Copy support).

## OP Implementation Examples
**tile-lang** provides the building blocks to implement a wide variety of operators. Some examples include:

- [Matrix Multiplication](./examples/gemm/)
- [Dequantization GEMM](./examples/dequantize_gemm/)
- [Flash Attention](./examples/flash_attention/)
- [Flash Linear Attention](./examples/linear_attention/)
- [Flash MLA Decoding](./examples/deepseek_mla/)
- [Native Sparse Attention](./examples/deepseek_nsa/)

Within the `examples` directory, you will also find additional complex kernels—such as convolutions, forward/backward passes for FlashAttention, more operators will continuously be added.

## Benchmark Summary

TileLang achieves exceptional performance across a variety of computational patterns. Comprehensive benchmark scripts and settings are available at [tilelang-benchmark](https://github.com/tile-ai/tilelang-benchmark). Below are selected results showcasing its capabilities:

- MLA Decoding Performance on H100

  <div style="display: flex; gap: 10px; justify-content: center;">
    <div style="flex: 1;">
      <img src="./examples/deepseek_mla/figures/bs64_float16.png" alt="mla decode performance bs64 on H100" width="100%" />
    </div>
    <div style="flex: 1;">
      <img src="./examples/deepseek_mla/figures/bs128_float16.png" alt="mla decode performance bs128 on H100" width="100%" />
    </div>
  </div>

- Flash Attention Performance on H100

  <div align="center">    <img src="./images/mha_performance_h100.png" alt="operator performance on H100" width=80% />
  </div>

- Matmul Performance on GPUs (RTX 4090, A100, H100, MI300X)

  <div>
    <img src="./images/op_benchmark_consistent_gemm_fp16.png" alt="gemm fp16 performance on Gpus" />
  </div>

- Dequantize Matmul Performance on A100

  <div>
    <img src="./images/op_benchmark_a100_wq_gemv.png" alt="dequantize gemv performance on A100" />
  </div>

## Installation
### Method 1: Install with Pip

The quickest way to get started is to install the latest release from PyPI:

```bash
pip install tilelang
```

Alternatively, you can install directly from the GitHub repository:

```bash
pip install git+https://github.com/tile-ai/tilelang
```

Or install locally:

```bash
# install required system dependencies
sudo apt-get update
sudo apt-get install -y python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

pip install -e . -v # remove -e option if you don't want to install in editable mode, -v for verbose output
```

### Method 2: Build from Source
We currently provide three ways to install **tile-lang** from source:
- [Install from Source (using your own TVM installation)](./docs/get_started/Installation.md#method-1-install-from-source-using-your-own-tvm-installation)
- [Install from Source (using the bundled TVM submodule)](./docs/get_started/Installation.md#method-2-install-from-source-using-the-bundled-tvm-submodule)
- [Install Using the Provided Script](./docs/get_started/Installation.md#method-3-install-using-the-provided-script)

### Method 3: Install with Nightly Version

For users who want access to the latest features and improvements before official releases, we provide nightly builds of **tile-lang**.

```bash
pip install tilelang -f https://tile-ai.github.io/whl/nightly
# or pip install tilelang --find-links https://tile-ai.github.io/whl/nightly
```

> **Note:** Nightly builds contain the most recent code changes but may be less stable than official releases. They're ideal for testing new features or if you need a specific bugfix that hasn't been released yet.

## Quick Start

In this section, you'll learn how to write and execute a straightforward GEMM (matrix multiplication) kernel using tile-lang, followed by techniques for layout optimizations, pipelining, and L2-cache–friendly swizzling.

### GEMM Example with Annotations (Layout, L2 Cache Swizzling, and Pipelining, etc.)

Below is an example that demonstrates more advanced features: layout annotation, parallelized copy, and swizzle for improved L2 cache locality. This snippet shows how to adapt your kernel to maximize performance on complex hardware.

```python
# @tilelang.jit(target="cuda")
# target currently can be "cuda" or "hip" or "cpu".
# if not specified, it will be inferred from the input tensors during compile time
@tilelang.jit
def matmul_relu(
    A, B,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 64,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    # declare compilation shape constant
    M, N, K = T.const('M, N, K')

    # annotate input tensor shape
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]

    # allocate output tensor
    C = T.empty([M, N], dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        # Enable rasterization for better L2 cache locality (Optional)
        # T.use_swizzle(panel_size=10, enable=True)

        # Clear local accumulation
        T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            # Copy tile of A
            # This is a sugar syntax for parallelized copy
            T.copy(A[by * block_M, ko * block_K], A_shared)

            # Copy tile of B
            T.copy(B[ko * block_K, bx * block_N], B_shared)

            # Perform a tile-level GEMM on the shared buffers
            # Currently we dispatch to the cute/hip on Nvidia/AMD GPUs
            T.gemm(A_shared, B_shared, C_local)

        # relu
        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)

        # Copy result back to global memory
        T.copy(C_local, C[by * block_M, bx * block_N])

    # You can write multiple cuda kernel in one function, they execute sequentially
    # with T.Kernel(...) as ...

    # Return the tensor, you can also return multiple tensors
    return C


M, N, K = 1024, 1024, 1024

a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c_ref = torch.relu(a @ b)

# Call the kernel
c = matmul_relu(a, b)
torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)

# Call the kernel with overwritten compilation constants
c = matmul_relu(a, b, block_M=128, block_N=128, block_K=64)
torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)

# Retrieve the compiled kernel
kernel = matmul_relu.compile(a, b) # use torch.Tensor
kernel = matmul_relu.compile(      # use T.Tensor as placeholder
  T.Tensor((M, K), T.float16),
  T.Tensor((K, N), T.float16)
)
kernel = matmul_relu.compile(      # directly specify the shape constants
  M=M, N=N, K=K,
  block_M=128, block_N=128, block_K=64
)
print(kernel.get_kernel_source())
c = kernel(a, b)

# 5.Profile latency with kernel
profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
latency = profiler.do_bench()
print(f"Latency: {latency} ms")
```

### Dive Deep into TileLang Beyond GEMM

In addition to GEMM, we provide a variety of examples to showcase the versatility and power of TileLang, including:

- [Dequantize GEMM](./examples/dequantize_gemm/): Achieve high-performance dequantization by **fine-grained control over per-thread operations**, with many features now adopted as default behaviors in [BitBLAS](https://github.com/microsoft/BitBLAS), which utilizing magic layout transformation and intrins to accelerate dequantize gemm.
- [FlashAttention](./examples/flash_attention/): Enable cross-operator fusion with simple and intuitive syntax, and we also provide an example of auto tuning.
- [LinearAttention](./examples/linear_attention/): Examples include RetNet and Mamba implementations.
- [Convolution](./examples/convolution/): Implementations of Convolution with IM2Col.

## Upcoming Features

Check our [tilelang v0.2.0 release plan](https://github.com/tile-ai/tilelang/issues/79) for upcoming features.

---

TileLang has now been used in project [BitBLAS](https://github.com/microsoft/BitBLAS) and [AttentionEngine](https://github.com/microsoft/AttentionEngine).

## Join the Discussion

Welcome to join our Discord community for discussions, support, and collaboration!

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20Us-blue?logo=discord&style=for-the-badge)](https://discord.gg/TUrHyJnKPG)

## Acknowledgments

We would like to express our gratitude to the [TVM](https://github.com/apache/tvm) community for their invaluable contributions. The initial version of this project was mainly developed by [LeiWang1999](https://github.com/LeiWang1999), [chengyupku](https://github.com/chengyupku) and [nox-410](https://github.com/nox-410) with supervision from Prof. [Zhi Yang](https://yangzhihome.github.io) at Peking University. Part of this work was carried out during an internship at Microsoft Research, where Dr. Lingxiao Ma, Dr. Yuqing Xia, Dr. Jilong Xue, and Dr. Fan Yang offered valuable advice and support. We deeply appreciate their mentorship and contributions.
