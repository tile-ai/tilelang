# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TileLang is a domain-specific language (DSL) for developing high-performance GPU/CPU kernels (GEMM, FlashAttention, etc.) with Pythonic syntax. Built on Apache TVM, it supports NVIDIA CUDA, AMD HIP, Apple Metal, and CPU backends.

## Build & Development Commands

```bash
# Install in editable mode (requires system deps: cmake, gcc, libtinfo-dev, zlib1g-dev, libedit-dev, libxml2-dev)
pip install -e . -v

# Alternative: Manual build via PYTHONPATH (faster for development)
mkdir -p build && cd build
cmake .. -DUSE_CUDA=ON  # -DUSE_ROCM=ON -DUSE_METAL=ON for other backends
make -j
export PYTHONPATH=/path/to/tilelang:$PYTHONPATH

# Run tests
python -m pytest testing/
python -m pytest testing/python/kernel/ -xvs  # specific test directory
python -m pytest testing/python/kernel/test_tilelang_kernel_gemm.py -xvs --timeout=300  # single test file

# Lint and format
pre-commit run --all-files
ruff format .
ruff check . --fix
```

## Architecture

### Compilation Pipeline

```
Python DSL (@tilelang.jit decorated functions)
    ↓
Language Parser (tilelang/language/parser/)
    ↓
TVM TIR-based IR
    ↓
Compiler Passes (tilelang/transform/, src/transform/)
    ↓
JIT Compilation (tilelang/jit/) with backend adapters
    ↓
Target Code Generation (src/tl_templates/cuda|hip|metal|cpu/)
    ↓
Kernel Execution via TVM FFI/Cython/NVRTC backends
```

### Key Components

- **tilelang/language/**: DSL primitives - `T.Kernel`, `T.alloc_shared`, `T.alloc_fragment`, `T.copy`, `T.gemm`, `T.Parallel`, `T.Pipelined`
- **tilelang/jit/**: JIT compilation with multiple backend adapters (TVM FFI default, Cython, NVRTC)
- **tilelang/engine/**: Kernel lowering from DSL to target code
- **tilelang/layout/**: Memory layout inference and swizzling for cache optimization
- **src/op/**: C++ operator implementations (GEMM, copy, atomic ops)
- **src/layout/**: C++ layout engine with Tensor Core-aware optimizations
- **src/tl_templates/**: Backend-specific code generation templates

### JIT Modes

- **Eager Mode**: Immediate compilation and execution
- **Lazy Mode**: Deferred compilation for optimization
- **Auto Mode**: Backend selection based on tensor type (torch, cupy, numpy)

## DSL Pattern

```python
import tilelang
import tilelang.language as T

@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float16), B: T.Tensor((K, N), T.float16), C: T.Tensor((M, N), T.float16)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return kernel
```

## Test Organization

- `testing/python/kernel/`: Full kernel execution tests (GEMM, GEMV, attention)
- `testing/python/language/`: DSL feature tests (allocation, control flow, intrinsics)
- `testing/python/layout/`: Layout inference verification
- `testing/python/transform/`: Compiler pass tests
- `testing/python/tilelibrary/`: High-level library tests
- `testing/python/autotune/`: Auto-tuner functionality

## Code Style

- Python: Ruff (line length 140, target Python 3.9+)
- C++: clang-format
- Pre-commit hooks enforce formatting on commit
