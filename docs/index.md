# 👋 Welcome to Tile Language

[GitHub](https://github.com/tile-ai/tilelang)

Tile Language (tile-lang) is a concise domain-specific language designed to streamline
the development of high-performance GPU/CPU kernels (e.g., GEMM, Dequant GEMM, FlashAttention, LinearAttention).
By employing a Pythonic syntax with an underlying compiler infrastructure on top of TVM,
tile-lang allows developers to focus on productivity without sacrificing the
low-level optimizations necessary for state-of-the-art performance.

:::{toctree}
:maxdepth: 2
:caption: GET STARTED

get_started/Installation
get_started/overview
get_started/targets
:::

:::{toctree}
:maxdepth: 1
:caption: TUTORIALS

tutorials/debug_tools_for_tilelang
tutorials/auto_tuning
tutorials/logging
:::

:::{toctree}
:maxdepth: 1
:caption: PROGRAMMING GUIDES

programming_guides/overview
programming_guides/language_basics
programming_guides/instructions
programming_guides/control_flow
programming_guides/python_compatibility
programming_guides/autotuning
programming_guides/type_system
:::

:::{toctree}
:maxdepth: 1
:caption: DEEP LEARNING OPERATORS

deeplearning_operators/elementwise
deeplearning_operators/gemv
deeplearning_operators/matmul
deeplearning_operators/matmul_sparse
deeplearning_operators/deepseek_mla
:::

:::{toctree}
:maxdepth: 1
:caption: COMPILER INTERNALS

compiler_internals/letstmt_inline
compiler_internals/inject_fence_proxy
compiler_internals/tensor_checks
:::

:::{toctree}
:maxdepth: 1
:caption: API Reference

autoapi/tilelang/index
:::

:::{toctree}
:maxdepth: 1
:caption: Privacy

privacy
:::
