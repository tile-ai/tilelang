# MLX execution and in-place state updates

This optional integration for [issue #3222](https://github.com/tile-ai/tilelang/issues/3222)
executes TileLang-generated Metal kernels as native MLX primitives. It reuses
TileLang's Metal lowering and complete generated shader; no MSL signature
rewriting or PyTorch tensor conversion is involved. MLX manages lazy execution,
stream ordering, memory lifetime, and shader compilation.

## Install and run

In a TileLang source development environment on Apple Silicon running macOS 26
or later, install the optional native runtime with Xcode Command Line Tools available:

```bash
python -m pip install ./tilelang/contrib/mlx_runtime
python examples/metal/mlx_cache_update.py
python -m pytest testing/python/metal/test_metal_mlx.py -q
```

The runtime pins **MLX 0.32.4 and nanobind 3.1.0**. It uses MLX's native primitive
and Metal encoder APIs, whose ABI is version-specific. The Python adapter checks
the MLX version before importing the extension. Updating MLX requires updating
and rebuilding this integration, not just changing the dependency pin.

## Calling convention

```python
from tilelang.contrib.mlx import compile as compile_mlx

# Ordinary output: third parameter is allocated, omitted from call arguments.
kernel = compile_mlx(prim_func, out_idx=2)
output = kernel(input_a, input_b)

# State update: first parameter remains an input and returns an alias.
update = compile_mlx(state_func, inplace_idx=0)
state = update(state, rows, slot_ids)
```

Indices refer to PrimFunc parameters; negative indices are accepted. `out_idx`
and `inplace_idx` must be distinct and disjoint. Inputs follow PrimFunc order
with only `out_idx` omitted. Results are fresh outputs in `out_idx` order,
followed by state aliases in `inplace_idx` order. One result returns an array;
multiple results return a tuple. For example, with `out_idx=1, inplace_idx=0`,
`result, state = kernel(state)` returns a fresh result and the updated state.

Always **rebind state to the returned alias**. It shares the original allocation
and carries the dependency on the write. Keeping or reading a stale alias in a
concurrent computation does not provide ordering. Treat state as exclusively
owned mutable storage; do not branch independent writes from an old alias.
Use `mx.eval(state)` when host completion is needed, or pass the returned state
to subsequent MLX operations. `stream=` accepts an MLX stream or device; the
current default GPU stream is used otherwise.

This follows vLLM-metal's
[GDN primitive](https://github.com/vllm-project/vllm-metal/blob/2b1e76aec530ba59e3f89dbdcbd9d6b9909a0471/vllm_metal/metal/gdn.cpp):
`copy_shared_buffer` preserves state storage and `set_output_array` registers the
write with the MLX encoder. Ordinary outputs get independent allocations.

## Layout and supported scope

- Static positive shapes, static launch dimensions, and one kernel launch.
- FP16, FP32, int32, and uint32 tensor parameters. Runtime values such as slot IDs
  can be passed in tensors; Python scalar parameters must be specialized.
- Read-only inputs are made contiguous when necessary.
- In-place inputs are **never materialized**. Their actual strides must match
  the declared buffer strides, except singleton axes whose stride does not
  affect addressing. `T.StridedTensor` supports static non-overlapping
  row-major state layouts, including padded pages. A contiguous buffer
  declaration rejects a transposed state view instead of silently updating a
  copy. Runtime checks conservatively reject overlapping input ranges when one
  is writable.
- State views may have a nonzero storage offset; all kernel accesses are relative
  to that view. Vectorization is disabled for this adapter so arbitrary element
  offsets do not inherit wider vector alignment assumptions.
- Fresh outputs must be fully initialized by the kernel. In-place outputs retain
  elements not written by the kernel. The kernel author must guard invalid slot
  IDs and guarantee distinct destinations when scattering concurrently.
- Ordinary Metal SIMD-group operations, including shared-memory GEMM, are
  supported. The adapter deliberately selects the non-Metal-4 target.

Dynamic shapes, multiple launches, autodiff, `vmap`, and TileLang's persistent
cache/export/profiler APIs are not implemented. This is an explicit
`tilelang.contrib.mlx.compile` API, not a registered `execution_backend="mlx"`.
Rebinding state inside `mx.compile` is covered by tests; arbitrary effectful
program transformations are not claimed to be safe.

## Validation and relation to vLLM-metal

Tests run on Apple M4 with MLX 0.32.4. Besides numerical results and GEMM, they
exercise lazy chains, input lifetime, cross-stream writes and consumers,
compiled repeated updates with distinct outputs, multiple disjoint state aliases,
singleton axes, invalid contracts, and
preservation of original backing storage.

The scatter cases are informed by vLLM-metal's
[GDN tests](https://github.com/vllm-project/vllm-metal/blob/2b1e76aec530ba59e3f89dbdcbd9d6b9909a0471/tests/test_gdn_lazy_kernels.py):
invalid slots, untouched rows, offset views, and two padded pages with three live
tokens per page at widths 5 and 12. They compare the entire backing allocation,
including padding, against an independent CPU reference.

These are adapter and state-scatter regression tests, not a port of the full
GDN recurrence or a vLLM serving benchmark. End-to-end vLLM-metal integration,
performance, and a public JIT-backend contract remain separate work.

The native wheel was also installed and exercised in a separate virtual
environment with MLX, outside the TileLang checkout. That smoke test runs an
exported TileLang-generated shader for both ordinary output and in-place state
updates, checking that the extension loads from the installed wheel.
