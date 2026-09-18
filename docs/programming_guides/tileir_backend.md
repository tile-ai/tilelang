# Using and Tuning the TileIR Backend

The TileIR backend lowers TileLang programs to NVIDIA CUDA Tile IR and launches
the assembled cubin through the cuTile runtime.

## Install the toolchain

TileIR currently requires a supported NVIDIA GPU, the CUDA Tile IR 13.4 Python
bindings, `tileiras` 13.4, and the cuTile 1.5 runtime. Install the packaged
assembler and runtime dependencies with:

```bash
pip install 'tilelang[tileir]'
```

Build the CUDA Tile IR Python bindings from the matching public
`NVIDIA/cuda-tile` release. The
[TileIR backend dependency instructions](../compiler_internals/tileir_backend.md)
include the build command and environment setup.

Use `check_tileir_available()` to check the required components before compiling
a kernel:

```python
from tilelang.tileir.checks import check_tileir_available

check_tileir_available()
```

## Compile a kernel

Set the target to the GPU's compute capability and select the `tileir` execution
backend:

```python
import torch
import tilelang
from tilelang.tileir import language as T

n = 1 << 20


@T.prim_func
def vector_add(
    A: T.Tensor((n,), "float32"),
    B: T.Tensor((n,), "float32"),
    C: T.Tensor((n,), "float32"),
):
    with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
        for tx in T.Parallel(256):
            i = bx * 256 + tx
            if i < n:
                C[i] = A[i] + B[i]


major, minor = torch.cuda.get_device_capability()
kernel = tilelang.compile(
    vector_add,
    out_idx=[-1],
    target=f"tileir -arch=sm_{major}{minor}",
    execution_backend="tileir",
)
```

Benchmark the kernel without TileIR-specific hints first. Leaving a hint unset
uses the CUDA Tile IR toolchain default and gives the tuning run a useful
baseline.

## Choose a tuning space

Start with the program parameters that define the kernel's algorithm and tiling,
such as block sizes and thread count. Then try TileIR entry and copy hints around
the best program configurations. The example below uses a small staged search
to keep its runtime short; this is an example strategy, not a universal cuTile
recommendation. Use an exhaustive Cartesian product when the kernel and tuning
budget justify it.

### Program parameters

Program parameters change the input passed to TileIR and their useful ranges are
kernel-specific. They are separate from CUDA Tile IR `optimization_hints`.

The current TileIR lowering treats `T.Pipelined(..., num_stages=N)` as an
on/off pipeline marker: zero leaves the loop unmarked, while any positive value
creates the same token-chained pipelined loop. The numeric stage count is not
forwarded to CUDA Tile IR, so do not sweep several positive `num_stages` values.
It also does not set `T.copy(latency=N)`; copy latency is a separate hint.

`T.use_swizzle(...)` is currently treated as a scheduling hint and ignored by
the TileIR backend. The logical block coordinates remain unchanged, and the
CUDA Tile IR toolchain owns block scheduling.

### Entry hints

Import `from tilelang.tileir import language as T` to use the TileIR-specific
`T.Kernel` and `T.copy` keyword hints. The common and CUDA dialects retain their
own APIs; existing CUDA kernels without TileIR-specific keywords can still be
compiled with `execution_backend="tileir"` within the supported subset.

The current lowering accepts these entry-scoped hints:

| TileLang API | CUDA Tile IR hint | Valid values |
| --- | --- | --- |
| `T.Kernel(num_ctas=...)` | `num_cta_in_cga` | Unset, or a power of two in `[1, 16]` |
| `T.Kernel(occupancy=...)` | `occupancy` | Unset, or an integer in `[1, 32]` |
| `T.Kernel(num_worker_warps=...)` | `num_worker_warps_per_cta` | Unset, `4`, or `8` |

There is no architecture-independent best value for `num_ctas` or `occupancy`.
`num_worker_warps` is mainly useful for warp-specialized kernels with high
register pressure; `4` and `8` are its complete explicit value set.

`T.Kernel(tileir_hints=...)` packages the same hints into per-architecture
dictionaries. The values below demonstrate the dictionary shape; they are not
recommended settings for those architectures:

```python
tileir_hints = {
    "sm_100": {"num_cta_in_cga": 2, "occupancy": 2},
    "sm_120": {"num_cta_in_cga": 4, "num_worker_warps_per_cta": 8},
    "default": {"occupancy": 1},
}
```

The scalar arguments are convenient while tuning one architecture. Use
`tileir_hints` to package selected values for multiple SMs. The two forms cannot
be used together.

### Copy hints

The current lowering accepts two hints on global-memory `T.copy` operations:

| TileLang API | CUDA Tile IR hint | Valid values | Behavior |
| --- | --- | --- | --- |
| `T.copy(latency=...)` | `latency` | Unset, or an integer in `[1, 10]` | Unset lets the compiler estimate; larger values suggest deeper prefetching |
| `T.copy(disable_tma=...)` | `allow_tma=false` | `False` or `True` | `False` leaves the hint unset; `True` disallows TMA for that copy |

These hints apply only to the corresponding load or store, not to the kernel
entry. TileLang does not emit an explicit `allow_tma=true`; leaving
`disable_tma=False` uses the compiler default, which allows TMA when applicable.

CUDA Tile IR hints are advisory. The compiler may ignore a hint when it conflicts
with hardware constraints or a profitable schedule, and it does not have to
diagnose that decision. See the official
[cuTile Python performance tuning guide](https://docs.nvidia.com/cuda/cutile-python/performance.html)
for the underlying hint semantics and exhaustive-search API.

### Compiler directives

Each autotuner configuration may include a reserved `pass_configs` field. TileIR
uses the following directives:

| Pass config | Values | Notes |
| --- | --- | --- |
| `tl.tileir.opt_level` | Integers in `[0, 3]` | Defaults to `3`; lower levels are mainly useful for compiler diagnosis |
| `tl.enable_fast_math` | Boolean | Changes numerical semantics; enable only when the operation's tolerance permits it |
| `tl.disable_tma_lower` | Boolean | Deprecated global switch; prefer per-copy `disable_tma` |

For example, fast math can be enabled for one candidate without changing the
rest of the search:

```python
from tilelang.transform import PassConfigKey

config = {
    "block_size": 256,
    "pass_configs": {PassConfigKey.TL_ENABLE_FAST_MATH: True},
}
```

Config keys only affect the generated kernel when its factory forwards them to
`T.Kernel(...)` or `T.copy(...)`. Keep an unhinted candidate in the search;
explicit hints can be slower than the toolchain defaults.

### CUDA-specific parameters

The TileIR backend does not run TileLang's CUDA/PTX lowering pipeline. Do not
carry the following CUDA tuning parameters into a TileIR search:

| Parameter | TileIR behavior |
| --- | --- |
| `T.Pipelined(order=..., stage=..., group=...)` | Rejected; explicit pipeline schedules are not supported |
| `T.Pipelined(sync=...)` | Has no effect because the frontend does not currently encode this argument in the loop annotations |
| `T.copy(coalesced_width=..., eviction_policy=..., prefer_instruction=..., loop_layout=...)` | Not consumed by ordinary TileIR copy lowering; use `latency` and `disable_tma` instead |
| `T.annotate_layout(...)`, `T.annotate_l2_hit_ratio(...)`, `T.annotate_safe_value(...)` | Not consumed by the current TileIR lowering |
| `T.set_max_nreg(...)` and the producer/consumer register-allocation annotations | Accepted as scheduling hints but emit no TileIR operation |
| `T.annotate_min_blocks_per_sm(...)` | Not supported; use the TileIR `occupancy` entry hint instead |
| `T.Kernel(prelude=...)` and arbitrary CUDA C/PTX extern helpers | Not a TileIR escape hatch; injected CUDA source is not compiled and unsupported extern calls are rejected |
| `T.ClusterKernel(cluster_dims=...)` | `cluster_dims` is not consumed by the TileIR launch path; use `T.Kernel(num_ctas=...)` for the CGA entry hint |
| CUDA/PTX pass configs and device `compile_flags` | Not consumed by TileIR codegen; examples include register-usage, async-copy, LDG/STG, WGMMA, vectorization, and warp-specialization controls |

The CUDA-only APIs in this table are available when importing the CUDA
dialect; the TileIR dialect does not re-export them.

An ignored parameter should not be left in the search space: candidates that
differ only by that value compile to the same TileIR and make tuning results
misleading.

## Run the autotuning example

The repository includes a tiled-divide example that tunes `block_size` first,
then searches the applicable TileIR entry hints, copy hints, and fast math:

```bash
python examples/tileir/example_tileir_autotune.py
```

Every candidate is checked against the same PyTorch reference. The script prints
the candidate spaces, selected configuration, latency, and speedup over the
unhinted baseline. Its candidate lists are intentionally small and illustrative;
they are not recommended values for every kernel or GPU. Use it as a template
for exposing kernel-specific parameters from another kernel factory.

## Troubleshooting

- **Dependency or version error:** run `check_tileir_available()` and follow the
  component-specific error message.
- **Target architecture error:** pass the GPU's base SM, for example
  `tileir -arch=sm_90`, `sm_100`, or `sm_120`.
- **Unsupported semantic construct:** simplify the kernel or report the first
  unsupported construct from the exception.
- **Some candidates fail:** remove configurations that exceed the kernel or
  GPU's resource and hint constraints.
- **All explicit hints are slower:** keep the unhinted configuration.
