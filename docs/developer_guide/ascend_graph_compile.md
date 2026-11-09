# Ascend Torch graph compilation

The experimental `tilelang_ascend` Dynamo backend compiles supported inference
partitions through FX, Relax, scalar TIR, and Ascend TileLang scheduling. Torch
owns GM tensors between kernels. Kernel compilation and execution reuse either
the AscendC compiler and `tvm_ffi` adapter, or PTODSL code generation and the
existing `pto` adapter. Both paths use Torch's current NPU stream.

This implementation independently follows the graph compilation approach in
[PR #2763](https://github.com/tile-ai/tilelang/pull/2763), using the
`497bd7fb9fc7be6e742b504c34446e1580005398` snapshot as a reference. It uses the
repository's current TVM `FuseOps` and `FuseTIR` and current backend compilation
interfaces and does not depend on that PR being merged. It does not require
the PR's copied primitive-function lowering,
GPU templates, scheduled-block canonicalizer, or Relax VM runtime.

## Usage

```python
import torch
import tilelang.graph  # Explicitly register the experimental Dynamo backend.


def model(x, bias):
    return torch.relu(x + bias) * bias


compiled = torch.compile(model, backend="tilelang_ascend")
x = torch.randn(7, 65, device="npu", dtype=torch.float32)
bias = torch.randn(65, device="npu", dtype=torch.float32)
with torch.no_grad():
    result = compiled(x, bias)
    torch.testing.assert_close(result, model(x, bias))
```

Importing `tilelang` alone does not register Dynamo backends. Backend options
are captured by an immutable `tilelang.graph.GraphConfig`; pass fields through
`torch.compile(options=...)` or `tilelang.graph.make_backend(config)`.

The default compilation target is `{"kind": "ascend", "mcpu": "dav-3510"}` and
`execution_backend` defaults to `"auto"`. Select an explicitly supported target
for the machine running the graph. The same `tilelang_ascend` graph backend
supports both AscendC and PTO:

```python
compiled = torch.compile(
    model,
    backend="tilelang_ascend",
    options={"target": "pto", "execution_backend": "pto"},
)
```

Target resolution selects the native `ascend` target kind with a backend
variant. Plain AscendC with `"auto"` resolves to `"tvm_ffi"`; `target="pto"`
with `"auto"` resolves to the `pto` variant and adapter. Explicit
`target="ascend", execution_backend="pto"` also selects PTO, preserving target
attributes such as `mcpu`. To keep an explicit architecture while selecting PTO,
pass a JSON target string and `execution_backend="pto"` through graph options.
`target="pto", execution_backend="tvm_ffi"` is incompatible and is rejected.

PTO kernel compilation requires PTODSL, `ptoas`, and the compatible Bisheng/CANN
toolchain used by the existing PTO adapter. Source-only PTO generation is a
separate check: emitted PTODSL does not prove successful assembly, binary
compilation, numerical execution, or performance.

## Execution and fallback

The frontend partitions supported pure operators and converts each partition
to Relax `call_tir` functions. Compilation occurs on its first invocation with
real tensors, rather than during Dynamo's FakeTensor capture. Each kernel is
scheduled and compiled before the executor is installed. The executor allocates
fresh outputs using Torch and passes Torch tensors directly to the existing
adapter, preserving the selected adapter's current NPU stream handling.

Initial inputs are positive static shapes, contiguous with zero storage offset,
on one NPU, and use fp16, bf16, or fp32. Autograd inputs, mutation, and unsafe
aliasing are outside the compiled capability contract. Unsupported partitions
execute their original Torch NPU graph. Effects that cannot safely be
partitioned cause the whole captured graph to remain in Torch. `strict=True`
reports unsupported or failed partition compilation instead of silently using
that fallback; it is useful for integration tests.

Kernel execution errors propagate. The backend does not retry after a device
launch or synchronization failure. Numerical validation failures can select
Torch; a failed fusion candidate can select the independently compiled
unfused baseline.

The callable returned by `make_backend(config)(fx_graph, example_inputs)` has a
`graph_report`. Each partition reports whether compilation succeeded, its
kernel call count, compiled and fallback call counts, and its fallback reason.
A passing Torch output alone is not proof that a partition used TileLang or
that PTO was selected. Inspect the resolved backend, execution adapter, and
actual compiled calls when validating a target configuration.

## Scheduling and fusion

The injective schedule inlines pure producer blocks before constructing a
single-output kernel. It maps output coordinates to a one-dimensional core
grid and a local `SimtVF` thread domain, with guards for tile and tensor tails.
Inlined scalar intermediates have no GM allocations. Broadcast reads are
checked against their input regions. When aligned, contiguous four-element
chunks can be proved, the scheduler vectorizes GM transfers through private
per-thread local arrays while preserving scalar rounding and NaN semantics.
`vector_width=0` chooses this automatically; `vector_width=1` retains scalar
transfers. Unaligned tails and unsupported index maps use scalar transfers.
The implementation rejects input writes,
unproven read bounds, unsupported scalar effects, and unsupported output maps.

The initial operator set includes floating add/subtract (`alpha=1`), multiply,
divide, ReLU, exp/sqrt/rsqrt, square, casts, and static singleton broadcasting.
Reductions cover the complete last axis with at most 16384 elements. Softmax
is decomposed into separate max, subtract, exp, sum, and divide kernels with
fp32 statistics. RMSNorm arithmetic with explicit fp32 statistics can use the
same baseline. A multi-reduction fusion candidate may remain unsupported.

Two-dimensional fp16/bf16 GEMM requires M/N/K divisible by 16 and fp32
accumulation. Logical NN inputs are explicitly converted to the NT L1 input
layout. Supported local bias/add/ReLU/cast epilogues use an L0C-to-UB handoff
inside one mixed kernel. General Vector-to-Cube and Cube-to-Cube fusion,
cross-core reductions, and dynamic/strided input layouts use fallback.

Reduction and GEMM schedules have separate capability checks. The target's
existing pipeline remains responsible for layouts, task scheduling, core
resolution, synchronization, multibuffering, and code generation. Graph
grouping happens before VF construction, so a generic block-flattening pass
never needs to erase or rebuild an existing VF boundary.

Fusion is off by default. Enable measured selection with:

```python
compiled = torch.compile(
    model,
    backend="tilelang_ascend",
    options={"fusion": "profile", "min_speedup": 0.05, "max_regression": 0.02},
)
```

The candidate is built using Relax `FuseOps` and `FuseTIR`. The Ascend schedule
must still accept every resulting PrimFunc. Fusing Relax functions alone does
not establish that an intermediate is on chip. Unsupported groups retain the
baseline; general cross-core redistribution, Vector-to-Cube, and Cube-to-Cube
fusion are not enabled by merely grouping their TIR.

Before selection, both runners are checked against the original Torch
partition. Tensor structure, dtype, shape, device, NaN/Inf behavior, and values
are checked. Initial `rtol/atol` pairs are:

| dtype | Injective | Reduction/GEMM |
| --- | --- | --- |
| fp32 | `1e-4 / 1e-5` | `1e-4 / 1e-4` |
| fp16 | `1e-3 / 1e-3` | `1e-2 / 1e-2` |
| bf16 | `1e-2 / 1e-2` | `2e-2 / 2e-2` |

Fusion selection alternates the measurement order across at least three
complete rounds. Device measurements use an explicit NPU device and the
configured profiler; graph wall measurements include allocation and dispatch.
Missing or non-finite samples, inconsistent improvement, or excessive
variability retain the baseline. The default policy requires at least 5%
device latency reduction and at most 2% graph wall regression. These are
selection policies, not a claim that every candidate improves performance.

The selected callable carries a `fusion_decision` with raw samples and the
acceptance or rejection reason. A missing profiler result is distinct from a
device execution failure.

## Verification and attribution

Hardware-independent coverage exercises FX/Relax conversion, graph execution
with injected CPU kernel implementations, capability failures, source-only
AscendC/PTO lowering, output validation, and fusion measurement policy. Hardware
tests must additionally assert actual compilation, numerical correctness,
non-default stream ordering, and output validity across repeated calls.

For performance attribution, compare:

1. A: independent kernels with the default tile.
2. B: independent kernels with the candidate scheduling configuration.
3. C: an on-chip fusion candidate using B's requested configuration.

Isolating fusion requires the same effective tile shapes, tile origins,
block/core mapping, and VF execution geometry in B and C. Equal configuration
values do not prove those properties; inspect the scheduled TIR for every
producer and consumer.

In the example's elementwise workload, B and C use the same linear output tiles
and ownership mapping, so B to C isolates fusion, including reduced launches.
A to B measures the change in unfused scheduling from the requested tile size.

The current GEMM comparison includes an additional scheduling change. B runs
separate cast, bias, and ReLU kernels using the injective scheduler's linear
`tile_size` (2048 by default for the candidate). C executes those operations on
the Cube output's two-dimensional tile, normally 64 by 64, with smaller aligned
tiles where required. Their effective epilogue regions and core mapping differ
even when both runners use the same `GraphConfig`; the GEMM rule does not use
the injective `tile_size` to choose its Cube tile. B to C therefore measures
combined fusion and epilogue scheduling benefits. The example marks this
comparison with `fusion_isolated=false`, reports
`combined_epilogue_speedup_B_over_C`, and leaves `fusion_speedup_B_over_C` null.
To isolate GEMM fusion, a further unfused baseline must retain GM handoffs while
matching the fused candidate's effective epilogue tile and ownership mapping.

Report device time separately from graph wall time and retain the raw samples,
target, compiler version, tile configuration, and actual kernel call counts.
Inspect generated TIR/source for the intermediate being eliminated. Additional
memory tracing is needed to measure actual GM traffic, rather than estimating
traffic from tensor sizes alone.

The [graph example](../../examples/ascend/graph/torch_compile.py) runs a real
Dynamo compilation, validates A/B/C, optionally records event and wall samples,
and exports generated sources. For example:

```bash
python examples/ascend/graph/torch_compile.py --bench --output /tmp/graph.json
python examples/ascend/graph/torch_compile.py --workload gemm_epilogue \
    --rows 128 --cols 192 --k 256 --dtype bfloat16 --bench
python examples/ascend/graph/torch_compile.py --target pto \
    --execution-backend pto --bench --output /tmp/pto-graph.json \
    --source-dir /tmp/pto-graph-sources
```

Event timing uses the profiler's cache-flush policy; wall timing does not flush
the cache. Compare each metric within its own boundary. A/B/C and the Dynamo
recording backend use the same requested target and execution backend. The JSON
records requested options, each kernel's normalized target/backend/execution
adapter, configuration, and raw samples. Sources use `.asc` for AscendC and
`.ptodsl.py` for PTO, based on the actual selected backend. Successful fusion relative
to independent TileLang kernels does not imply it outperforms Torch NPU eager.

Full Relax VM execution would require a separate Ascend runtime project:
device allocation/free, copies, device and stream context, asynchronous storage
reuse, and DLPack output ownership. Torch-managed graph execution does not
establish that those DeviceAPI operations work.
