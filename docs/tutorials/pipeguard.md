# Optional static checks with PipeGuard

PipeGuard checks memory and asynchronous synchronization hazards in a fixed
lowered kernel and launch configuration. It is an optional external Python
package; normal TileLang compilation and execution do not depend on it.

## Install

```bash
git submodule update --init 3rdparty/pipeguard
python -m pip install ./3rdparty/pipeguard
```

Use Python 3.10+ in the same environment as TileLang. The pinned dependency
contains the checker, capture adapters, and unit tests, without benchmark
corpora or evaluation results. Access to the PipeGuard repository is required.

## Quick start

Run the complete kernels in [`examples/pipeguard`](../../examples/pipeguard/README.md)
from the TileLang checkout:

```bash
python examples/pipeguard/check_kernels.py
```

Expected output (compiler logs may also appear):

```text
shared_safe: SAFE (expected SAFE)
shared_race: DATA_HAZARD (expected DATA_HAZARD)
tma_safe: SAFE (expected SAFE)
tma_missing_arrival: SYNC_HAZARD (expected SYNC_HAZARD)
mbarrier_reuse_safe: SAFE (expected SAFE)
mbarrier_reuse_race: SYNC_HAZARD (expected SYNC_HAZARD)
```

The shared-memory pair demonstrates cross-warp publication with and without a
CTA barrier. The TMA pair demonstrates correct completion observation and missing
thread arrivals. The mbarrier pair demonstrates phase reuse with and without an
ordered completion observation. The examples use a fixed SM90a target and one 128-thread CTA;
they lower and check the kernels without launching them. Automatic thread-storage
synchronization is disabled to preserve the intentionally missing barrier.

To inspect one case and its diagnostics:

```bash
python examples/pipeguard/check_kernels.py --case tma_missing_arrival --json
```

## Check your own kernel

Pass a `T.prim_func` to lower and check it without executing it:

```python
from tilelang.contrib.pipeguard import check

report = check(
    my_prim_func,
    target={"kind": "cuda", "arch": "sm_90a"},
    launch_grid=(16, 8, 1),  # the actual invocation
    selected_cta=(0, 0, 0),
    timeout=120,
)
print(report["verdict"])
print(report["diagnostics"])
```

Or check a compiled `JITKernel` using its retained IR and compilation settings:

```python
report = kernel.check_pipeguard(launch_grid=(16, 8, 1), timeout=120)
```

`check` also accepts a `CompiledArtifact` from `tilelang.lower`. Cached kernels
without retained IR must be checked through their original PrimFunc. Multi-kernel
artifacts are unsupported; each invocation needs an exact kernel/launch pairing.

Supply actual integer parameters with `scalar_values={"N": N}`. Indirect pointers
and constant integer inputs use `pointer_table_allocations` and `buffer_values`.
Callers must respect the IR's no-alias contract. Exact TMA shapes and strides are
recovered from host IR where possible; `descriptors` can supplement evidence but
cannot contradict that IR. Missing evidence returns `UNSUPPORTED`, without guessed
values. Launch dimensions must match the lowered kernel.

Capture and checking run in a subprocess with one shared deadline (120 seconds
by default). Timeout returns `INCONCLUSIVE` / `CAPTURE_CHECK_TIMEOUT`. The deadline
starts after lowering and IR serialization. No GPU kernel is launched.

Frontend mappings and protocol semantics are documented in the dependency's
[frontend guide](../../3rdparty/pipeguard/docs/frontends.md) and
[semantics guide](../../3rdparty/pipeguard/docs/semantics.md).

## Results and scope

- `SAFE`: no hazard found by the current model.
- `DATA_HAZARD` / `SYNC_HAZARD`: inspect the witness and launch bindings.
- `UNSUPPORTED`: required operations or exact evidence could not be modeled.
- `INCONCLUSIVE`: analysis did not conclude within its conditions or budget.
- `INTERNAL_ERROR`: infrastructure failed; not a safety result.

This pins the existing experimental PipeGuard 0.2 implementation. Known limitations
include collective execution/visibility aggregation and schedule-dependent phase
checks. A semantic refactor is separate. Do not treat SAFE as an independent proof.

The report's `scope` records the IR hash, launch grid, and selected CTA. It does not
cover cross-CTA races, numerical correctness, or later compiler transformations.
Final PTX register checking is not enabled. The target is the checker's SM90 subset;
CPU, Blackwell-only operations, and other unsupported constructs fail closed.
